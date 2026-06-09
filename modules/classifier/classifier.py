"""
키스트로크 감정 분류기 학습 및 평가 모듈

모델: XGBoost (기본 선택)
      SVM (--model svm 옵션 시 활성화)

평가: StratifiedKFold 5-fold 교차 검증
      지표: Accuracy, Macro F1, Per-class F1, Confusion Matrix
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

from preprocessing import build_feature_matrix


N_SPLITS = 5
RANDOM_STATE = 42


PARAM_DIST = {
    "n_estimators":     [100, 200, 300, 400],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "subsample":        [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha":        [0, 0.1, 0.5, 1.0],
    "reg_lambda":       [1.0, 1.5, 2.0, 3.0],
}


def tune_hyperparameters(
    X: pd.DataFrame, y: np.ndarray, n_classes: int, n_iter: int = 50
) -> dict:
    """
    RandomizedSearchCV로 XGBoost 하이퍼파라미터를 탐색한다.

    Macro F1을 최적화 기준으로 사용한다.
    sample_weight는 fit_params로 전달하여 불균형 보정을 유지한다.

    Parameters
    ----------
    X : pd.DataFrame
        피처 행렬.
    y : np.ndarray
        감정 레이블.
    n_classes : int
        감정 클래스 수.
    n_iter : int
        탐색할 파라미터 조합 수 (기본값: 50).

    Returns
    -------
    best_params : dict
        최적 하이퍼파라미터 딕셔너리.
    """
    base_model = XGBClassifier(
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    sample_weight = compute_sample_weight("balanced", y)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X.values if isinstance(X, pd.DataFrame) else X, y,
               sample_weight=sample_weight)

    print(f"[tune] 최적 Macro-F1 (CV): {search.best_score_:.4f}")
    print(f"[tune] 최적 파라미터: {search.best_params_}")
    return search.best_params_


def build_xgboost(n_classes: int, **kwargs) -> XGBClassifier:
    """
    XGBoost 분류기를 반환한다.

    kwargs로 하이퍼파라미터를 오버라이드할 수 있다.
    기본값은 tune_hyperparameters 미사용 시의 보수적 설정이다.
    eval_metric은 다중 분류에 적합한 mlogloss를 사용한다.
    """
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(kwargs)
    return XGBClassifier(**params)


def build_svm() -> Pipeline:
    """
    SVM 분류기를 반환한다.
    SVM은 피처 스케일에 민감하므로 StandardScaler를 파이프라인에 포함한다.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                    decision_function_shape="ovr",
                    random_state=RANDOM_STATE)),
    ])


def cross_validate(model, X: pd.DataFrame, y: np.ndarray) -> dict:
    """
    StratifiedKFold 교차 검증을 수행하고 결과 딕셔너리를 반환한다.

    Stratified 분할을 쓰는 이유: 감정 클래스 분포가 균등하지 않을 수 있으므로
    각 fold에서 클래스 비율을 유지해야 편향 없는 평가가 된다.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    fold_acc, fold_f1 = [], []
    all_y_true, all_y_pred = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y), 1):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        sample_weight = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_acc.append(acc)
        fold_f1.append(f1)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold_idx}: Acc={acc:.4f}  Macro-F1={f1:.4f}")

    results = {
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std":  float(np.std(fold_acc)),
        "f1_mean":  float(np.mean(fold_f1)),
        "f1_std":   float(np.std(fold_f1)),
        "y_true":   all_y_true,
        "y_pred":   all_y_pred,
    }
    return results


def print_report(results: dict, class_names: list[str]) -> None:
    print("\n" + "=" * 60)
    print("교차 검증 결과 요약")
    print("=" * 60)
    print(f"Accuracy  : {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
    print(f"Macro-F1  : {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print("\n분류 리포트 (전체 fold 통합):")
    print(classification_report(
        results["y_true"], results["y_pred"],
        target_names=class_names, zero_division=0
    ))


def plot_confusion_matrix(
    results: dict, class_names: list[str], save_path: str = "confusion_matrix.png"
) -> None:
    cm = confusion_matrix(results["y_true"], results["y_pred"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix (5-fold CV, aggregated)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[plot] confusion matrix 저장: {save_path}")
    plt.close()


def train_final_model(model, X: pd.DataFrame, y: np.ndarray):
    """
    교차 검증 완료 후 전체 데이터로 최종 모델을 학습한다.
    이 모델이 실제 추론 파이프라인에서 사용된다.
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    sample_weight = compute_sample_weight("balanced", y)
    model.fit(X_arr, y, sample_weight=sample_weight)
    print("[train] 전체 데이터 기반 최종 모델 학습 완료")
    return model


def save_model(model, le, path: str = "keystroke_classifier.pkl") -> None:
    """
    모델과 LabelEncoder를 함께 저장한다.

    추론 시 정수 인덱스를 감정 레이블로 복원하려면 LabelEncoder가 필요하다.
    """
    import pickle
    payload = {"model": model, "label_encoder": le}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save] 모델 저장: {path}")


def main(data_dir: str, model_type: str = "xgboost") -> None:
    print(f"[main] 데이터 로드 및 전처리 시작: {data_dir}")
    X, y, le = build_feature_matrix(data_dir)

    n_classes = len(le.classes_)
    class_names = list(le.classes_)

    print(f"\n[main] 모델: {model_type}  클래스 수: {n_classes}")
    if model_type == "xgboost":
        best_params = tune_hyperparameters(X, y, n_classes)
        model = build_xgboost(n_classes, **best_params)
    else:
        model = build_svm()

    print(f"\n[main] {N_SPLITS}-fold 교차 검증 시작")
    t0 = time.time()
    results = cross_validate(model, X, y)
    elapsed = time.time() - t0

    print_report(results, class_names)
    print(f"\n소요 시간: {elapsed:.1f}초")

    plot_confusion_matrix(results, class_names)

    if model_type == "xgboost":
        final_model = build_xgboost(n_classes, **best_params)
    else:
        final_model = build_svm()
    final_model = train_final_model(final_model, X, y)
    save_model(final_model, le)

    summary = {
        "model": model_type,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "accuracy_mean": results["acc_mean"],
        "accuracy_std": results["acc_std"],
        "macro_f1_mean": results["f1_mean"],
        "macro_f1_std": results["f1_std"],
        "classes": class_names,
    }
    with open("eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[save] 평가 요약 저장: eval_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmoSurv 키스트로크 감정 분류기")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="./data/emosurv",
        help="EmoSurv CSV 파일 디렉토리 경로 (기본값: ./data/emosurv)"
    )
    parser.add_argument(
        "--model", choices=["xgboost", "svm"], default="xgboost",
        help="분류기 종류 (기본값: xgboost)"
    )
    args = parser.parse_args()
    main(args.data_dir, args.model)
