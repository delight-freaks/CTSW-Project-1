"""
EmoSurv 데이터셋 전처리 모듈

EmoSurv 파일 구성:
  fixed_text.csv   : UserID, EmotionIndex, Index, KeyCode, KeyDown, KeyUp,
                     D1U1, D1U2, D1D2, U1D2, U1U2, D1U3, D1D3, Answer
  free_text.csv    : 동일 컬럼 구조
  frequency.csv    : UserID, textIndex, EmotionIndex, DelFreq, LeftFreq, TotTime
  participants.csv : 인구통계 (본 모듈에서는 미사용)

전처리 전략:
  keystroke 행 단위 데이터(fixed/free)를 (UserID, EmotionIndex, TextType) 기준으로
  집계하여 세션 수준 피처 벡터를 생성한 뒤 frequency의 빈도 특성과 병합한다.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


EMOTION_MAP = {
    "H": "happy",
    "S": "sad",
    "A": "angry",
    "C": "calm",
    "N": "neutral",
}

TIMING_FEATURES = ["d1u1", "d1d2", "u1d2", "u1u2"]

AGG_FUNCS = ["mean", "std", "median", "min", "max"]


FILE_NAMES = {
    "fixed": "fixed_text.csv",
    "free":  "free_text.csv",
    "freq":  "frequency.csv",
}


def load_raw_files(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    EmoSurv CSV 파일을 로드한다.

    Parameters
    ----------
    data_dir : str
        fixed_text.csv, free_text.csv, frequency.csv 가 위치한 디렉토리 경로.

    Returns
    -------
    df_fixed, df_free, df_freq : (DataFrame, DataFrame, DataFrame)
    """
    loaded = {}
    for key, name in FILE_NAMES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{name}' 파일을 찾을 수 없습니다. (data_dir='{data_dir}')"
            )
        loaded[key] = pd.read_csv(path, sep=";")
        print(f"[load] {key}: {path}  ({len(loaded[key])} rows)")

    return loaded["fixed"], loaded["free"], loaded["freq"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명 공백·대소문자를 정규화한다."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    col_map = {
        "userid":       "user_id",
        "emotionindex": "emotion_index",
        "keycode":      "key_code",
        "keydown":      "key_down",
        "keyup":        "key_up",
        "textindex":    "text_index",
        "delfreq":      "del_freq",
        "leftfreq":     "left_freq",
        "tottime":      "tot_time",
    }
    df = df.rename(columns=col_map)
    return df


def _clean_timing(df: pd.DataFrame) -> pd.DataFrame:
    """
    타이밍 피처의 이상치를 제거한다.
    - 음수 값은 측정 오류이므로 NaN 처리
    - 5초(5000ms) 초과는 의도적 중단으로 간주하여 상한 클리핑
    """
    for col in TIMING_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df[col].clip(upper=5000)
    return df


def _aggregate_session(df: pd.DataFrame, text_type: str) -> pd.DataFrame:
    """
    keystroke 행 단위 DataFrame을 세션 단위로 집계한다.

    집계 기준: (user_id, emotion_index)
    생성 피처: 각 타이밍 피처 x {mean, std, median, min, max}
    """
    df = df.copy()
    df["text_type"] = text_type

    agg_dict = {}
    for feat in TIMING_FEATURES:
        if feat in df.columns:
            agg_dict[feat] = AGG_FUNCS

    session_df = (
        df.groupby(["user_id", "emotion_index"])
        .agg(agg_dict)
        .reset_index()
    )

    # 멀티레벨 컬럼 → 단일 문자열
    session_df.columns = [
        "_".join(filter(None, col)).strip() if isinstance(col, tuple) else col
        for col in session_df.columns
    ]
    session_df["text_type"] = text_type
    return session_df


def build_feature_matrix(
    data_dir: str,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    전체 전처리 파이프라인을 실행하여 (X, y, label_encoder)를 반환한다.

    Parameters
    ----------
    data_dir : str
        EmoSurv CSV 파일 디렉토리.

    Returns
    -------
    X : pd.DataFrame
        세션 단위 피처 매트릭스.
    y : np.ndarray
        정수 인코딩된 감정 레이블.
    le : LabelEncoder
        le.classes_ 로 원래 레이블 복원 가능.
    """
    df_fixed, df_free, df_freq = load_raw_files(data_dir)

    df_fixed = _standardize_columns(df_fixed)
    df_free  = _standardize_columns(df_free)
    df_freq  = _standardize_columns(df_freq)

    df_fixed = _clean_timing(df_fixed)
    df_free  = _clean_timing(df_free)

    sess_fixed = _aggregate_session(df_fixed, "fixed")
    sess_free  = _aggregate_session(df_free,  "free")
    sessions   = pd.concat([sess_fixed, sess_free], ignore_index=True)

    # File3 병합: DelFreq, LeftFreq, TotTime
    freq_agg = (
        df_freq.groupby(["user_id", "emotion_index"])
        .agg(
            del_freq_mean=("del_freq", "mean"),
            left_freq_mean=("left_freq", "mean"),
            tot_time_mean=("tot_time", "mean"),
        )
        .reset_index()
    )
    sessions = sessions.merge(
        freq_agg, on=["user_id", "emotion_index"], how="left"
    )

    # 감정 레이블 매핑 및 인코딩
    sessions["emotion_label"] = sessions["emotion_index"].map(EMOTION_MAP)
    missing = sessions["emotion_label"].isna().sum()
    if missing > 0:
        print(f"[warn] emotion_index 매핑 실패 {missing}건 — 해당 행 제거")
        sessions = sessions.dropna(subset=["emotion_label"])

    feature_cols = [
        c for c in sessions.columns
        if c not in ("user_id", "emotion_index", "emotion_label", "text_type")
    ]

    X = sessions[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    le = LabelEncoder()
    y = le.fit_transform(sessions["emotion_label"])

    print(f"[build] 세션 수: {len(X)}  피처 수: {X.shape[1]}")
    print(f"[build] 클래스 분포:\n{pd.Series(le.inverse_transform(y)).value_counts()}")

    return X, y, le


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/emosurv"
    X, y, le = build_feature_matrix(data_dir)
    print(f"\n피처 목록:\n{list(X.columns)}")
