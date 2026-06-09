"""
DDAMFN++ (벤더링) 성능 평가 + 시각화 — RAF-DB test.

ddamfn_rafdb_acc0.9204.pth 가중치의 성능 지표(Accuracy, Macro F1,
클래스별 P/R/F1, Confusion Matrix)를 측정하고, eval_affectnet.png 와
동일한 다크 테마 스타일의 PNG 리포트를 생성한다.

기본은 RAF-DB 네이티브 7-class 평가(파일명 0.9204 검증).
--mode spec6 를 주면 현재 파이프라인의 6-class 매핑으로도 평가한다.

실행:
    python eval_ddamfn_plot.py --data "/path/to/RAF-DB"      # test/1..7 구조
    python eval_ddamfn_plot.py --data data/rafdb --mode spec6
    python eval_ddamfn_plot.py --out checkpoints --batch 64

RAF-DB 데이터 구조 (숫자 폴더 1~7, *_aligned.jpg):
    <data>/test/1/  2/  3/  4/  5/  6/  7/
    1=Surprise 2=Fear 3=Disgust 4=Happy 5=Sad 6=Anger 7=Neutral
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix,
)

_HERE = Path(__file__).parent
sys.path.insert(0, str((_HERE / "ddamfn").resolve()))
from DDAM import DDAMNet  # noqa: E402

# ── 레이블 정의 ──────────────────────────────────────────────────────────────

# RAF-DB 폴더(1~7) → 감정
FOLDER_EMO = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happy",
              "5": "sad", "6": "angry", "7": "neutral"}

# model 출력 인덱스 → 감정 (DDAMFN RAF-DB 학습 순서). 두 가설을 자동 검증한다.
MAP_A = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]
MAP_B = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# 7-class 표시 순서(RAF-DB 표준)
RAF7 = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# 6-class spec (현재 파이프라인과 동일). disgust→angry 병합.
SPEC6 = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]
EMO_TO_SPEC = {"surprise": "surprised", "fear": "fearful", "disgust": "angry",
               "happy": "happy", "sad": "sad", "angry": "angry", "neutral": "neutral"}

TFM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── 모델 / 데이터 로드 ────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = DDAMNet(num_class=7, num_head=2, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[로드] missing={len(missing)} unexpected={len(unexpected)} "
              f"(missing={missing[:3]}, unexpected={unexpected[:3]})")
    else:
        print("[로드] 가중치 완전 일치 (missing=0, unexpected=0)")
    return model.to(device).eval()


def load_samples(data_root: str):
    test_dir = Path(data_root) / "test"
    if not test_dir.exists():
        sys.exit(
            f"[오류] 테스트셋을 찾을 수 없습니다: {test_dir}\n"
            f"  RAF-DB 를 <data>/test/1 .. 7 (각 *_aligned.jpg) 구조로 배치하세요."
        )
    samples = []
    found = []
    for d in sorted(test_dir.iterdir()):
        if d.is_dir() and d.name in FOLDER_EMO:
            found.append(d.name)
            for f in d.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    samples.append((str(f), FOLDER_EMO[d.name]))
    if not samples:
        sys.exit(f"[오류] {test_dir} 에서 이미지를 찾지 못했습니다.")
    print(f"[데이터] RAF-DB test — 폴더 {found}, 총 {len(samples)}장")
    return samples


# ── 추론 (배치) ───────────────────────────────────────────────────────────────

def run_eval(model, samples, device, batch_size: int):
    true_emo, pred_idx, confs = [], [], []
    total = len(samples)
    buf_t, buf_emo = [], []

    def flush():
        if not buf_t:
            return
        batch = torch.stack(buf_t).to(device)
        with torch.no_grad():
            out, _, _ = model(batch)
            p = torch.softmax(out, dim=1).cpu().numpy()
        pred_idx.extend(p.argmax(1).tolist())
        confs.extend(p.max(1).tolist())
        true_emo.extend(buf_emo)
        buf_t.clear(); buf_emo.clear()

    for i, (path, emo) in enumerate(samples):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        buf_t.append(TFM(img)); buf_emo.append(emo)
        if len(buf_t) >= batch_size:
            flush()
            print(f"  추론 중... {i+1}/{total}", end="\r", flush=True)
    flush()
    print(f"  추론 완료: {len(pred_idx)}/{total}장        ")
    return true_emo, np.array(pred_idx), np.array(confs)


def pick_mapping(true_emo, pred_idx):
    accs = {}
    for name, mp in (("A", MAP_A), ("B", MAP_B)):
        pred7 = [mp[k] for k in pred_idx]
        accs[name] = accuracy_score(true_emo, pred7)
    best = "A" if accs["A"] >= accs["B"] else "B"
    print(f"[매핑] 자동 검증 A={accs['A']*100:.2f}%  B={accs['B']*100:.2f}%  → 채택 {best}")
    return MAP_A if best == "A" else MAP_B


# ── 시각화 (eval_affectnet.png 스타일) ───────────────────────────────────────

def make_report(labels, preds, class_labels, out_path: Path,
                model_name: str, dataset_name: str):
    n_cls = len(class_labels)
    idx = list(range(n_cls))
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", labels=idx, zero_division=0)
    prec     = precision_score(labels, preds, average=None, labels=idx, zero_division=0)
    rec      = recall_score(labels, preds, average=None, labels=idx, zero_division=0)
    f1       = f1_score(labels, preds, average=None, labels=idx, zero_division=0)
    cm       = confusion_matrix(labels, preds, labels=idx)

    # 터미널 출력
    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  모델   : {model_name}")
    print(f"  데이터 : {dataset_name}  ({len(labels)}장)")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(sep)
    print(f"  {'클래스':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'샘플수':>7}")
    print(f"  {'-'*50}")
    for i, lbl in enumerate(class_labels):
        n = int((np.array(labels) == i).sum())
        print(f"  {lbl:<12} {prec[i]:>10.4f} {rec[i]:>8.4f} {f1[i]:>8.4f} {n:>7}")
    print(sep)

    disp = [c.capitalize() for c in class_labels]

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # 왼쪽: Confusion Matrix (행 정규화 색 + 실제 개수 주석)
    ax_cm = fig.add_subplot(gs[0])
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.maximum(row_sum, 1)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d",
        cmap="Blues", linewidths=0.4, linecolor="#2a2d36",
        xticklabels=disp, yticklabels=disp,
        ax=ax_cm, cbar_kws={"shrink": 0.8}, annot_kws={"size": 10},
    )
    ax_cm.set_facecolor("#0f1117")
    ax_cm.set_xlabel("Predicted", color="white", fontsize=11, labelpad=8)
    ax_cm.set_ylabel("True", color="white", fontsize=11, labelpad=8)
    ax_cm.set_title("Confusion Matrix", color="white", fontsize=13, pad=12)
    ax_cm.tick_params(colors="white")
    for spine in ax_cm.spines.values():
        spine.set_edgecolor("#2a2d36")
    plt.setp(ax_cm.get_xticklabels(), rotation=30, ha="right", color="white", fontsize=9)
    plt.setp(ax_cm.get_yticklabels(), rotation=0, color="white", fontsize=9)
    cbar = ax_cm.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # 오른쪽: 클래스별 F1 바 차트
    ax_f1 = fig.add_subplot(gs[1])
    ax_f1.set_facecolor("#0f1117")
    colors = ["#4c9be8" if v >= macro_f1 else "#c0392b" for v in f1]
    bars = ax_f1.barh(disp, f1, color=colors, edgecolor="#2a2d36", height=0.55)
    ax_f1.invert_yaxis()
    ax_f1.axvline(macro_f1, color="#f39c12", linestyle="--", linewidth=1.4,
                  label=f"Macro F1 = {macro_f1:.3f}")
    ax_f1.set_xlim(0, 1.0)
    ax_f1.set_xlabel("F1 Score", color="white", fontsize=11)
    ax_f1.set_title("Per-class F1 Score", color="white", fontsize=13, pad=12)
    ax_f1.tick_params(colors="white")
    ax_f1.legend(facecolor="#1c1f26", edgecolor="#2a2d36", labelcolor="white", fontsize=10)
    for spine in ax_f1.spines.values():
        spine.set_edgecolor("#2a2d36")
    plt.setp(ax_f1.get_yticklabels(), color="white", fontsize=10)
    plt.setp(ax_f1.get_xticklabels(), color="white")
    for bar, val in zip(bars, f1):
        ax_f1.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", ha="left", color="white", fontsize=9)

    fig.suptitle(
        f"{model_name}  ·  {dataset_name}\n"
        f"Accuracy: {acc*100:.2f}%   |   Macro F1: {macro_f1:.4f}"
        f"   |   Test samples: {len(labels)}",
        color="white", fontsize=12, y=1.02,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  시각화 저장: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DDAMFN++ RAF-DB 성능 평가 + PNG 리포트")
    ap.add_argument("--data", default="data/rafdb", help="RAF-DB 루트 (test/1..7)")
    ap.add_argument("--model", default="checkpoints/ddamfn_rafdb_acc0.9204.pth")
    ap.add_argument("--out", default="checkpoints", help="PNG 저장 폴더")
    ap.add_argument("--mode", choices=["raf7", "spec6"], default="raf7",
                    help="raf7=네이티브 7-class(기본), spec6=파이프라인 6-class 매핑")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    model = load_model(Path(args.model), device)
    samples = load_samples(args.data)
    true_emo, pred_idx, _ = run_eval(model, samples, device, args.batch)
    best_map = pick_mapping(true_emo, pred_idx)
    pred_emo = [best_map[k] for k in pred_idx]

    out_dir = Path(args.out)
    if args.mode == "raf7":
        order = {e: i for i, e in enumerate(RAF7)}
        labels = [order[e] for e in true_emo]
        preds = [order[e] for e in pred_emo]
        make_report(labels, preds, RAF7, out_dir / "eval_ddamfn_rafdb.png",
                    "DDAMFN++ (rafdb ckpt, acc0.9204)", "RAF-DB test set (7-class)")
    else:
        order = {e: i for i, e in enumerate(SPEC6)}
        labels = [order[EMO_TO_SPEC[e]] for e in true_emo]
        preds = [order[EMO_TO_SPEC[e]] for e in pred_emo]
        make_report(labels, preds, SPEC6, out_dir / "eval_ddamfn_spec6.png",
                    "DDAMFN++ (rafdb ckpt, acc0.9204)", "RAF-DB test set (spec 6-class)")


if __name__ == "__main__":
    main()
