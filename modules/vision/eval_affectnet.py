"""
FER_static_ResNet50_AffectNet 모델 성능 평가 스크립트
RAF-DB 테스트셋 기준 Accuracy, Macro F1, Confusion Matrix 생성

실행:
    python eval_affectnet.py
    python eval_affectnet.py --data data/rafdb --model checkpoints/FER_static_ResNet50_AffectNet.pt
    python eval_affectnet.py --out results/

RAF-DB 데이터 구조 (두 가지 형식 모두 자동 감지):
    data/rafdb/test/Surprise/  Fear/  Disgust/  Happiness/  Sadness/  Anger/  Neutral/
    data/rafdb/test/1/  2/  3/  4/  5/  6/  7/          (숫자 폴더 형식)
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
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

# ── 레이블 정의 ──────────────────────────────────────────────────────────────

SPEC_LABELS = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]
LABEL_TO_IDX = {label: i for i, label in enumerate(SPEC_LABELS)}

# RAF-DB 폴더명 → spec 6-class (대소문자 무관, 숫자 폴더 모두 지원)
# 숫자: 1=Surprise 2=Fear 3=Disgust 4=Happiness 5=Sadness 6=Anger 7=Neutral
RAFDB_TO_SPEC = {
    "surprise":  "surprised",  "1": "surprised",
    "fear":      "fearful",    "2": "fearful",
    "disgust":   "angry",      "3": "angry",
    "happiness": "happy",      "4": "happy",
    "sadness":   "sad",        "5": "sad",
    "anger":     "angry",      "6": "angry",
    "neutral":   "neutral",    "7": "neutral",
}

# AffectNet 7-class 출력 인덱스 → spec 6-class 인덱스
# AffectNet 순서: neutral(0) happy(1) sad(2) surprise(3) fear(4) disgust(5) anger(6)
# disgust(5) → angry(0) 합산
_AFFECTNET_TO_SPEC = [3, 2, 4, 5, 1, 0, 0]


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_fer_static_model(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    def remap(state_dict):
        new = {}
        for k, v in state_dict.items():
            if k.startswith("fc1."):
                k = "fc.0." + k[4:]
            elif k.startswith("fc2."):
                k = "fc.2." + k[4:]
            else:
                k = k.replace("conv_layer_s2_same.", "conv1.")
                if k.startswith("batch_norm1."):
                    k = "bn1." + k[12:]
                k = re.sub(r"(layer\d+\.\d+\.)batch_norm(\d+)\.", r"\1bn\2.", k)
                k = k.replace(".i_downsample.", ".downsample.")
            new[k] = v
        return new

    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Sequential(
        nn.Linear(backbone.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 7),
    )
    backbone.load_state_dict(remap(ckpt))
    backbone.to(device).eval()
    return backbone


# ── 전처리 (VGGFace2 정규화) ─────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),
    transforms.Normalize(mean=[131.0912, 103.8827, 91.4953], std=[1.0, 1.0, 1.0]),
])


# ── 추론 ─────────────────────────────────────────────────────────────────────

def predict(model: nn.Module, img: Image.Image, device: torch.device) -> int:
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

    spec_probs = np.zeros(len(SPEC_LABELS))
    for affectnet_idx, spec_idx in enumerate(_AFFECTNET_TO_SPEC):
        spec_probs[spec_idx] += probs[affectnet_idx]

    return int(np.argmax(spec_probs))


# ── RAF-DB 데이터 로드 ────────────────────────────────────────────────────────

def load_rafdb_test(data_root: str):
    test_dir = Path(data_root) / "test"
    if not test_dir.exists():
        sys.exit(
            f"[오류] 테스트셋 경로를 찾을 수 없습니다: {test_dir}\n"
            f"  RAF-DB 데이터를 아래 구조로 배치해주세요:\n"
            f"  {data_root}/test/Surprise/, Fear/, Disgust/, Happiness/, Sadness/, Anger/, Neutral/\n"
            f"  또는 숫자 폴더: {data_root}/test/1/, 2/, 3/, 4/, 5/, 6/, 7/"
        )

    samples = []
    found_classes = []
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        key = cls_dir.name.lower()
        spec_cls = RAFDB_TO_SPEC.get(key)
        if spec_cls is None:
            continue
        label_idx = LABEL_TO_IDX[spec_cls]
        found_classes.append(cls_dir.name)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((str(cls_dir / fname), label_idx))

    if not samples:
        sys.exit(
            f"[오류] {test_dir} 에서 이미지를 찾지 못했습니다.\n"
            f"  발견된 폴더: {list(test_dir.iterdir())}"
        )

    print(f"[데이터] RAF-DB test — 감지된 클래스: {found_classes}")
    print(f"[데이터] 총 샘플 수: {len(samples)}장")
    return samples


# ── 평가 실행 ─────────────────────────────────────────────────────────────────

def run_eval(model, samples, device):
    all_preds, all_labels = [], []
    total = len(samples)

    for i, (path, label) in enumerate(samples):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  추론 중... {i+1}/{total}", end="\r", flush=True)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        pred = predict(model, img, device)
        all_preds.append(pred)
        all_labels.append(label)

    print(f"  추론 완료: {len(all_preds)}장        ")
    return np.array(all_labels), np.array(all_preds)


# ── 시각화 ───────────────────────────────────────────────────────────────────

def make_report(labels, preds, out_dir: Path, model_name: str, dataset_name: str):
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    prec     = precision_score(labels, preds, average=None, zero_division=0)
    rec      = recall_score(labels, preds, average=None, zero_division=0)
    f1       = f1_score(labels, preds, average=None, zero_division=0)
    cm       = confusion_matrix(labels, preds)

    # 터미널 출력
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  모델   : {model_name}")
    print(f"  데이터 : {dataset_name}  ({len(labels)}장)")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(sep)
    print(f"  {'클래스':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'샘플수':>7}")
    print(f"  {'-'*50}")
    for i, lbl in enumerate(SPEC_LABELS):
        n = int((labels == i).sum())
        print(f"  {lbl:<12} {prec[i]:>10.4f} {rec[i]:>8.4f} {f1[i]:>8.4f} {n:>7}")
    print(sep)

    # ── 그래프 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # 왼쪽: Confusion Matrix
    ax_cm = fig.add_subplot(gs[0])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d",
        cmap="Blues", linewidths=0.4, linecolor="#2a2d36",
        xticklabels=SPEC_LABELS, yticklabels=SPEC_LABELS,
        ax=ax_cm, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 11},
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
    ax_cm.collections[0].colorbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(ax_cm.collections[0].colorbar.ax.yaxis.get_ticklabels(), color="white")

    # 오른쪽: 클래스별 F1 바 차트
    ax_f1 = fig.add_subplot(gs[1])
    ax_f1.set_facecolor("#0f1117")
    colors = ["#4c9be8" if v >= macro_f1 else "#c0392b" for v in f1]
    bars = ax_f1.barh(SPEC_LABELS, f1, color=colors, edgecolor="#2a2d36", height=0.55)
    ax_f1.axvline(macro_f1, color="#f39c12", linestyle="--", linewidth=1.4,
                  label=f"Macro F1 = {macro_f1:.3f}")
    ax_f1.set_xlim(0, 1.0)
    ax_f1.set_xlabel("F1 Score", color="white", fontsize=11)
    ax_f1.set_title("Per-class F1 Score", color="white", fontsize=13, pad=12)
    ax_f1.tick_params(colors="white")
    ax_f1.legend(facecolor="#1c1f26", edgecolor="#2a2d36",
                 labelcolor="white", fontsize=10)
    for spine in ax_f1.spines.values():
        spine.set_edgecolor("#2a2d36")
    plt.setp(ax_f1.get_yticklabels(), color="white", fontsize=10)
    plt.setp(ax_f1.get_xticklabels(), color="white")
    for bar, val in zip(bars, f1):
        ax_f1.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", ha="left",
                   color="white", fontsize=9)

    fig.suptitle(
        f"{model_name}  ·  {dataset_name}\n"
        f"Accuracy: {acc*100:.2f}%   |   Macro F1: {macro_f1:.4f}"
        f"   |   Test samples: {len(labels)}",
        color="white", fontsize=12, y=1.02,
    )

    out_path = out_dir / "eval_affectnet.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  시각화 저장: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FER_static AffectNet 모델 성능 평가 (RAF-DB)")
    parser.add_argument("--data",  default="data/rafdb",
                        help="RAF-DB 루트 경로 (기본: data/rafdb)")
    parser.add_argument("--model", default="checkpoints/FER_static_ResNet50_AffectNet.pt",
                        help="체크포인트 경로")
    parser.add_argument("--out",   default="checkpoints",
                        help="결과 이미지 저장 폴더 (기본: checkpoints)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    print(f"[모델] {args.model} 로드 중...")
    model = load_fer_static_model(args.model, device)

    samples = load_rafdb_test(args.data)
    labels, preds = run_eval(model, samples, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_report(
        labels, preds, out_dir,
        model_name="FER_static_ResNet50_AffectNet",
        dataset_name="RAF-DB test set",
    )


if __name__ == "__main__":
    main()
