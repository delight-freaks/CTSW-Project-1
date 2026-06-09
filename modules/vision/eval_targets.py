"""
현재 모델(FER_static_ResNet50_AffectNet)을 재학습 없이 어디까지 끌어올릴 수 있는지
RAF-DB test에서 측정한다. 한 번 추론하고 여러 타깃/조건으로 지표를 계산:

  1. 6-class 전체 (baseline)
  2. confidence 게이팅: max prob >= thr 인 샘플만 채택했을 때 정확도 + 커버리지
  3. 부정정서 이진 탐지: {sad, angry, fearful} vs {happy, neutral, surprised}
     (상담 시스템의 실제 핵심 신호 = "사용자가 부정정서 상태인가")

실행: python eval_targets.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

SPEC_LABELS = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]
LABEL_TO_IDX = {l: i for i, l in enumerate(SPEC_LABELS)}
NEGATIVE = {LABEL_TO_IDX["sad"], LABEL_TO_IDX["angry"], LABEL_TO_IDX["fearful"]}

RAFDB_TO_SPEC = {
    "1": "surprised", "2": "fearful", "3": "angry", "4": "happy",
    "5": "sad", "6": "angry", "7": "neutral",
    "surprise": "surprised", "fear": "fearful", "disgust": "angry",
    "happiness": "happy", "sadness": "sad", "anger": "angry", "neutral": "neutral",
}
_AFFECTNET_TO_SPEC = [3, 2, 4, 5, 1, 0, 0]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),
    transforms.Normalize(mean=[131.0912, 103.8827, 91.4953], std=[1.0, 1.0, 1.0]),
])


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    def remap(sd):
        new = {}
        for k, v in sd.items():
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

    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 512), nn.ReLU(), nn.Linear(512, 7))
    m.load_state_dict(remap(ckpt))
    return m.to(device).eval()


def load_samples(data_root):
    test_dir = Path(data_root) / "test"
    samples = []
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        spec = RAFDB_TO_SPEC.get(cls_dir.name.lower())
        if spec is None:
            continue
        idx = LABEL_TO_IDX[spec]
        for f in os.listdir(cls_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((str(cls_dir / f), idx))
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("checkpoints/FER_static_ResNet50_AffectNet.pt", device)
    samples = load_samples("data/rafdb")
    print(f"[데이터] RAF-DB test {len(samples)}장 / device={device}")

    labels, preds, confs = [], [], []
    for i, (path, lab) in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(samples)}", end="\r", flush=True)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        t = TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            p7 = torch.softmax(model(t), dim=1).squeeze().cpu().numpy()
        spec = np.zeros(len(SPEC_LABELS))
        for a, s in enumerate(_AFFECTNET_TO_SPEC):
            spec[s] += p7[a]
        labels.append(lab)
        preds.append(int(np.argmax(spec)))
        confs.append(float(spec.max()))

    labels = np.array(labels); preds = np.array(preds); confs = np.array(confs)
    sep = "═" * 60

    print(f"\n{sep}\n  1) 6-class 전체")
    print(f"     Accuracy {accuracy_score(labels, preds)*100:5.2f}%  "
          f"Macro-F1 {f1_score(labels, preds, average='macro', zero_division=0):.3f}")

    print(f"{sep}\n  2) confidence 게이팅 (max prob >= thr 인 샘플만 채택)")
    print(f"     {'thr':>5} {'coverage':>9} {'accuracy':>9}")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        mask = confs >= thr
        cov = mask.mean() * 100
        acc = accuracy_score(labels[mask], preds[mask]) * 100 if mask.any() else 0.0
        print(f"     {thr:>5.1f} {cov:>8.1f}% {acc:>8.2f}%")

    print(f"{sep}\n  3) 부정정서 이진 탐지  {{sad,angry,fearful}} vs 그 외")
    y_true = np.array([1 if l in NEGATIVE else 0 for l in labels])
    y_pred = np.array([1 if p in NEGATIVE else 0 for p in preds])
    print(f"     Accuracy {accuracy_score(y_true, y_pred)*100:5.2f}%  "
          f"F1(neg) {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"     + confidence 게이팅:")
    print(f"     {'thr':>5} {'coverage':>9} {'accuracy':>9}")
    for thr in (0.0, 0.6, 0.7, 0.8):
        mask = confs >= thr
        if mask.any():
            acc = accuracy_score(y_true[mask], y_pred[mask]) * 100
            print(f"     {thr:>5.1f} {mask.mean()*100:>8.1f}% {acc:>8.2f}%")
    print(sep)


if __name__ == "__main__":
    main()
