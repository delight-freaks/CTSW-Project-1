"""
DDAMFN++ (벤더링) 성능 평가 — RAF-DB test.
현재 FER_static 모델(spec-6 47.3%)과 '동일 조건'으로 비교하고,
DDAMFN의 7-class in-distribution 성능(논문 92.04%)도 재현 확인한다.

라벨 순서 모호성 회피: model 출력 인덱스→감정 매핑 두 가설(A/B)을 모두 계산해
정확도가 높은(=올바른) 쪽을 자동 채택한다.

실행: python eval_ddamfn.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "ddamfn"))
from DDAM import DDAMNet  # noqa: E402

CKPT = _HERE / "checkpoints" / "ddamfn_rafdb_acc0.9204.pth"

# spec 6-class (현재 파이프라인과 동일)
SPEC = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]
SPEC_IDX = {s: i for i, s in enumerate(SPEC)}
NEGATIVE = {SPEC_IDX["sad"], SPEC_IDX["angry"], SPEC_IDX["fearful"]}

# RAF-DB 폴더(1~7, 표준) → 감정
FOLDER_EMO = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happy",
              "5": "sad", "6": "angry", "7": "neutral"}
EMO_TO_SPEC = {"surprise": "surprised", "fear": "fearful", "disgust": "angry",
               "happy": "happy", "sad": "sad", "angry": "angry", "neutral": "neutral"}

# model 출력 인덱스 → 감정 두 가설
MAP_A = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]  # 문서 class_names
MAP_B = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]  # RAF-DB 표준 정렬

TFM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_samples(root):
    test_dir = Path(root) / "test"
    out = []
    for d in sorted(test_dir.iterdir()):
        if d.is_dir() and d.name in FOLDER_EMO:
            for f in d.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    out.append((str(f), FOLDER_EMO[d.name]))
    return out


def main():
    device = torch.device("cpu")
    model = DDAMNet(num_class=7, num_head=2, pretrained=False)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[로드] missing={len(missing)} unexpected={len(unexpected)} "
              f"(예시 missing={missing[:3]}, unexpected={unexpected[:3]})")
    model.to(device).eval()

    samples = load_samples("data/rafdb")
    print(f"[데이터] RAF-DB test {len(samples)}장 / model=DDAMFN++ (rafdb ckpt)")

    true_emo, pred_idx, confs = [], [], []
    for i, (path, emo) in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(samples)}", end="\r", flush=True)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        t = TFM(img).unsqueeze(0)
        with torch.no_grad():
            out, _, _ = model(t)
            p = torch.softmax(out, dim=1).squeeze().cpu().numpy()
        true_emo.append(emo)
        pred_idx.append(int(np.argmax(p)))
        confs.append(float(p.max()))

    pred_idx = np.array(pred_idx); confs = np.array(confs)
    sep = "═" * 60

    # --- 라벨 매핑 자동 채택: 7-class 감정 정확도가 높은 쪽 ---
    def norm(e):  # happy/happiness 등 정규화는 불필요(동일 키 사용)
        return e
    true7 = [norm(e) for e in true_emo]
    accs = {}
    for name, mp in (("A", MAP_A), ("B", MAP_B)):
        pred7 = [mp[k] for k in pred_idx]
        accs[name] = accuracy_score(true7, pred7)
    best = "A" if accs["A"] >= accs["B"] else "B"
    best_map = MAP_A if best == "A" else MAP_B

    print(f"\n{sep}\n  라벨 매핑 자동 검증: A={accs['A']*100:.2f}%  B={accs['B']*100:.2f}%  → 채택: {best}")
    print(f"{sep}\n  1) 7-class 감정 정확도 (in-distribution, 논문 92.04% 재현 확인)")
    pred7 = [best_map[k] for k in pred_idx]
    print(f"     Accuracy {accuracy_score(true7, pred7)*100:5.2f}%  "
          f"Macro-F1 {f1_score(true7, pred7, average='macro', zero_division=0):.3f}")

    # --- spec-6 (현재 모델과 동일 조건 비교) ---
    y_true6 = np.array([SPEC_IDX[EMO_TO_SPEC[e]] for e in true_emo])
    y_pred6 = np.array([SPEC_IDX[EMO_TO_SPEC[best_map[k]]] for k in pred_idx])
    print(f"{sep}\n  2) spec 6-class (현재 FER_static 모델 47.33%와 동일 조건)")
    print(f"     Accuracy {accuracy_score(y_true6, y_pred6)*100:5.2f}%  "
          f"Macro-F1 {f1_score(y_true6, y_pred6, average='macro', zero_division=0):.3f}")
    print(f"     클래스별 F1:")
    f1s = f1_score(y_true6, y_pred6, average=None, labels=list(range(6)), zero_division=0)
    for i, s in enumerate(SPEC):
        n = int((y_true6 == i).sum())
        print(f"       {s:<10} F1 {f1s[i]:.3f}  (n={n})")

    print(f"{sep}\n  3) confidence 게이팅 (spec-6)")
    print(f"     {'thr':>5} {'coverage':>9} {'accuracy':>9}")
    for thr in (0.0, 0.5, 0.7, 0.9):
        m = confs >= thr
        a = accuracy_score(y_true6[m], y_pred6[m]) * 100 if m.any() else 0.0
        print(f"     {thr:>5.1f} {m.mean()*100:>8.1f}% {a:>8.2f}%")

    print(f"{sep}\n  4) 부정정서 이진 탐지 {{sad,angry,fearful}} vs 그 외")
    yt = np.array([1 if v in NEGATIVE else 0 for v in y_true6])
    yp = np.array([1 if v in NEGATIVE else 0 for v in y_pred6])
    print(f"     Accuracy {accuracy_score(yt, yp)*100:5.2f}%  F1(neg) {f1_score(yt, yp, zero_division=0):.3f}")
    print(sep)


if __name__ == "__main__":
    main()
