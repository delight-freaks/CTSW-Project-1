"""
키스트로크 감정 추론 모듈

키스트로크 로거(조재현, B)로부터 raw 이벤트 JSON을 받아
학습된 모델로 감정을 예측하고 interface_spec.md 스펙에 맞는 JSON을 반환한다.

입력 (Module 2 - Keystroke Logger Raw Output):
    {
      "session_id": "...",
      "turn_id": 3,
      "events": [
        {"type": "keydown", "key": "a", "timestamp": 1710234560.001, "is_delete": false},
        {"type": "keyup",   "key": "a", "timestamp": 1710234560.087, "is_delete": false},
        ...
      ]
    }

출력 (Module 2 - Keystroke Classifier Output):
    {
      "session_id": "...",
      "turn_id": 3,
      "emotion": "sad",
      "confidence": 0.61,
      "avg_iki_ms": 2300.0,
      "backspace_rate": 0.14
    }
"""

import pickle
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "keystroke_classifier.pkl"

FEATURE_COLS = [
    "d1u1_mean", "d1u1_std", "d1u1_median", "d1u1_min", "d1u1_max",
    "d1d2_mean", "d1d2_std", "d1d2_median", "d1d2_min", "d1d2_max",
    "u1d2_mean", "u1d2_std", "u1d2_median", "u1d2_min", "u1d2_max",
    "u1u2_mean", "u1u2_std", "u1u2_median", "u1u2_min", "u1u2_max",
    "del_freq_mean", "left_freq_mean", "tot_time_mean",
]

# 타이밍 이상치 상한 (ms) — 학습 전처리와 동일한 기준
TIMING_CLIP_MS = 5000.0


def _load_model() -> tuple:
    """
    학습된 분류기와 LabelEncoder를 로드한다.

    Returns
    -------
    model, label_encoder
        XGBClassifier(또는 Pipeline)와 감정 레이블 복원용 LabelEncoder.
    """
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["label_encoder"]


def _extract_features(events: list[dict]) -> pd.DataFrame:
    """
    raw 이벤트 배열에서 모델 입력 피처 벡터를 계산한다.

    타이밍 피처(D1U1, D1D2, U1D2, U1U2)는 연속된 두 키 쌍(bigram)에서 계산하며
    삭제·방향 키는 타이밍 계산에서 제외하고 빈도 피처로만 반영한다.

    Parameters
    ----------
    events : list[dict]
        로거로부터 받은 키 이벤트 배열.
        각 항목: {"type", "key", "timestamp", "is_delete"}

    Returns
    -------
    pd.DataFrame
        FEATURE_COLS 순서의 단일 행 피처 벡터.
    """
    # 이벤트를 타임스탬프 오름차순으로 정렬
    events = sorted(events, key=lambda e: e["timestamp"])

    del_count = 0
    left_count = 0
    timestamps_ms = [e["timestamp"] * 1000 for e in events]
    tot_time_ms = (timestamps_ms[-1] - timestamps_ms[0]) if len(timestamps_ms) >= 2 else 0.0

    # 타이핑 키(일반 문자)의 keydown-keyup 쌍을 추출
    # keyup_queue: key -> deque of keyup timestamps (ms)
    keyup_queue: dict[str, deque] = {}
    for event in events:
        key = event["key"]
        ts_ms = event["timestamp"] * 1000

        if event.get("is_delete", False):
            if event["type"] == "keydown":
                del_count += 1
            continue

        if key == "ArrowLeft":
            if event["type"] == "keydown":
                left_count += 1
            continue

        # 타이밍 계산 대상에서 제외할 특수 키
        if event["type"] == "keyup" and len(key) > 1:
            continue

        if event["type"] == "keyup":
            if key not in keyup_queue:
                keyup_queue[key] = deque()
            keyup_queue[key].append(ts_ms)

    # keydown 순서대로 쌍 구성
    pairs: list[dict] = []
    for event in events:
        if event["type"] != "keydown":
            continue
        if event.get("is_delete", False) or event["key"] == "ArrowLeft":
            continue
        if len(event["key"]) > 1:
            continue

        key = event["key"]
        kd_ms = event["timestamp"] * 1000

        # 이 keydown 이후의 가장 빠른 keyup 매칭
        if key in keyup_queue and keyup_queue[key]:
            candidates = [t for t in keyup_queue[key] if t >= kd_ms]
            if candidates:
                ku_ms = min(candidates)
                keyup_queue[key].remove(ku_ms)
                pairs.append({"kd": kd_ms, "ku": ku_ms})

    # 타이밍 피처 계산
    d1u1 = [min(p["ku"] - p["kd"], TIMING_CLIP_MS) for p in pairs]
    d1u1 = [v for v in d1u1 if v >= 0]

    d1d2, u1d2, u1u2 = [], [], []
    for i in range(len(pairs) - 1):
        cur, nxt = pairs[i], pairs[i + 1]
        dd = nxt["kd"] - cur["kd"]
        ud = nxt["kd"] - cur["ku"]
        uu = nxt["ku"] - cur["ku"]

        if 0 <= dd <= TIMING_CLIP_MS:
            d1d2.append(dd)
        if ud <= TIMING_CLIP_MS:
            u1d2.append(ud)
        if 0 <= uu <= TIMING_CLIP_MS:
            u1u2.append(uu)

    def _stats(values: list[float]) -> dict:
        """리스트의 통계값을 반환한다. 비어 있으면 0으로 채운다."""
        if not values:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values, dtype=float)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "median": float(np.median(arr)),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
        }

    s_d1u1 = _stats(d1u1)
    s_d1d2 = _stats(d1d2)
    s_u1d2 = _stats(u1d2)
    s_u1u2 = _stats(u1u2)

    row = {
        **{f"d1u1_{k}": v for k, v in s_d1u1.items()},
        **{f"d1d2_{k}": v for k, v in s_d1d2.items()},
        **{f"u1d2_{k}": v for k, v in s_u1d2.items()},
        **{f"u1u2_{k}": v for k, v in s_u1u2.items()},
        "del_freq_mean":  float(del_count),
        "left_freq_mean": float(left_count),
        "tot_time_mean":  tot_time_ms,
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


def predict(raw_input: dict) -> dict:
    """
    raw 키스트로크 JSON을 받아 감정 예측 결과를 반환한다.

    Parameters
    ----------
    raw_input : dict
        키스트로크 로거(B)의 출력 JSON.
        필수 키: session_id, turn_id, events

    Returns
    -------
    dict
        interface_spec.md Module 2 Classifier Output 형식의 딕셔너리.

    Raises
    ------
    ValueError
        events 배열이 비어 있는 경우.
    """
    events = raw_input.get("events", [])
    if not events:
        raise ValueError("events 배열이 비어 있습니다.")

    model, le = _load_model()
    X = _extract_features(events)

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    emotion = str(le.inverse_transform([pred_idx])[0])

    # avg_iki_ms: D1D2 mean (연속 keydown 간 평균 간격)
    avg_iki_ms = float(X["d1d2_mean"].iloc[0])

    # backspace_rate: 삭제 키 수 / 전체 keydown 수
    total_keydowns = sum(1 for e in events if e["type"] == "keydown")
    del_count = sum(1 for e in events if e["type"] == "keydown" and e.get("is_delete", False))
    backspace_rate = round(del_count / total_keydowns, 4) if total_keydowns > 0 else 0.0

    return {
        "session_id":     raw_input["session_id"],
        "turn_id":        raw_input["turn_id"],
        "emotion":        emotion,
        "confidence":     round(confidence, 4),
        "avg_iki_ms":     round(avg_iki_ms, 1),
        "backspace_rate": backspace_rate,
    }


if __name__ == "__main__":
    import json

    sample_input = {
        "session_id": "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
        "turn_id": 1,
        "events": [
            {"type": "keydown", "key": "h", "timestamp": 1710234560.001, "is_delete": False},
            {"type": "keyup",   "key": "h", "timestamp": 1710234560.120, "is_delete": False},
            {"type": "keydown", "key": "i", "timestamp": 1710234560.350, "is_delete": False},
            {"type": "keyup",   "key": "i", "timestamp": 1710234560.430, "is_delete": False},
            {"type": "keydown", "key": "Backspace", "timestamp": 1710234561.200, "is_delete": True},
            {"type": "keyup",   "key": "Backspace", "timestamp": 1710234561.280, "is_delete": True},
            {"type": "keydown", "key": "i", "timestamp": 1710234562.500, "is_delete": False},
            {"type": "keyup",   "key": "i", "timestamp": 1710234562.580, "is_delete": False},
        ],
    }

    result = predict(sample_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
