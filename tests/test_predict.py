"""
predict() 엔드투엔드 테스트

keystroke_classifier.pkl 모델 파일이 있어야 실행 가능.
fixtures/sample_inputs.json 의 각 샘플에 대해
출력 스펙, 감정 레이블, 신뢰도 범위, 에지 케이스를 검증한다.

실행 방법:
    cd modules/classifier
    python -m pytest ../../tests/test_predict.py -v
또는
    python ../../tests/test_predict.py
"""

import json
import sys
import pathlib
import traceback

CLASSIFIER_DIR = pathlib.Path(__file__).parent.parent / "modules" / "classifier"
sys.path.insert(0, str(CLASSIFIER_DIR))

from predict import predict, FEATURE_COLS  # noqa: E402

FIXTURES_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_inputs.json"

VALID_EMOTIONS = {"angry", "sad", "happy", "neutral", "calm"}


def load_samples() -> list[dict]:
    """fixtures JSON에서 샘플 목록을 반환한다."""
    with open(FIXTURES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


SAMPLES = load_samples()
NORMAL_SAMPLES = [s for s in SAMPLES if not s["session_id"].startswith("test-edge")]


class TestOutputSpec:
    """출력 JSON이 interface_spec.md Module 2 Classifier Output 스펙을 준수하는지 검증한다."""

    REQUIRED_KEYS = {"session_id", "turn_id", "emotion", "confidence", "avg_iki_ms", "backspace_rate"}

    def _run_all(self) -> list[dict]:
        """일반 샘플 전체에 대해 predict()를 실행한다."""
        return [predict(s) for s in NORMAL_SAMPLES]

    def test_output_has_required_keys(self) -> None:
        """출력에 필수 키가 모두 포함된다."""
        for sample in NORMAL_SAMPLES:
            result = predict(sample)
            missing = self.REQUIRED_KEYS - result.keys()
            assert not missing, (
                f"[{sample['session_id']}] 누락된 키: {missing}"
            )

    def test_session_id_passthrough(self) -> None:
        """session_id가 입력과 동일하게 출력된다."""
        for sample in NORMAL_SAMPLES:
            result = predict(sample)
            assert result["session_id"] == sample["session_id"]

    def test_turn_id_passthrough(self) -> None:
        """turn_id가 입력과 동일하게 출력된다."""
        for sample in NORMAL_SAMPLES:
            result = predict(sample)
            assert result["turn_id"] == sample["turn_id"]

    def test_emotion_is_valid_label(self) -> None:
        """emotion 값이 유효한 감정 레이블 중 하나이다."""
        for sample in NORMAL_SAMPLES:
            result = predict(sample)
            assert result["emotion"] in VALID_EMOTIONS, (
                f"[{sample['session_id']}] 알 수 없는 감정: {result['emotion']}"
            )

    def test_confidence_in_range(self) -> None:
        """confidence는 [0, 1] 범위 내이다."""
        for sample in NORMAL_SAMPLES:
            result = predict(sample)
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"[{sample['session_id']}] confidence 범위 오류: {result['confidence']}"
            )

    def test_avg_iki_ms_nonnegative(self) -> None:
        """avg_iki_ms는 0 이상이다."""
        for sample in NORMAL_SAMPLES:
            result = predict(sample)
            assert result["avg_iki_ms"] >= 0.0, (
                f"[{sample['session_id']}] avg_iki_ms 음수: {result['avg_iki_ms']}"
            )

    def test_backspace_rate_in_range(self) -> None:
        """backspace_rate는 [0, 1] 범위 내이다."""
        for sample in NORMAL_SAMPLES:
            result = predict(sample)
            assert 0.0 <= result["backspace_rate"] <= 1.0, (
                f"[{sample['session_id']}] backspace_rate 범위 오류: {result['backspace_rate']}"
            )

    def test_confidence_is_rounded_4dp(self) -> None:
        """confidence는 소수점 4자리로 반올림된다."""
        for sample in NORMAL_SAMPLES:
            result = predict(sample)
            rounded = round(result["confidence"], 4)
            assert result["confidence"] == rounded, (
                f"[{sample['session_id']}] confidence 반올림 불일치: {result['confidence']}"
            )


class TestPatternPrediction:
    """감정 패턴에 따라 피처값이 예측 결과에 올바르게 반영되는지 검증한다."""

    def _get(self, session_id: str) -> dict:
        """session_id로 샘플을 찾는다."""
        for s in SAMPLES:
            if s["session_id"] == session_id:
                return s
        raise KeyError(f"샘플 없음: {session_id}")

    def test_angry_backspace_rate_higher_than_calm(self) -> None:
        """angry 예측 결과의 backspace_rate가 calm보다 높아야 한다."""
        angry = predict(self._get("test-angry-001"))
        calm  = predict(self._get("test-calm-001"))
        assert angry["backspace_rate"] > calm["backspace_rate"], (
            f"angry={angry['backspace_rate']}, calm={calm['backspace_rate']}"
        )

    def test_sad_avg_iki_higher_than_happy(self) -> None:
        """sad 예측 결과의 avg_iki_ms가 happy보다 높아야 한다."""
        sad   = predict(self._get("test-sad-001"))
        happy = predict(self._get("test-happy-001"))
        assert sad["avg_iki_ms"] > happy["avg_iki_ms"], (
            f"sad={sad['avg_iki_ms']}, happy={happy['avg_iki_ms']}"
        )

    def test_angry_avg_iki_lower_than_sad(self) -> None:
        """angry 예측 결과의 avg_iki_ms가 sad보다 낮아야 한다."""
        angry = predict(self._get("test-angry-001"))
        sad   = predict(self._get("test-sad-001"))
        assert angry["avg_iki_ms"] < sad["avg_iki_ms"], (
            f"angry={angry['avg_iki_ms']}, sad={sad['avg_iki_ms']}"
        )


class TestEdgeCases:
    """에지 케이스에서 predict()가 올바르게 동작하는지 검증한다."""

    def _get(self, session_id: str) -> dict:
        for s in SAMPLES:
            if s["session_id"] == session_id:
                return s
        raise KeyError(f"샘플 없음: {session_id}")

    def test_empty_events_raises_value_error(self) -> None:
        """events가 빈 배열이면 ValueError를 발생시킨다."""
        raw = {"session_id": "test-empty", "turn_id": 1, "events": []}
        try:
            predict(raw)
            assert False, "ValueError가 발생해야 합니다"
        except ValueError:
            pass

    def test_minimal_events_returns_valid_output(self) -> None:
        """키 2개 이벤트에서 predict()가 정상 출력을 반환한다."""
        result = predict(self._get("test-edge-minimal-001"))
        assert result["emotion"] in VALID_EMOTIONS
        assert 0.0 <= result["confidence"] <= 1.0

    def test_only_deletes_backspace_rate_is_one(self) -> None:
        """삭제 키만 있을 때 backspace_rate는 1.0이어야 한다."""
        result = predict(self._get("test-edge-deletes-001"))
        assert result["backspace_rate"] == 1.0, (
            f"backspace_rate={result['backspace_rate']}"
        )

    def test_only_deletes_returns_valid_output(self) -> None:
        """삭제 키만 있을 때도 유효한 감정 레이블이 반환된다."""
        result = predict(self._get("test-edge-deletes-001"))
        assert result["emotion"] in VALID_EMOTIONS


if __name__ == "__main__":
    test_classes = [TestOutputSpec, TestPatternPrediction, TestEdgeCases]
    passed = failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            try:
                getattr(instance, method_name)()
                print(f"  PASS  {cls.__name__}::{method_name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}::{method_name}")
                traceback.print_exc()
                failed += 1

    print(f"\n결과: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
