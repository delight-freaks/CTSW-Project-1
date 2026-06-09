"""
_extract_features 단위 테스트

모델 파일(keystroke_classifier.pkl) 없이 실행 가능.
fixtures/sample_inputs.json 의 각 샘플에 대해
피처 벡터 형상, 컬럼 순서, 에지 케이스 처리를 검증한다.

실행 방법:
    cd modules/classifier
    python -m pytest ../../tests/test_extract_features.py -v
또는
    python ../../tests/test_extract_features.py
"""

import json
import sys
import pathlib
import importlib.util
import numpy as np
import pytest

# classifier 모듈 경로를 sys.path에 추가
CLASSIFIER_DIR = pathlib.Path(__file__).parent.parent / "modules" / "classifier"
sys.path.insert(0, str(CLASSIFIER_DIR))

from predict import _extract_features, FEATURE_COLS  # noqa: E402

FIXTURES_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_inputs.json"


def load_samples() -> list[dict]:
    """fixtures JSON에서 샘플 목록을 반환한다."""
    with open(FIXTURES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


SAMPLES = load_samples()


class TestFeatureShape:
    """피처 벡터의 형상과 컬럼 순서를 검증한다."""

    def test_output_columns_match_feature_cols(self) -> None:
        """모든 샘플에서 출력 컬럼이 FEATURE_COLS와 일치한다."""
        for sample in SAMPLES:
            df = _extract_features(sample["events"])
            assert list(df.columns) == FEATURE_COLS, (
                f"[{sample['session_id']}] 컬럼 불일치: {list(df.columns)}"
            )

    def test_output_shape_is_single_row(self) -> None:
        """모든 샘플에서 출력이 1행이다."""
        for sample in SAMPLES:
            df = _extract_features(sample["events"])
            assert df.shape[0] == 1, (
                f"[{sample['session_id']}] 행 수 오류: {df.shape[0]}"
            )

    def test_no_nan_values(self) -> None:
        """NaN 값이 없어야 한다 (빈 bigram은 0으로 채워야 함)."""
        for sample in SAMPLES:
            df = _extract_features(sample["events"])
            nan_cols = df.columns[df.isna().any()].tolist()
            assert not nan_cols, (
                f"[{sample['session_id']}] NaN 발생 컬럼: {nan_cols}"
            )

    def test_no_negative_timing_values(self) -> None:
        """D1U1(hold time)은 음수가 없어야 한다."""
        timing_d1u1_cols = [c for c in FEATURE_COLS if c.startswith("d1u1")]
        for sample in SAMPLES:
            df = _extract_features(sample["events"])
            for col in timing_d1u1_cols:
                val = df[col].iloc[0]
                assert val >= 0, (
                    f"[{sample['session_id']}] {col} 음수 값: {val}"
                )


class TestPatternProperties:
    """감정별 타이핑 패턴이 피처에 반영되는지 검증한다."""

    def _get_sample(self, session_id: str) -> dict:
        """session_id로 샘플을 찾는다."""
        for s in SAMPLES:
            if s["session_id"] == session_id:
                return s
        raise KeyError(f"샘플 없음: {session_id}")

    def test_angry_has_higher_backspace_rate_than_calm(self) -> None:
        """angry 샘플의 del_freq_mean이 calm보다 높아야 한다."""
        angry_df = _extract_features(self._get_sample("test-angry-001")["events"])
        calm_df  = _extract_features(self._get_sample("test-calm-001")["events"])
        assert angry_df["del_freq_mean"].iloc[0] > calm_df["del_freq_mean"].iloc[0]

    def test_sad_has_higher_iki_than_happy(self) -> None:
        """sad 샘플의 d1d2_mean(IKI)이 happy보다 높아야 한다."""
        sad_df   = _extract_features(self._get_sample("test-sad-001")["events"])
        happy_df = _extract_features(self._get_sample("test-happy-001")["events"])
        assert sad_df["d1d2_mean"].iloc[0] > happy_df["d1d2_mean"].iloc[0]

    def test_angry_has_lower_iki_than_sad(self) -> None:
        """angry 샘플의 d1d2_mean(IKI)이 sad보다 낮아야 한다."""
        angry_df = _extract_features(self._get_sample("test-angry-001")["events"])
        sad_df   = _extract_features(self._get_sample("test-sad-001")["events"])
        assert angry_df["d1d2_mean"].iloc[0] < sad_df["d1d2_mean"].iloc[0]

    def test_calm_has_low_std(self) -> None:
        """calm 샘플의 d1d2_std가 sad보다 낮아야 한다 (일정한 리듬)."""
        calm_df = _extract_features(self._get_sample("test-calm-001")["events"])
        sad_df  = _extract_features(self._get_sample("test-sad-001")["events"])
        assert calm_df["d1d2_std"].iloc[0] < sad_df["d1d2_std"].iloc[0]


class TestEdgeCases:
    """에지 케이스 처리를 검증한다."""

    def _get_sample(self, session_id: str) -> dict:
        for s in SAMPLES:
            if s["session_id"] == session_id:
                return s
        raise KeyError(f"샘플 없음: {session_id}")

    def test_minimal_events_no_crash(self) -> None:
        """키 2개 이벤트에서 피처 추출이 정상 완료된다."""
        sample = self._get_sample("test-edge-minimal-001")
        df = _extract_features(sample["events"])
        assert df.shape == (1, len(FEATURE_COLS))

    def test_minimal_events_bigram_stats_zero(self) -> None:
        """키 2개에서는 bigram 1개뿐이므로 std가 0이어야 한다."""
        sample = self._get_sample("test-edge-minimal-001")
        df = _extract_features(sample["events"])
        # d1d2는 bigram 1개 → std=0
        assert df["d1d2_std"].iloc[0] == 0.0

    def test_only_deletes_no_crash(self) -> None:
        """삭제 키만 있는 이벤트에서 피처 추출이 정상 완료된다."""
        sample = self._get_sample("test-edge-deletes-001")
        df = _extract_features(sample["events"])
        assert df.shape == (1, len(FEATURE_COLS))

    def test_only_deletes_timing_is_zero(self) -> None:
        """삭제 키만 있을 때 타이밍 피처는 모두 0이어야 한다."""
        sample = self._get_sample("test-edge-deletes-001")
        df = _extract_features(sample["events"])
        timing_cols = [c for c in FEATURE_COLS if not c.endswith(("freq_mean", "time_mean"))]
        for col in timing_cols:
            assert df[col].iloc[0] == 0.0, f"{col} = {df[col].iloc[0]}"

    def test_only_deletes_del_freq_nonzero(self) -> None:
        """삭제 키만 있을 때 del_freq_mean은 0보다 커야 한다."""
        sample = self._get_sample("test-edge-deletes-001")
        df = _extract_features(sample["events"])
        assert df["del_freq_mean"].iloc[0] > 0


if __name__ == "__main__":
    # pytest 없이도 실행 가능한 간단 러너
    import traceback

    test_classes = [TestFeatureShape, TestPatternProperties, TestEdgeCases]
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
