"""
파이프라인 통합 모듈

predict() -> assemble_prompt() -> call_claude_api() 를 연결하는 end-to-end 파이프라인.

실행:
  python pipeline.py           # mock 데이터로 전체 흐름 실행 (API 호출 없음)
  python pipeline.py --claude  # Claude API 호출까지 진행 (ANTHROPIC_API_KEY 필요)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 경로에 추가 (직접 실행 시)
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.classifier.predict import predict
from modules.pipeline.prompt_assembler import (
    VisionOutput,
    KeystrokeOutput,
    TextInput,
    SilenceEvent,
    DeletedSegment,
    assemble_prompt,
    evaluate_silence_trigger,
)
from modules.pipeline.llm_client import call_claude_api


# ---------------------------------------------------------------------------
# 1. dict → dataclass 변환
# ---------------------------------------------------------------------------

def _vision_from_dict(d: dict) -> VisionOutput:
    """
    비전 모듈 출력 dict를 VisionOutput 데이터클래스로 변환한다.

    Parameters
    ----------
    d : dict
        interface_spec.md Module 1 형식의 딕셔너리.

    Returns
    -------
    VisionOutput
    """
    head_pose_raw = d.get("head_pose")
    return VisionOutput(
        timestamp=d["timestamp"],
        face_detected=d["face_detected"],
        emotion=d.get("emotion"),
        confidence=d.get("confidence"),
        emotion_scores=d.get("emotion_scores"),
        head_pose=head_pose_raw,
        peak_emotion=d.get("peak_emotion"),
        peak_confidence=d.get("peak_confidence"),
        peak_detected_at=d.get("peak_detected_at"),
    )


def _keystroke_from_dict(d: dict) -> KeystrokeOutput:
    """
    키스트로크 분류기 출력 dict를 KeystrokeOutput 데이터클래스로 변환한다.

    predict()의 반환값을 그대로 받는다.

    Parameters
    ----------
    d : dict
        interface_spec.md Module 2 Classifier Output 형식의 딕셔너리.

    Returns
    -------
    KeystrokeOutput
    """
    return KeystrokeOutput(
        session_id=d["session_id"],
        turn_id=d["turn_id"],
        emotion=d["emotion"],
        confidence=d["confidence"],
        avg_iki_ms=d.get("avg_iki_ms"),
        backspace_rate=d.get("backspace_rate"),
    )


def _text_from_dict(d: dict) -> TextInput:
    """
    텍스트 모듈 출력 dict를 TextInput 데이터클래스로 변환한다.

    Parameters
    ----------
    d : dict
        interface_spec.md Module 3 형식의 딕셔너리.

    Returns
    -------
    TextInput
    """
    segments = [
        DeletedSegment(text=seg["text"], deleted_at=seg["deleted_at"])
        for seg in d.get("deleted_segments", [])
    ]
    return TextInput(
        session_id=d["session_id"],
        turn_id=d["turn_id"],
        final_text=d["final_text"],
        deleted_segments=segments,
    )


def _silence_from_dict(d: dict) -> SilenceEvent:
    """
    침묵 이벤트 dict를 SilenceEvent 데이터클래스로 변환한다.

    Parameters
    ----------
    d : dict
        interface_spec.md Module 4 형식의 딕셔너리.

    Returns
    -------
    SilenceEvent
    """
    return SilenceEvent(
        session_id=d["session_id"],
        turn_id=d["turn_id"],
        type=d["type"],
        silence_duration_sec=d["silence_duration_sec"],
        context=d["context"],
        last_keystroke_at=d.get("last_keystroke_at"),
        timestamp=d.get("timestamp"),
    )


# ---------------------------------------------------------------------------
# 2. 파이프라인 진입점
# ---------------------------------------------------------------------------

def run_pipeline(
    keystroke_raw: dict,
    vision_dict: dict,
    text_dict: dict,
    silence_dict: Optional[dict] = None,
    call_llm: bool = False,
) -> dict:
    """
    raw 키스트로크 이벤트부터 LLM 호출까지의 end-to-end 파이프라인을 실행한다.

    Parameters
    ----------
    keystroke_raw : dict
        키스트로크 로거(B) 출력. interface_spec.md Module 2 Logger Output 형식.
    vision_dict : dict
        비전 모듈(C) 출력. interface_spec.md Module 1 형식.
    text_dict : dict
        텍스트 캡처(A) 출력. interface_spec.md Module 3 형식.
    silence_dict : dict, optional
        침묵 이벤트(E) 출력. interface_spec.md Module 4 형식.
        None이면 전송 트리거로 처리한다.
    call_llm : bool
        True이면 Claude API를 실제로 호출한다. 기본값: False.

    Returns
    -------
    dict
        {
          "trigger":        "normal" | "silence" | "silence_suppressed",
          "system_prompt":  str,
          "user_prompt":    str,
          "keystroke_output": dict,   # predict() 반환값
          "llm_response":   str | None,
        }
    """
    # Step 1. 키스트로크 감정 예측
    keystroke_result = predict(keystroke_raw)

    # Step 2. dict -> dataclass
    vision = _vision_from_dict(vision_dict)
    keystroke = _keystroke_from_dict(keystroke_result)
    text = _text_from_dict(text_dict)
    silence: Optional[SilenceEvent] = _silence_from_dict(silence_dict) if silence_dict else None

    # Step 3. 침묵 트리거 평가 (silence_dict가 있는 경우에만)
    trigger = "normal"
    if silence is not None:
        if evaluate_silence_trigger(silence, vision, keystroke):
            trigger = "silence"
        else:
            trigger = "silence_suppressed"

    # Step 4. 프롬프트 조립
    #   silence_suppressed: 개입 보류 - LLM 호출하지 않음
    assembled_silence = silence if trigger == "silence" else None
    system_prompt, user_prompt = assemble_prompt(vision, keystroke, text, assembled_silence)

    # Step 5. LLM 호출 (선택적, silence_suppressed는 호출 안 함)
    llm_response: Optional[str] = None
    if call_llm and trigger != "silence_suppressed":
        llm_response = call_claude_api(system_prompt, user_prompt)

    return {
        "trigger":          trigger,
        "system_prompt":    system_prompt,
        "user_prompt":      user_prompt,
        "keystroke_output": keystroke_result,
        "llm_response":     llm_response,
    }


# ---------------------------------------------------------------------------
# 3. Mock 데이터
# ---------------------------------------------------------------------------

MOCK_KEYSTROKE_RAW = {
    "session_id": "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
    "turn_id": 3,
    "events": [
        {"type": "keydown", "key": "g", "timestamp": 1710234558.000, "is_delete": False},
        {"type": "keyup",   "key": "g", "timestamp": 1710234558.090, "is_delete": False},
        {"type": "keydown", "key": "e", "timestamp": 1710234558.400, "is_delete": False},
        {"type": "keyup",   "key": "e", "timestamp": 1710234558.480, "is_delete": False},
        {"type": "keydown", "key": "u", "timestamp": 1710234560.800, "is_delete": False},
        {"type": "keyup",   "key": "u", "timestamp": 1710234560.890, "is_delete": False},
        {"type": "keydown", "key": "Backspace", "timestamp": 1710234561.200, "is_delete": True},
        {"type": "keyup",   "key": "Backspace", "timestamp": 1710234561.280, "is_delete": True},
        {"type": "keydown", "key": "u", "timestamp": 1710234563.500, "is_delete": False},
        {"type": "keyup",   "key": "u", "timestamp": 1710234563.580, "is_delete": False},
    ],
}

MOCK_VISION_DICT = {
    "timestamp": 1710234567.123,
    "face_detected": True,
    "emotion": "sad",
    "confidence": 0.72,
    "emotion_scores": {
        "sad": 0.72, "angry": 0.10, "happy": 0.05,
        "neutral": 0.08, "surprised": 0.03, "fearful": 0.02,
    },
    "head_pose": {"yaw": -12.3, "pitch": 5.1, "roll": 2.0},
    "peak_emotion": "fearful",
    "peak_confidence": 0.74,
    "peak_detected_at": 1710234564.001,
}

MOCK_TEXT_DICT = {
    "session_id": "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
    "turn_id": 3,
    "final_text": "그냥 힘들어요",
    "deleted_segments": [
        {"text": "죽고 싶어요", "deleted_at": 1710234561.200},
        {"text": "요즘 너무",   "deleted_at": 1710234563.800},
    ],
}

MOCK_SILENCE_DICT = {
    "session_id": "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
    "turn_id": 3,
    "type": "silence_event",
    "silence_duration_sec": 12.4,
    "context": "mid_typing",
    "last_keystroke_at": 1710234560.001,
    "timestamp": 1710234572.401,
}


# ---------------------------------------------------------------------------
# 4. 진입점
# ---------------------------------------------------------------------------

def main(call_llm: bool = False) -> None:
    """Mock 데이터로 end-to-end 파이프라인을 실행하고 결과를 출력한다."""
    print("=" * 60)
    print("End-to-End 파이프라인 통합 테스트")
    print("=" * 60)

    # --- 케이스 1: 전송 트리거 ---
    print("\n[케이스 1] 전송 트리거 (silence 없음)")
    result = run_pipeline(
        keystroke_raw=MOCK_KEYSTROKE_RAW,
        vision_dict=MOCK_VISION_DICT,
        text_dict=MOCK_TEXT_DICT,
        call_llm=call_llm,
    )
    print(f"  트리거: {result['trigger']}")
    print(f"  키스트로크 예측: {result['keystroke_output']}")
    print("\n  [USER PROMPT]")
    print(result["user_prompt"])
    if result["llm_response"]:
        print("\n  [Claude 응답]")
        print(result["llm_response"])

    # --- 케이스 2: 침묵 트리거 (개입) ---
    print("\n" + "=" * 60)
    print("[케이스 2] 침묵 트리거 - mid_typing (개입 예상)")
    result2 = run_pipeline(
        keystroke_raw=MOCK_KEYSTROKE_RAW,
        vision_dict=MOCK_VISION_DICT,
        text_dict=MOCK_TEXT_DICT,
        silence_dict=MOCK_SILENCE_DICT,
        call_llm=call_llm,
    )
    print(f"  트리거: {result2['trigger']}")
    print("\n  [SILENCE PROMPT]")
    print(result2["user_prompt"])
    if result2["llm_response"]:
        print("\n  [Claude 응답]")
        print(result2["llm_response"])

    # --- 케이스 3: 침묵 트리거 (보류) ---
    print("\n" + "=" * 60)
    suppressed_silence = {**MOCK_SILENCE_DICT, "context": "after_llm_response"}
    neutral_vision = {
        **MOCK_VISION_DICT,
        "emotion": "neutral", "confidence": 0.80,
        "peak_emotion": "neutral", "peak_confidence": 0.78,
    }
    print("[케이스 3] 침묵 트리거 - after_llm_response + 중립 감정 (보류 예상)")
    result3 = run_pipeline(
        keystroke_raw=MOCK_KEYSTROKE_RAW,
        vision_dict=neutral_vision,
        text_dict=MOCK_TEXT_DICT,
        silence_dict=suppressed_silence,
        call_llm=call_llm,
    )
    print(f"  트리거: {result3['trigger']} (LLM 호출 안 함)")

    # 결과 저장
    output = {
        "case1_normal":   {"trigger": result["trigger"],  "user_prompt": result["user_prompt"],  "keystroke_output": result["keystroke_output"]},
        "case2_silence":  {"trigger": result2["trigger"], "user_prompt": result2["user_prompt"]},
        "case3_suppress": {"trigger": result3["trigger"]},
    }
    out_path = _ROOT / "pipeline_test_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[save] 결과 저장: {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="파이프라인 통합 테스트")
    parser.add_argument(
        "--claude", action="store_true",
        help="Claude API를 실제로 호출한다 (ANTHROPIC_API_KEY 환경변수 필요)",
    )
    args = parser.parse_args()
    main(call_llm=args.claude)
