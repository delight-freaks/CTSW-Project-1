"""
프롬프트 조립 모듈

입력: 세 모달리티의 JSON 출력 + 텍스트 데이터
출력: LLM에 전달할 구조화된 프롬프트 문자열

실행:
  python prompt_assembler.py          # mock 데이터로 프롬프트 생성 및 출력
  python prompt_assembler.py --claude # Claude API 호출까지 진행 (API 키 필요)
"""

import json
import re
import argparse
from dataclasses import dataclass, field
from typing import Optional

try:
    from modules.pipeline.llm_client import call_claude_api
except ImportError:
    from llm_client import call_claude_api


# ---------------------------------------------------------------------------
# 1. 데이터 구조 정의
# ---------------------------------------------------------------------------

@dataclass
class VisionOutput:
    """비전 모듈(심인영) 출력 스펙 — Module 1"""
    timestamp: float
    face_detected: bool
    emotion: Optional[str]
    confidence: Optional[float]
    emotion_scores: Optional[dict]
    head_pose: Optional[dict]
    peak_emotion: Optional[str] = None
    peak_confidence: Optional[float] = None
    peak_detected_at: Optional[float] = None


@dataclass
class KeystrokeOutput:
    """키스트로크 분류기(박관용) 출력 스펙 — Module 2 Classifier Output"""
    session_id: str
    turn_id: int
    emotion: str
    confidence: float
    avg_iki_ms: Optional[float] = None      # 평균 IKI (밀리초). 프롬프트에서 "입력 지연 X초"로 변환
    backspace_rate: Optional[float] = None  # 삭제 키 비율. 프롬프트에서 망설임 정도 표현에 활용


@dataclass
class DeletedSegment:
    """삭제된 텍스트 세그먼트"""
    text: str
    deleted_at: float  # Unix timestamp


@dataclass
class TextInput:
    """텍스트 모듈(박관용) 출력 스펙 — Module 3"""
    session_id: str
    turn_id: int
    final_text: str
    deleted_segments: list[DeletedSegment] = field(default_factory=list)


@dataclass
class SilenceEvent:
    """침묵 모니터(이고은) 출력 스펙 — Module 4"""
    session_id: str
    turn_id: int
    type: str                               # 항상 "silence_event"
    silence_duration_sec: float
    context: str                            # "after_llm_response" | "mid_typing"
    last_keystroke_at: Optional[float] = None
    timestamp: Optional[float] = None


# ---------------------------------------------------------------------------
# 2. Mock 데이터
# ---------------------------------------------------------------------------

MOCK_VISION = VisionOutput(
    timestamp=1710234567.123,
    face_detected=True,
    emotion="sad",
    confidence=0.72,
    emotion_scores={
        "sad": 0.72, "angry": 0.10, "happy": 0.05,
        "neutral": 0.08, "surprised": 0.03, "fearful": 0.02,
    },
    head_pose={"yaw": -12.3, "pitch": 5.1, "roll": 2.0},
    peak_emotion="fearful",
    peak_confidence=0.74,
    peak_detected_at=1710234564.001,
)

MOCK_KEYSTROKE = KeystrokeOutput(
    session_id="a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
    turn_id=3,
    emotion="anxious",
    confidence=0.61,
    avg_iki_ms=2300.0,
    backspace_rate=0.14,
)

MOCK_TEXT = TextInput(
    session_id="a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
    turn_id=3,
    final_text="그냥 힘들어요",
    deleted_segments=[
        DeletedSegment(text="죽고 싶어요", deleted_at=1710234561.200),
        DeletedSegment(text="요즘 너무",   deleted_at=1710234563.800),
    ],
)

MOCK_SILENCE = SilenceEvent(
    session_id="a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
    turn_id=3,
    type="silence_event",
    silence_duration_sec=12.4,
    context="mid_typing",
    last_keystroke_at=1710234560.001,
    timestamp=1710234572.401,
)


# ---------------------------------------------------------------------------
# 3. PII 마스킹
# ---------------------------------------------------------------------------

# 한국 전화번호: 010-XXXX-XXXX, 02-XXX-XXXX 등 일반적인 형식
_RE_PHONE = re.compile(
    r"(?<!\d)"
    r"(0(?:10|1[1-9]|2|[3-9][0-9]?))"   # 지역번호 / 휴대폰 번호 앞자리
    r"[-\s.]?"
    r"(\d{3,4})"
    r"[-\s.]?"
    r"(\d{4})"
    r"(?!\d)"
)

# 주민등록번호: XXXXXX-XXXXXXX 또는 붙여쓴 13자리
_RE_RRN = re.compile(
    r"(?<!\d)"
    r"\d{6}"
    r"[-\s]?"
    r"[1-4]\d{6}"
    r"(?!\d)"
)

# 이메일 주소
_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)


def mask_pii(text: str) -> str:
    """
    text에서 개인식별정보(PII)를 마스킹하여 반환한다.

    마스킹 대상:
      - 한국 전화번호  → [전화번호]
      - 주민등록번호   → [주민등록번호]
      - 이메일 주소    → [이메일]
    """
    text = _RE_RRN.sub("[주민등록번호]", text)
    text = _RE_PHONE.sub("[전화번호]", text)
    text = _RE_EMAIL.sub("[이메일]", text)
    return text


# ---------------------------------------------------------------------------
# 4. Semantic Mapping: raw 수치 → 심리적 의미 레이블
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.45

# 복합 침묵 트리거에서 부정 감정으로 간주하는 레이블 집합
NEGATIVE_EMOTIONS_VISION    = {"sad", "fearful", "angry"}
NEGATIVE_EMOTIONS_KEYSTROKE = {"anxious", "sad", "angry"}

# 침묵 복합 트리거 감정 신뢰도 임계값
# 현재 값은 임시값이며, 파일럿 테스트(N=5)에서 수집된 신뢰도 분포를 기반으로 확정한다
THETA_VISION    = 0.60
THETA_KEYSTROKE = 0.55

EMOTION_KR = {
    "happy":     "행복",
    "sad":       "슬픔",
    "angry":     "분노",
    "calm":      "평온",
    "neutral":   "중립",
    "fearful":   "두려움",
    "surprised": "놀람",
    "anxious":   "불안",
    "disgust":   "혐오",
}


def interpret_head_pose(yaw: float, pitch: float) -> str:
    """
    head pose 수치를 심리적 자세 레이블로 변환한다.

    임계값 근거:
      yaw 20도 기준: Ekman & Friesen(1978)의 gaze aversion 연구 및
      MediaPipe head pose 관련 HCI 문헌에서 일반적으로 사용되는 기준.
      파일럿 테스트에서 팀원 5명이 실제 자세로 검증 후 확정할 것.
    """
    if abs(yaw) > 20:
        return "시선 회피 (고개가 옆으로 돌아있음)"
    elif pitch < -15:
        return "고개 숙임 (위축된 자세)"
    else:
        return "정면 응시"


def interpret_iki(avg_iki_ms: float) -> str:
    """
    평균 IKI를 심리적 의미 레이블로 변환한다.

    임계값(2000ms, 800ms)은 파일럿 테스트에서 검증 후 보고서에 근거를 명시한다.
    """
    if avg_iki_ms > 2000:
        return f"입력 지연 {avg_iki_ms / 1000:.1f}초 (긴 망설임)"
    elif avg_iki_ms > 800:
        return f"입력 지연 {avg_iki_ms / 1000:.1f}초 (보통)"
    else:
        return "빠른 입력"


def format_emotion_label(emotion: str, confidence: float) -> str:
    """감정 레이블을 신뢰도와 함께 한국어로 포맷한다."""
    label_kr = EMOTION_KR.get(emotion, emotion)
    if confidence >= CONFIDENCE_THRESHOLD:
        return f"{label_kr} (신뢰도 {confidence:.2f})"
    else:
        return f"{label_kr} (신뢰도 낮음 — {confidence:.2f}, 참고 수준)"


# ---------------------------------------------------------------------------
# 5. 특수 토큰 생성
# ---------------------------------------------------------------------------

def build_special_tokens(
    keystroke: KeystrokeOutput,
    vision: VisionOutput,
    text: TextInput,
) -> list[str]:
    """
    MARS 논문의 방법론에 기반하여 비언어 신호를 이산 토큰으로 변환한다.
    실제 LLM 파인튜닝 시 vocabulary에 추가될 토큰들이다.
    현재는 프롬프트 내 텍스트 힌트로 포함한다.
    """
    tokens = []

    if text.deleted_segments:
        tokens.append("[BACKSPACE_DETECTED]")

    if vision.face_detected and vision.emotion:
        tokens.append(f"[EMOTION:{vision.emotion.upper()}]")

    if vision.face_detected and vision.head_pose:
        yaw = vision.head_pose.get("yaw", 0)
        if abs(yaw) > 20:
            tokens.append("[GAZE:AVERTED]")
        else:
            tokens.append("[GAZE:FORWARD]")

    return tokens


# ---------------------------------------------------------------------------
# 6. Trigger Evaluator — 침묵 복합 조건
# ---------------------------------------------------------------------------

def evaluate_silence_trigger(
    silence_event: SilenceEvent,
    vision: VisionOutput,
    keystroke: KeystrokeOutput,
) -> bool:
    """
    침묵 이벤트 수신 시 LLM 호출 여부를 결정한다.

    8초 단일 시간 조건에서 출발하여, 감정 신호 조건을 AND로 결합한
    복합 트리거(Composite Trigger)를 적용한다.

    개입 조건 (모두 충족 시 True):
      1. silence_duration_sec >= 8.0
      2. 다음 중 하나 이상:
         a. 비전 peak_emotion이 부정 감정 AND peak_confidence >= THETA_VISION
         b. 키스트로크 emotion이 부정 감정 AND confidence >= THETA_KEYSTROKE
         c. context == "mid_typing"  (타이핑 중 멈춤 — 망설임 신호)

    개입 보류 조건 (해당 시 False):
      - context == "after_llm_response" AND 감정 신호 모두 임계값 미만
        → LLM이 방금 응답했고 감정도 안정적이면 읽고 생각하는 중으로 간주

    임계값(THETA_VISION, THETA_KEYSTROKE)은 파일럿 테스트(N=5) 후 확정한다.
    현재 값은 임시값이다.
    """
    # 1. 시간 조건 (필수)
    if silence_event.silence_duration_sec < 8.0:
        return False

    # 2. 감정 신호 평가
    vision_negative = (
        vision.face_detected
        and vision.peak_emotion in NEGATIVE_EMOTIONS_VISION
        and (vision.peak_confidence or 0.0) >= THETA_VISION
    )
    keystroke_negative = (
        keystroke.emotion in NEGATIVE_EMOTIONS_KEYSTROKE
        and keystroke.confidence >= THETA_KEYSTROKE
    )
    mid_typing = silence_event.context == "mid_typing"

    # 3. 개입 보류: LLM 응답 직후 + 감정 신호 없음 → 읽는 중으로 간주
    if silence_event.context == "after_llm_response" and not (vision_negative or keystroke_negative):
        return False

    # 4. 개입 조건: 감정 신호 하나 이상 충족
    return vision_negative or keystroke_negative or mid_typing


# ---------------------------------------------------------------------------
# 7. 프롬프트 조립
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """당신은 심리 상담 보조 AI입니다.
사용자의 언어적 표현뿐만 아니라 비언어적 신호(표정, 타이핑 패턴, 삭제된 텍스트, 침묵)를
종합적으로 고려하여 공감적이고 탐색적인 방식으로 응답합니다.

핵심 원칙:
1. 사용자의 감정을 단정하지 않고 부드럽게 탐색합니다.
2. 삭제된 텍스트가 있다면 사용자가 말하지 못한 무언가가 있을 수 있음을 인식합니다.
3. 즉각적인 해결책 제시보다 공감과 경청을 우선합니다.
4. 자해, 자살 관련 신호가 감지되면 반드시 안전 확인 절차를 따릅니다.

[분석 데이터 비노출 원칙 — 반드시 준수]
당신은 사용자의 타이핑 패턴, 삭제된 텍스트, 표정, 침묵 시간 등 비언어적 신호를
내부 참고 자료로만 활용합니다. 이 데이터의 존재와 분석 사실을 응답에서 절대 드러내지 않습니다.

금지 표현 예시 (이런 방식으로 절대 말하지 않습니다):
- "삭제하신 내용을 보니...", "지우셨던 텍스트에서..."
- "타이핑 패턴을 보면...", "입력 속도를 분석해보니..."
- "X초간 침묵하셨는데...", "오래 입력하지 않으셨군요..."
- "표정 분석 결과...", "감정 신호가 감지되어..."

대신, 위 신호에서 읽은 감정 상태를 자연스러운 공감 표현으로 녹여냅니다.
예: 삭제된 텍스트에서 고통이 느껴지면 → "뭔가 말하기 어려운 부분이 있으신 것 같아요."
예: 침묵 신호 → "천천히 하셔도 괜찮아요. 여기 있을게요." """


def assemble_prompt(
    vision: VisionOutput,
    keystroke: KeystrokeOutput,
    text: TextInput,
    silence: Optional[SilenceEvent] = None,
) -> tuple[str, str]:
    """
    세 모달리티 출력을 하나의 구조화된 user prompt로 조립한다.
    silence가 전달된 경우 침묵 트리거 전용 프롬프트를 생성한다.

    Returns
    -------
    system_prompt : str
    user_prompt   : str
    """
    if silence is not None:
        return SYSTEM_PROMPT, _assemble_silence_prompt(vision, silence)
    return SYSTEM_PROMPT, _assemble_normal_prompt(vision, keystroke, text)


def _assemble_normal_prompt(
    vision: VisionOutput,
    keystroke: KeystrokeOutput,
    text: TextInput,
) -> str:
    """전송 트리거 — 일반 프롬프트 조립"""
    masked_final   = mask_pii(text.final_text)
    masked_deleted = [mask_pii(seg.text) for seg in text.deleted_segments]

    lines = ["[사용자 상태 분석]"]

    # --- 표정 (전송 시점) ---
    if vision.face_detected and vision.emotion and vision.confidence is not None:
        lines.append(f"표정(전송 시점): {format_emotion_label(vision.emotion, vision.confidence)}")
    else:
        lines.append("표정(전송 시점): 데이터 없음 (카메라 미감지 또는 비활성화)")

    # --- 표정 (턴 중 최고점 peak_emotion) ---
    if (
        vision.face_detected
        and vision.peak_emotion
        and vision.peak_confidence is not None
        and vision.peak_detected_at is not None
    ):
        elapsed = round(vision.timestamp - vision.peak_detected_at, 1)
        peak_label = format_emotion_label(vision.peak_emotion, vision.peak_confidence)
        lines.append(f"표정(턴 중 최고점): {peak_label}, 전송 {elapsed}초 전")

    # --- 시선/자세 ---
    if vision.face_detected and vision.head_pose:
        pose_label = interpret_head_pose(
            vision.head_pose.get("yaw", 0),
            vision.head_pose.get("pitch", 0),
        )
        lines.append(f"시선: {pose_label}")

    # --- 타이핑 패턴 ---
    ks_emotion = format_emotion_label(keystroke.emotion, keystroke.confidence)
    iki_label  = interpret_iki(keystroke.avg_iki_ms) if keystroke.avg_iki_ms is not None else "입력 지연 정보 없음"
    timing_parts = [ks_emotion, iki_label]
    if keystroke.backspace_rate is not None:
        timing_parts.append(f"삭제율 {keystroke.backspace_rate:.0%}")
    lines.append(f"타이핑 패턴: {', '.join(timing_parts)}")

    # --- 삭제된 텍스트 ---
    if masked_deleted:
        for deleted in masked_deleted:
            lines.append(f'삭제된 텍스트: "{deleted}"')

    # --- 최종 입력 ---
    lines.append(f'최종 입력: "{masked_final}"')

    # --- 특수 토큰 ---
    tokens = build_special_tokens(keystroke, vision, text)
    if tokens:
        lines.append(f"신호 토큰: {' '.join(tokens)}")

    lines.append("")
    lines.append("사용자가 말하지 못한 감정이 있을 수 있습니다.")
    lines.append(
        "위 신호들을 종합하여 판단하되, 단정하지 말고 "
        "공감적으로 탐색하는 방식으로 응답하세요."
    )

    # 자해/자살 위험 신호 감지 (삭제된 텍스트 포함)
    crisis_keywords = ["죽", "자살", "사라", "없어지", "힘들어 죽"]
    all_texts = " ".join([seg.text for seg in text.deleted_segments] + [text.final_text])
    if any(kw in all_texts for kw in crisis_keywords):
        lines.append("")
        lines.append(
            "[주의] 위기 신호 감지: 삭제되거나 전송된 텍스트에 "
            "자해·자살 관련 표현이 포함되어 있습니다. "
            "안전 확인 절차에 따라 즉각 반응하세요."
        )

    return "\n".join(lines)


def _assemble_silence_prompt(
    vision: VisionOutput,
    silence: SilenceEvent,
) -> str:
    """침묵 트리거 — 침묵 프롬프트 조립"""
    lines = ["[사용자 상태 분석]"]

    # --- 표정 ---
    if vision.face_detected and vision.emotion and vision.confidence is not None:
        lines.append(f"표정: {format_emotion_label(vision.emotion, vision.confidence)}")
    else:
        lines.append("표정: 데이터 없음 (카메라 미감지 또는 비활성화)")

    # --- 시선/자세 ---
    if vision.face_detected and vision.head_pose:
        pose_label = interpret_head_pose(
            vision.head_pose.get("yaw", 0),
            vision.head_pose.get("pitch", 0),
        )
        lines.append(f"시선: {pose_label}")

    lines.append(f"침묵 지속: {silence.silence_duration_sec}초")
    lines.append(f"맥락: {silence.context}")
    lines.append("")
    lines.append(f"사용자가 {silence.silence_duration_sec}초간 입력하지 않고 있습니다.")
    lines.append("말하기 어렵거나 정리가 필요한 상황일 수 있습니다.")
    lines.append("강요하지 말고, 공간을 주는 방식으로 부드럽게 말을 건네세요.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. 진입점
# ---------------------------------------------------------------------------

def main(call_api: bool = False) -> None:
    print("=" * 60)
    print("Mock 데이터 기반 프롬프트 조립 테스트")
    print("=" * 60)

    print("\n--- [일반 프롬프트 (전송 트리거)] ---")
    system_prompt, user_prompt = assemble_prompt(MOCK_VISION, MOCK_KEYSTROKE, MOCK_TEXT)
    print("\n[SYSTEM PROMPT]")
    print(system_prompt)
    print("\n[USER PROMPT]")
    print(user_prompt)

    print("\n--- [침묵 프롬프트 (침묵 트리거)] ---")
    _, silence_prompt = assemble_prompt(MOCK_VISION, MOCK_KEYSTROKE, MOCK_TEXT, silence=MOCK_SILENCE)
    print("\n[SILENCE PROMPT]")
    print(silence_prompt)

    should_trigger = evaluate_silence_trigger(MOCK_SILENCE, MOCK_VISION, MOCK_KEYSTROKE)
    print(f"\n[침묵 트리거 평가 결과: {'개입' if should_trigger else '보류'}]")

    if call_api:
        print("\n[Claude API 호출 중...]")
        response = call_claude_api(system_prompt, user_prompt)
        print("\n[Claude 응답]")
        print(response)
    else:
        print("\n(--claude 플래그 없이 실행: API 호출 생략)")

    # JSON 직렬화 테스트 — 파이프라인 연동 시 인터페이스 검증용
    payload = {
        "system": system_prompt,
        "user": user_prompt,
        "meta": {
            "vision_emotion": MOCK_VISION.emotion,
            "keystroke_emotion": MOCK_KEYSTROKE.emotion,
            "has_deleted_text": len(MOCK_TEXT.deleted_segments) > 0,
        },
    }
    with open("prompt_payload_sample.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n[save] 프롬프트 페이로드 샘플 저장: prompt_payload_sample.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="프롬프트 조립 모듈 테스트")
    parser.add_argument(
        "--claude", action="store_true",
        help="Claude API를 실제로 호출한다 (ANTHROPIC_API_KEY 환경변수 필요)"
    )
    args = parser.parse_args()
    main(call_api=args.claude)
