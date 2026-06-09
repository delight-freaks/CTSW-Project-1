"""
베이스라인 상담 챗봇 파이프라인

멀티모달 신호(키스트로크, 비전, 삭제된 텍스트, 침묵) 없이
최종 전송 텍스트만으로 LLM에 응답을 요청하는 베이스라인 모듈.

비교 실험에서 멀티모달 파이프라인과 대조군으로 사용한다.

실행:
  python baseline_pipeline.py          # mock 데이터로 프롬프트 확인 (API 호출 없음)
  python baseline_pipeline.py --claude # Claude API 호출까지 진행 (ANTHROPIC_API_KEY 필요)
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.pipeline.llm_client import call_claude_api
from modules.pipeline.prompt_assembler import mask_pii


# ---------------------------------------------------------------------------
# 시스템 프롬프트 — 멀티모달 신호 없음
# ---------------------------------------------------------------------------

BASELINE_SYSTEM_PROMPT = """당신은 심리 상담 보조 AI입니다.
사용자가 전송한 텍스트 메시지를 바탕으로 공감적이고 탐색적인 방식으로 응답합니다.

핵심 원칙:
1. 사용자의 감정을 단정하지 않고 부드럽게 탐색합니다.
2. 즉각적인 해결책 제시보다 공감과 경청을 우선합니다.
3. 자해, 자살 관련 신호가 감지되면 반드시 안전 확인 절차를 따릅니다."""


# ---------------------------------------------------------------------------
# 프롬프트 조립
# ---------------------------------------------------------------------------

def assemble_baseline_prompt(final_text: str) -> tuple[str, str]:
    """
    최종 전송 텍스트만으로 프롬프트를 조립한다.

    Parameters
    ----------
    final_text : str
        사용자가 전송한 텍스트.

    Returns
    -------
    system_prompt : str
    user_prompt : str
    """
    masked = mask_pii(final_text)

    lines = [f'사용자 메시지: "{masked}"']

    crisis_keywords = ["죽", "자살", "사라", "없어지", "힘들어 죽"]
    if any(kw in final_text for kw in crisis_keywords):
        lines.append("")
        lines.append(
            "[주의] 위기 신호 감지: 자해·자살 관련 표현이 포함되어 있습니다. "
            "안전 확인 절차에 따라 즉각 반응하세요."
        )

    return BASELINE_SYSTEM_PROMPT, "\n".join(lines)


def run_baseline(final_text: str, call_llm: bool = False) -> dict:
    """
    베이스라인 파이프라인을 실행한다.

    Parameters
    ----------
    final_text : str
        사용자가 전송한 텍스트.
    call_llm : bool
        True이면 Claude API를 실제로 호출한다.

    Returns
    -------
    dict
        {
          "system_prompt": str,
          "user_prompt":   str,
          "llm_response":  str | None,
        }
    """
    system_prompt, user_prompt = assemble_baseline_prompt(final_text)

    llm_response = None
    if call_llm:
        llm_response = call_claude_api(system_prompt, user_prompt)

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm_response": llm_response,
    }


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main(call_llm: bool = False) -> None:
    """Mock 텍스트로 베이스라인 파이프라인을 실행하고 결과를 출력한다."""
    mock_text = "그냥 힘들어요"

    print("=" * 60)
    print("베이스라인 챗봇 파이프라인 테스트")
    print("=" * 60)

    result = run_baseline(mock_text, call_llm=call_llm)

    print("\n[SYSTEM PROMPT]")
    print(result["system_prompt"])
    print("\n[USER PROMPT]")
    print(result["user_prompt"])

    if result["llm_response"]:
        print("\n[Claude 응답]")
        print(result["llm_response"])
    else:
        print("\n(--claude 플래그 없이 실행: API 호출 생략)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="베이스라인 파이프라인 테스트")
    parser.add_argument(
        "--claude", action="store_true",
        help="Claude API를 실제로 호출한다 (ANTHROPIC_API_KEY 환경변수 필요)",
    )
    args = parser.parse_args()
    main(call_llm=args.claude)
