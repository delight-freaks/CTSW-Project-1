"""
LLM 클라이언트 모듈

Claude API 호출을 담당하는 독립 모듈.
prompt_assembler.py가 조립한 프롬프트를 받아 Claude API를 호출하고 응답을 반환한다.

실행:
  python llm_client.py  # 단순 import 테스트 (API 호출 없음)
"""

import os
from typing import Optional


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024


def call_claude_api(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: Optional[str] = None,
) -> str:
    """
    Claude API를 호출하여 상담 응답을 받는다.

    Parameters
    ----------
    system_prompt : str
        시스템 프롬프트 (상담 AI 역할 지시문)
    user_prompt : str
        사용자 프롬프트 (assemble_prompt가 조립한 다모달 컨텍스트)
    model : str
        사용할 Claude 모델 ID. 기본값: claude-sonnet-4-6
    max_tokens : int
        최대 생성 토큰 수. 기본값: 1024
    api_key : str, optional
        Anthropic API 키. None이면 환경변수 ANTHROPIC_API_KEY를 사용한다.

    Returns
    -------
    str
        Claude의 응답 텍스트

    Raises
    ------
    ImportError
        anthropic 패키지가 설치되지 않은 경우
    KeyError
        api_key가 None이고 환경변수 ANTHROPIC_API_KEY가 설정되지 않은 경우
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic 후 재실행하세요.")

    resolved_key = api_key or os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=resolved_key)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text
