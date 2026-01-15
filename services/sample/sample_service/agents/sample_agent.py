"""
Sample Agent

도메인별 에이전트 구현 예시입니다.
새 서비스 생성 시 이 파일을 복사하여 도메인에 맞게 수정하세요.
"""

from typing import Any

from agentic_agents.base import BaseAgent


class SampleAgent(BaseAgent):
    """
    샘플 에이전트

    TODO: 도메인에 맞게 다음을 수정하세요:
    1. 클래스명 변경 (예: CustomerServiceAgent, KnowledgeAgent)
    2. system_prompt 수정
    3. 필요한 tools 등록
    """

    name = "sample_agent"
    description = "샘플 도메인 에이전트"

    system_prompt = """
    당신은 샘플 서비스의 AI 어시스턴트입니다.

    TODO: 도메인에 맞는 시스템 프롬프트로 수정하세요.
    - 역할과 책임 정의
    - 응답 스타일 가이드
    - 제약 조건
    """

    async def process(self, message: str, context: dict[str, Any] | None = None) -> str:
        """
        사용자 메시지 처리

        Args:
            message: 사용자 입력 메시지
            context: 추가 컨텍스트 (세션 정보, 사용자 정보 등)

        Returns:
            에이전트 응답
        """
        # TODO: 실제 LLM 호출 및 도구 사용 로직 구현
        return f"[SampleAgent] 메시지 수신: {message}"
