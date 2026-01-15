"""
Sample Tool

도메인별 도구 구현 예시입니다.
새 서비스 생성 시 이 파일을 복사하여 도메인에 맞게 수정하세요.
"""

from typing import Any

from agentic_agents.tools import BaseTool


class SampleTool(BaseTool):
    """
    샘플 도구

    TODO: 도메인에 맞게 다음을 수정하세요:
    1. 클래스명 변경 (예: NotionSearchTool, DatabaseQueryTool)
    2. name, description 수정
    3. parameters 정의
    4. execute 로직 구현
    """

    name = "sample_tool"
    description = "샘플 도구 - 도메인에 맞게 수정하세요"

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색 쿼리",
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        도구 실행

        Args:
            **kwargs: 도구 파라미터

        Returns:
            실행 결과
        """
        query = kwargs.get("query", "")

        # TODO: 실제 도구 로직 구현
        # 예: DB 조회, API 호출, 파일 검색 등

        return {
            "success": True,
            "result": f"Query executed: {query}",
        }
