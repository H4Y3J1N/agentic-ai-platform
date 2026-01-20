"""
Tools Package - Tool Registry and Base Classes

Tool 추상화 및 Registry 패턴 제공
- BaseTool: 모든 Tool의 추상 기반 클래스
- ToolRegistry: Tool 등록/조회/관리
- ToolConfig: Tool 설정

Usage:
    from agentic_core.tools import BaseTool, ToolRegistry, tool

    # 데코레이터로 Tool 정의
    @tool(name="search", description="Search documents")
    class SearchTool(BaseTool):
        async def execute(self, query: str, **kwargs) -> ToolResult:
            ...

    # Registry 사용
    registry = ToolRegistry()
    registry.register(SearchTool())

    # 또는 auto-discovery
    registry.discover_tools("my_package.tools")
"""

from .base import (
    BaseTool,
    ToolResult,
    ToolConfig,
    ToolCapability,
    ToolMetadata,
    tool,
)

from .registry import (
    ToolRegistry,
    ToolRegistryConfig,
    ToolNotFoundError,
    ToolAlreadyExistsError,
)

__all__ = [
    # Base
    "BaseTool",
    "ToolResult",
    "ToolConfig",
    "ToolCapability",
    "ToolMetadata",
    "tool",
    # Registry
    "ToolRegistry",
    "ToolRegistryConfig",
    "ToolNotFoundError",
    "ToolAlreadyExistsError",
]
