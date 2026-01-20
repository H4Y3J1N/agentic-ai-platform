"""
Base Tool Classes and Types

Tool 추상화를 위한 기반 클래스 및 타입 정의
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable, TypeVar
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class ToolCapability(Enum):
    """Tool이 제공하는 기능 유형"""
    SEARCH = "search"           # 검색 기능
    CREATE = "create"           # 생성 기능
    UPDATE = "update"           # 수정 기능
    DELETE = "delete"           # 삭제 기능
    RETRIEVE = "retrieve"       # 조회 기능
    ANALYZE = "analyze"         # 분석 기능
    TRANSFORM = "transform"     # 변환 기능
    NOTIFY = "notify"           # 알림 기능


@dataclass
class ToolConfig:
    """Tool 설정"""
    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    cache_enabled: bool = False
    cache_ttl_seconds: int = 300
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetadata:
    """Tool 메타데이터"""
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: List[ToolCapability] = field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None


@dataclass
class ToolResult:
    """Tool 실행 결과"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any, **metadata) -> "ToolResult":
        """성공 결과 생성"""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> "ToolResult":
        """실패 결과 생성"""
        return cls(success=False, error=error, metadata=metadata)


class BaseTool(ABC):
    """
    Tool 추상 기반 클래스

    모든 Tool은 이 클래스를 상속받아야 합니다.

    Example:
        class NotionSearchTool(BaseTool):
            name = "notion_search"
            description = "Search Notion documents"
            capabilities = [ToolCapability.SEARCH]

            async def execute(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
                results = await self._search(query, top_k)
                return ToolResult.ok(results)
    """

    # 서브클래스에서 오버라이드할 클래스 속성
    name: str = "base_tool"
    description: str = "Base tool"
    version: str = "1.0.0"
    capabilities: List[ToolCapability] = []
    tags: List[str] = []

    def __init__(self, config: Optional[ToolConfig] = None):
        self.config = config or ToolConfig()
        self._initialized = False
        self._logger = logging.getLogger(f"{__name__}.{self.name}")

    @property
    def metadata(self) -> ToolMetadata:
        """Tool 메타데이터 반환"""
        return ToolMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            capabilities=self.capabilities,
            tags=self.tags,
        )

    @property
    def is_enabled(self) -> bool:
        """Tool 활성화 여부"""
        return self.config.enabled

    async def initialize(self) -> None:
        """
        Tool 초기화 (선택적 오버라이드)

        리소스 할당, 연결 설정 등을 수행합니다.
        """
        self._initialized = True

    async def cleanup(self) -> None:
        """
        Tool 정리 (선택적 오버라이드)

        리소스 해제, 연결 종료 등을 수행합니다.
        """
        self._initialized = False

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Tool 실행 (필수 구현)

        Args:
            **kwargs: Tool별 파라미터

        Returns:
            ToolResult: 실행 결과
        """
        pass

    async def __call__(self, **kwargs) -> ToolResult:
        """
        Tool 호출 (with timing, error handling, retry)
        """
        if not self.is_enabled:
            return ToolResult.fail(f"Tool '{self.name}' is disabled")

        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                result = await self.execute(**kwargs)
                result.execution_time_ms = (time.time() - start_time) * 1000
                return result

            except Exception as e:
                last_error = str(e)
                self._logger.warning(
                    f"Tool '{self.name}' attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await self._delay(self.config.retry_delay_seconds)

        execution_time_ms = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            error=f"Tool '{self.name}' failed after {self.config.max_retries} attempts: {last_error}",
            execution_time_ms=execution_time_ms,
        )

    async def _delay(self, seconds: float) -> None:
        """비동기 대기"""
        import asyncio
        await asyncio.sleep(seconds)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, enabled={self.is_enabled})>"


# Type variable for tool decorator
T = TypeVar("T", bound=BaseTool)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    capabilities: Optional[List[ToolCapability]] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
) -> Callable[[Type[T]], Type[T]]:
    """
    Tool 클래스 데코레이터

    클래스 속성을 편리하게 설정합니다.

    Example:
        @tool(
            name="notion_search",
            description="Search Notion documents",
            capabilities=[ToolCapability.SEARCH],
            tags=["notion", "search"]
        )
        class NotionSearchTool(BaseTool):
            async def execute(self, query: str, **kwargs) -> ToolResult:
                ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        if name is not None:
            cls.name = name
        if description is not None:
            cls.description = description
        if capabilities is not None:
            cls.capabilities = capabilities
        if tags is not None:
            cls.tags = tags
        cls.version = version
        return cls
    return decorator
