"""
Tool Registry - Tool 등록 및 관리

Tool을 중앙에서 관리하고 조회하는 Registry 패턴 구현
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Iterator, Any, Callable
import importlib
import pkgutil
import logging

from .base import BaseTool, ToolCapability, ToolConfig, ToolMetadata

logger = logging.getLogger(__name__)


class ToolNotFoundError(Exception):
    """Tool을 찾을 수 없을 때 발생"""
    pass


class ToolAlreadyExistsError(Exception):
    """이미 등록된 Tool을 다시 등록하려 할 때 발생"""
    pass


@dataclass
class ToolRegistryConfig:
    """Registry 설정"""
    allow_override: bool = False  # 기존 Tool 덮어쓰기 허용
    auto_initialize: bool = True  # 등록 시 자동 초기화
    lazy_loading: bool = True     # 지연 로딩 사용
    default_tool_config: Optional[ToolConfig] = None


class ToolRegistry:
    """
    Tool Registry

    Tool 인스턴스를 중앙에서 관리합니다.

    Features:
    - Tool 등록/해제
    - 이름/태그/기능별 조회
    - 자동 초기화/정리
    - 지연 로딩 지원
    - 패키지 자동 검색

    Example:
        registry = ToolRegistry()

        # 인스턴스 등록
        registry.register(NotionSearchTool())
        registry.register(SlackSearchTool())

        # 조회
        tool = registry.get("notion_search")
        search_tools = registry.get_by_capability(ToolCapability.SEARCH)

        # 실행
        result = await registry.execute("notion_search", query="프로젝트")

        # 설정 기반 등록
        registry.register_from_config({
            "notion_search": {"enabled": True, "timeout_seconds": 30},
            "slack_search": {"enabled": False},
        })
    """

    def __init__(self, config: Optional[ToolRegistryConfig] = None):
        self.config = config or ToolRegistryConfig()
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}  # For lazy loading
        self._tool_configs: Dict[str, ToolConfig] = {}

    def register(
        self,
        tool: BaseTool,
        name: Optional[str] = None,
        override: bool = False,
    ) -> "ToolRegistry":
        """
        Tool 인스턴스 등록

        Args:
            tool: Tool 인스턴스
            name: 등록 이름 (기본: tool.name)
            override: 기존 Tool 덮어쓰기

        Returns:
            self (for chaining)

        Raises:
            ToolAlreadyExistsError: 이미 존재하고 override=False인 경우
        """
        tool_name = name or tool.name

        if tool_name in self._tools and not (override or self.config.allow_override):
            raise ToolAlreadyExistsError(
                f"Tool '{tool_name}' already registered. Use override=True to replace."
            )

        # Apply config if exists
        if tool_name in self._tool_configs:
            tool.config = self._tool_configs[tool_name]

        self._tools[tool_name] = tool
        logger.info(f"Registered tool: {tool_name}")

        return self

    def register_class(
        self,
        tool_class: Type[BaseTool],
        name: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ) -> "ToolRegistry":
        """
        Tool 클래스 등록 (지연 로딩)

        실제 사용 시점에 인스턴스화됩니다.

        Args:
            tool_class: Tool 클래스
            name: 등록 이름
            config: Tool 설정
        """
        tool_name = name or tool_class.name

        self._tool_classes[tool_name] = tool_class
        if config:
            self._tool_configs[tool_name] = config

        logger.info(f"Registered tool class: {tool_name} (lazy)")

        return self

    def register_many(self, tools: List[BaseTool]) -> "ToolRegistry":
        """여러 Tool 한번에 등록"""
        for tool in tools:
            self.register(tool)
        return self

    def unregister(self, name: str) -> Optional[BaseTool]:
        """
        Tool 등록 해제

        Args:
            name: Tool 이름

        Returns:
            제거된 Tool (없으면 None)
        """
        tool = self._tools.pop(name, None)
        self._tool_classes.pop(name, None)

        if tool:
            logger.info(f"Unregistered tool: {name}")

        return tool

    def get(self, name: str) -> BaseTool:
        """
        Tool 조회

        Args:
            name: Tool 이름

        Returns:
            Tool 인스턴스

        Raises:
            ToolNotFoundError: Tool이 없는 경우
        """
        # 이미 인스턴스화된 경우
        if name in self._tools:
            return self._tools[name]

        # 지연 로딩
        if name in self._tool_classes:
            config = self._tool_configs.get(name, self.config.default_tool_config)
            tool = self._tool_classes[name](config)
            self._tools[name] = tool
            logger.info(f"Lazy-loaded tool: {name}")
            return tool

        raise ToolNotFoundError(
            f"Tool '{name}' not found. Available: {list(self.available_tools)}"
        )

    def get_optional(self, name: str) -> Optional[BaseTool]:
        """Tool 조회 (없으면 None 반환)"""
        try:
            return self.get(name)
        except ToolNotFoundError:
            return None

    def has(self, name: str) -> bool:
        """Tool 존재 여부"""
        return name in self._tools or name in self._tool_classes

    def get_by_capability(self, capability: ToolCapability) -> List[BaseTool]:
        """특정 기능을 가진 Tool 목록 조회"""
        return [
            tool for tool in self.all_tools
            if capability in tool.capabilities
        ]

    def get_by_tag(self, tag: str) -> List[BaseTool]:
        """특정 태그를 가진 Tool 목록 조회"""
        return [
            tool for tool in self.all_tools
            if tag in tool.tags
        ]

    def get_enabled(self) -> List[BaseTool]:
        """활성화된 Tool 목록"""
        return [tool for tool in self.all_tools if tool.is_enabled]

    @property
    def available_tools(self) -> List[str]:
        """사용 가능한 Tool 이름 목록"""
        return list(set(self._tools.keys()) | set(self._tool_classes.keys()))

    @property
    def all_tools(self) -> List[BaseTool]:
        """모든 Tool 인스턴스 (지연 로딩 포함)"""
        # Instantiate all lazy tools
        for name in self._tool_classes:
            if name not in self._tools:
                self.get(name)
        return list(self._tools.values())

    @property
    def metadata(self) -> Dict[str, ToolMetadata]:
        """모든 Tool의 메타데이터"""
        return {name: tool.metadata for name, tool in self._tools.items()}

    async def execute(self, tool_name: str, **kwargs) -> Any:
        """
        Tool 실행

        Args:
            tool_name: Tool 이름
            **kwargs: Tool 파라미터

        Returns:
            ToolResult
        """
        tool = self.get(tool_name)
        return await tool(**kwargs)

    async def initialize_all(self) -> None:
        """모든 Tool 초기화"""
        for tool in self.all_tools:
            if not tool._initialized:
                await tool.initialize()
                logger.info(f"Initialized tool: {tool.name}")

    async def cleanup_all(self) -> None:
        """모든 Tool 정리"""
        for tool in self._tools.values():
            if tool._initialized:
                await tool.cleanup()
                logger.info(f"Cleaned up tool: {tool.name}")

    def configure(self, configs: Dict[str, Dict[str, Any]]) -> "ToolRegistry":
        """
        Tool 설정 적용

        Args:
            configs: {tool_name: config_dict} 형태의 설정

        Example:
            registry.configure({
                "notion_search": {"enabled": True, "timeout_seconds": 30},
                "slack_search": {"enabled": False},
            })
        """
        for name, config_dict in configs.items():
            tool_config = ToolConfig(**config_dict)
            self._tool_configs[name] = tool_config

            # 이미 등록된 Tool에 적용
            if name in self._tools:
                self._tools[name].config = tool_config

        return self

    def discover_tools(
        self,
        package_name: str,
        base_class: Type[BaseTool] = BaseTool,
    ) -> "ToolRegistry":
        """
        패키지에서 Tool 클래스 자동 검색

        Args:
            package_name: 검색할 패키지 이름
            base_class: 검색할 기반 클래스

        Example:
            registry.discover_tools("my_service.tools")
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.warning(f"Failed to import package {package_name}: {e}")
            return self

        for importer, module_name, is_pkg in pkgutil.walk_packages(
            package.__path__,
            prefix=f"{package_name}.",
        ):
            try:
                module = importlib.import_module(module_name)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # BaseTool 서브클래스인지 확인
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, base_class)
                        and attr is not base_class
                        and not getattr(attr, "_abstract", False)
                    ):
                        self.register_class(attr)

            except ImportError as e:
                logger.warning(f"Failed to import module {module_name}: {e}")

        return self

    def __iter__(self) -> Iterator[BaseTool]:
        """Tool 순회"""
        return iter(self.all_tools)

    def __len__(self) -> int:
        """등록된 Tool 수"""
        return len(self.available_tools)

    def __contains__(self, name: str) -> bool:
        """Tool 존재 여부"""
        return self.has(name)

    def __repr__(self) -> str:
        return f"<ToolRegistry(tools={self.available_tools})>"


# Global registry singleton
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """전역 Registry 인스턴스 반환"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: BaseTool) -> BaseTool:
    """전역 Registry에 Tool 등록 (데코레이터용)"""
    get_global_registry().register(tool)
    return tool
