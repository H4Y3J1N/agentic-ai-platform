"""
Parser Base

파서 베이스 클래스 및 유틸리티
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ...schema import ParsedContent, ParsedSection, DocumentType, SourceType


class Parser(ABC):
    """파서 베이스 클래스"""

    source_type: SourceType = SourceType.UNKNOWN
    supported_types: List[DocumentType] = []

    @abstractmethod
    async def parse(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """
        콘텐츠 파싱

        Args:
            content: 원본 콘텐츠
            metadata: 원본 메타데이터

        Returns:
            파싱된 콘텐츠
        """
        pass

    def can_parse(self, doc_type: DocumentType) -> bool:
        """파싱 가능 여부"""
        return doc_type in self.supported_types or not self.supported_types


class ParserRegistry:
    """파서 레지스트리"""

    _parsers: Dict[str, Parser] = {}

    @classmethod
    def register(
        cls,
        source_type: SourceType,
        doc_type: Optional[DocumentType] = None
    ):
        """파서 등록 데코레이터"""
        def decorator(parser_cls):
            key = f"{source_type.value}"
            if doc_type:
                key += f":{doc_type.value}"
            cls._parsers[key] = parser_cls()
            return parser_cls
        return decorator

    @classmethod
    def get(
        cls,
        source_type: SourceType,
        doc_type: Optional[DocumentType] = None
    ) -> Optional[Parser]:
        """파서 조회"""
        # 정확한 매칭
        if doc_type:
            key = f"{source_type.value}:{doc_type.value}"
            if key in cls._parsers:
                return cls._parsers[key]

        # 소스 타입만으로 조회
        key = source_type.value
        return cls._parsers.get(key)

    @classmethod
    def list_all(cls) -> Dict[str, Parser]:
        """모든 파서 조회"""
        return cls._parsers.copy()


@dataclass
class RichText:
    """리치 텍스트 헬퍼"""
    plain_text: str
    annotations: Dict[str, Any] = None
    href: Optional[str] = None

    @classmethod
    def from_notion(cls, rich_text_list: List[Dict]) -> "RichText":
        """Notion 리치 텍스트에서 생성"""
        texts = []
        for item in rich_text_list:
            if isinstance(item, dict):
                texts.append(item.get("plain_text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return cls(plain_text="".join(texts))

    def __str__(self) -> str:
        return self.plain_text
