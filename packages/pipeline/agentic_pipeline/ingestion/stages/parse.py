"""
Parse Stage

문서 파싱 스테이지
"""

from typing import Optional, Dict, Any, Type
import logging

from .base import Stage
from ..context import PipelineContext, StageError
from ...schema import DocumentType, SourceType, ParsedContent, ParsedSection

logger = logging.getLogger(__name__)


class ParseStage(Stage):
    """문서 파싱 스테이지"""

    def __init__(self, parsers: Optional[Dict[str, "BaseParser"]] = None):
        super().__init__("ParseStage")
        self.parsers = parsers or {}
        self._default_parser = DefaultParser()

    def register_parser(
        self,
        source_type: SourceType,
        doc_type: DocumentType,
        parser: "BaseParser"
    ) -> None:
        """파서 등록"""
        key = f"{source_type.value}:{doc_type.value}"
        self.parsers[key] = parser

    def get_parser(
        self,
        source_type: SourceType,
        doc_type: Optional[DocumentType] = None
    ) -> "BaseParser":
        """파서 조회"""
        if doc_type:
            key = f"{source_type.value}:{doc_type.value}"
            if key in self.parsers:
                return self.parsers[key]

        # 소스 타입만으로 조회
        for key, parser in self.parsers.items():
            if key.startswith(source_type.value):
                return parser

        return self._default_parser

    async def process(self, context: PipelineContext) -> PipelineContext:
        """파싱 실행"""
        source = context.source_item

        # raw 데이터에서 콘텐츠 추출
        raw_content = self._extract_raw_content(source.raw_data)
        if not raw_content:
            context.mark_skip("No content to parse")
            return context

        context.raw_content = raw_content

        # 문서 타입 감지
        doc_type = self._detect_document_type(raw_content, source.metadata)
        context.document_type = doc_type

        # 파서 선택 및 실행
        parser = self.get_parser(source.source_type, doc_type)

        try:
            parsed = await parser.parse(raw_content, source.metadata)
            context.parsed_content = parsed
            logger.debug(f"Parsed document: {len(parsed.sections)} sections")

        except Exception as e:
            # 파싱 실패해도 raw_content는 유지
            context.add_error(StageError(
                stage_name=self.name,
                error_type="ParseError",
                message=str(e),
                is_fatal=False
            ))

        return context

    def _extract_raw_content(self, raw_data: Dict[str, Any]) -> Optional[str]:
        """raw 데이터에서 텍스트 콘텐츠 추출"""
        # 일반적인 키들 시도
        for key in ["content", "text", "body", "message", "plain_text"]:
            if key in raw_data and raw_data[key]:
                return str(raw_data[key])

        # blocks 형태 (Notion)
        if "blocks" in raw_data:
            return self._blocks_to_text(raw_data["blocks"])

        # 전체 데이터를 문자열로
        if isinstance(raw_data, str):
            return raw_data

        return None

    def _blocks_to_text(self, blocks: list) -> str:
        """블록 리스트를 텍스트로 변환"""
        texts = []
        for block in blocks:
            if isinstance(block, dict):
                text = block.get("text", block.get("content", ""))
                if text:
                    texts.append(str(text))
            elif isinstance(block, str):
                texts.append(block)
        return "\n\n".join(texts)

    def _detect_document_type(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> DocumentType:
        """문서 타입 감지"""
        # 메타데이터에서 힌트
        if "type" in metadata:
            type_str = str(metadata["type"]).lower()
            for dt in DocumentType:
                if dt.value in type_str:
                    return dt

        content_lower = content.lower()

        # 키워드 기반 감지
        if any(kw in content_lower for kw in ["회의록", "meeting", "minutes", "참석자"]):
            return DocumentType.MEETING_NOTE

        if any(kw in content_lower for kw in ["정책", "policy", "규정", "규칙"]):
            return DocumentType.POLICY

        if any(kw in content_lower for kw in ["faq", "자주 묻는", "질문"]):
            return DocumentType.FAQ

        if "```" in content or any(kw in content_lower for kw in ["def ", "function", "class "]):
            return DocumentType.TECHNICAL_DOC

        return DocumentType.UNKNOWN


class BaseParser:
    """파서 베이스 클래스"""

    async def parse(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """파싱 실행"""
        raise NotImplementedError


class DefaultParser(BaseParser):
    """기본 파서 (마크다운 기반)"""

    async def parse(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """마크다운 파싱"""
        import re

        sections = []
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        # 헤더로 섹션 분리
        last_pos = 0
        current_section = None

        for match in header_pattern.finditer(content):
            # 이전 섹션 콘텐츠 저장
            if current_section is not None:
                section_content = content[last_pos:match.start()].strip()
                if section_content:
                    current_section.content = section_content
                    sections.append(current_section)

            # 새 섹션 시작
            level = len(match.group(1))
            header_text = match.group(2).strip()
            current_section = ParsedSection(
                name=header_text,
                content="",
                level=level
            )
            last_pos = match.end()

        # 마지막 섹션
        if current_section is not None:
            section_content = content[last_pos:].strip()
            current_section.content = section_content
            sections.append(current_section)
        elif content.strip():
            # 헤더 없는 단일 섹션
            sections.append(ParsedSection(
                name="Content",
                content=content.strip(),
                level=0
            ))

        # 코드 블록 추출
        code_blocks = []
        code_pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
        for match in code_pattern.finditer(content):
            code_blocks.append({
                "language": match.group(1) or "text",
                "code": match.group(2)
            })

        return ParsedContent(
            sections=sections,
            code_blocks=code_blocks,
            structured_data={}
        )
