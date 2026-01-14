"""
Notion Parser

Notion 문서 파서
"""

from typing import Dict, Any, List, Optional
import re

from .base import Parser, ParserRegistry, RichText
from ...schema import ParsedContent, ParsedSection, DocumentType, SourceType


@ParserRegistry.register(SourceType.NOTION)
class NotionParser(Parser):
    """Notion 문서 파서"""

    source_type = SourceType.NOTION
    supported_types = [
        DocumentType.POLICY,
        DocumentType.MEETING_NOTE,
        DocumentType.WIKI,
        DocumentType.TECHNICAL_DOC,
        DocumentType.FAQ,
    ]

    async def parse(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """Notion 콘텐츠 파싱"""
        # blocks 형태인 경우
        if isinstance(content, str):
            try:
                import json
                data = json.loads(content)
                if isinstance(data, dict) and "blocks" in data:
                    return await self._parse_blocks(data["blocks"], metadata)
            except:
                pass

            # 마크다운 형태
            return await self._parse_markdown(content, metadata)

        return ParsedContent(sections=[], structured_data={})

    async def _parse_blocks(
        self,
        blocks: List[Dict],
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """Notion 블록 파싱"""
        sections = []
        current_section = None
        code_blocks = []
        tables = []

        for block in blocks:
            block_type = block.get("type", "")

            # 헤더 블록
            if block_type.startswith("heading_"):
                level = int(block_type[-1])
                text = self._extract_text(block.get(block_type, {}))

                if current_section:
                    sections.append(current_section)

                current_section = ParsedSection(
                    name=text,
                    content="",
                    level=level
                )

            # 텍스트 블록
            elif block_type == "paragraph":
                text = self._extract_text(block.get("paragraph", {}))
                if current_section:
                    current_section.content += text + "\n\n"
                else:
                    current_section = ParsedSection(
                        name="Content",
                        content=text + "\n\n",
                        level=0
                    )

            # 리스트 블록
            elif block_type in ["bulleted_list_item", "numbered_list_item"]:
                text = self._extract_text(block.get(block_type, {}))
                prefix = "• " if block_type == "bulleted_list_item" else "- "
                if current_section:
                    current_section.content += prefix + text + "\n"

            # 코드 블록
            elif block_type == "code":
                code_data = block.get("code", {})
                language = code_data.get("language", "text")
                code_text = self._extract_text(code_data)
                code_blocks.append({
                    "language": language,
                    "code": code_text
                })
                if current_section:
                    current_section.content += f"```{language}\n{code_text}\n```\n\n"

            # 테이블 블록
            elif block_type == "table":
                table_data = self._parse_table(block)
                if table_data:
                    tables.append(table_data)

            # 할일 블록
            elif block_type == "to_do":
                todo_data = block.get("to_do", {})
                checked = todo_data.get("checked", False)
                text = self._extract_text(todo_data)
                checkbox = "[x]" if checked else "[ ]"
                if current_section:
                    current_section.content += f"{checkbox} {text}\n"

            # 인용 블록
            elif block_type == "quote":
                text = self._extract_text(block.get("quote", {}))
                if current_section:
                    current_section.content += f"> {text}\n\n"

            # 콜아웃 블록
            elif block_type == "callout":
                callout = block.get("callout", {})
                icon = callout.get("icon", {}).get("emoji", "")
                text = self._extract_text(callout)
                if current_section:
                    current_section.content += f"{icon} {text}\n\n"

        # 마지막 섹션 추가
        if current_section:
            sections.append(current_section)

        # 구조화된 데이터 추출
        structured = self._extract_structured_data(blocks, metadata)

        return ParsedContent(
            sections=sections,
            code_blocks=code_blocks,
            tables=tables,
            structured_data=structured
        )

    async def _parse_markdown(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """마크다운 형식 파싱"""
        sections = []
        code_blocks = []

        # 헤더로 섹션 분리
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        last_pos = 0
        current_section = None

        for match in header_pattern.finditer(content):
            if current_section:
                section_content = content[last_pos:match.start()].strip()
                current_section.content = section_content
                sections.append(current_section)

            level = len(match.group(1))
            header_text = match.group(2).strip()
            current_section = ParsedSection(
                name=header_text,
                content="",
                level=level
            )
            last_pos = match.end()

        # 마지막 섹션
        if current_section:
            current_section.content = content[last_pos:].strip()
            sections.append(current_section)
        elif content.strip():
            sections.append(ParsedSection(
                name="Content",
                content=content.strip(),
                level=0
            ))

        # 코드 블록 추출
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

    def _extract_text(self, block_content: Dict) -> str:
        """블록에서 텍스트 추출"""
        rich_text = block_content.get("rich_text", [])
        if not rich_text:
            rich_text = block_content.get("text", [])
        return RichText.from_notion(rich_text).plain_text

    def _parse_table(self, block: Dict) -> Optional[List[List[str]]]:
        """테이블 블록 파싱"""
        # Notion API에서는 테이블 자식 블록을 별도로 조회해야 함
        # 여기서는 기본 구조만 반환
        return None

    def _extract_structured_data(
        self,
        blocks: List[Dict],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """구조화된 데이터 추출"""
        structured = {}

        # 프로퍼티에서 추출
        properties = metadata.get("properties", {})
        for prop_name, prop_value in properties.items():
            if isinstance(prop_value, dict):
                prop_type = prop_value.get("type", "")
                if prop_type == "title":
                    titles = prop_value.get("title", [])
                    structured["title"] = RichText.from_notion(titles).plain_text
                elif prop_type == "select":
                    select = prop_value.get("select", {})
                    if select:
                        structured[prop_name] = select.get("name", "")
                elif prop_type == "multi_select":
                    options = prop_value.get("multi_select", [])
                    structured[prop_name] = [o.get("name", "") for o in options]
                elif prop_type == "date":
                    date = prop_value.get("date", {})
                    if date:
                        structured[prop_name] = date.get("start", "")

        return structured


@ParserRegistry.register(SourceType.NOTION, DocumentType.MEETING_NOTE)
class NotionMeetingNoteParser(NotionParser):
    """Notion 회의록 전용 파서"""

    async def parse(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """회의록 특화 파싱"""
        base_result = await super().parse(content, metadata)

        # 회의록 특화 구조 추출
        structured = base_result.structured_data.copy()

        full_text = " ".join(
            s.content for s in base_result.sections
        ).lower()

        # 참석자 추출
        attendees_pattern = r'(?:참석자|attendees?|participants?)[:\s]+([^\n]+)'
        attendees_match = re.search(attendees_pattern, full_text, re.IGNORECASE)
        if attendees_match:
            structured["attendees"] = [
                a.strip() for a in attendees_match.group(1).split(",")
            ]

        # 액션 아이템 추출
        action_items = []
        for section in base_result.sections:
            if "action" in section.name.lower() or "todo" in section.name.lower():
                items = re.findall(r'[-•]\s*(.+)', section.content)
                action_items.extend(items)
        if action_items:
            structured["action_items"] = action_items

        # 결정 사항 추출
        decisions = []
        for section in base_result.sections:
            if "결정" in section.name or "decision" in section.name.lower():
                items = re.findall(r'[-•]\s*(.+)', section.content)
                decisions.extend(items)
        if decisions:
            structured["decisions"] = decisions

        return ParsedContent(
            sections=base_result.sections,
            code_blocks=base_result.code_blocks,
            tables=base_result.tables,
            structured_data=structured
        )
