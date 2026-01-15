"""
Slack Parser

Slack 메시지 파서
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from .base import Parser, ParserRegistry, RichText
from ...schema import ParsedContent, ParsedSection, DocumentType, SourceType


@ParserRegistry.register(SourceType.SLACK)
class SlackParser(Parser):
    """Slack 메시지 파서"""

    source_type = SourceType.SLACK
    supported_types = [
        DocumentType.MEETING_NOTE,
        DocumentType.ANNOUNCEMENT,
        DocumentType.WIKI,
    ]

    async def parse(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """Slack 메시지 파싱"""
        # JSON 형식인 경우
        if isinstance(content, str):
            try:
                import json
                data = json.loads(content)
                if isinstance(data, dict):
                    if "messages" in data:
                        return await self._parse_thread(data["messages"], metadata)
                    elif "text" in data:
                        return await self._parse_single_message(data, metadata)
            except:
                pass

            # 일반 텍스트
            return await self._parse_text(content, metadata)

        return ParsedContent(sections=[], structured_data={})

    async def _parse_thread(
        self,
        messages: List[Dict],
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """스레드 메시지 파싱"""
        sections = []
        users_mentioned = set()
        channels_mentioned = set()
        links = []
        code_blocks = []

        for i, msg in enumerate(messages):
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            ts = msg.get("ts", "")

            # 사용자 멘션 추출
            user_mentions = re.findall(r'<@(\w+)>', text)
            users_mentioned.update(user_mentions)

            # 채널 멘션 추출
            channel_mentions = re.findall(r'<#(\w+)\|?([^>]*)>', text)
            channels_mentioned.update(c[0] for c in channel_mentions)

            # 링크 추출
            msg_links = re.findall(r'<(https?://[^|>]+)(?:\|([^>]+))?>', text)
            links.extend({"url": l[0], "text": l[1] or l[0]} for l in msg_links)

            # 코드 블록 추출
            code_pattern = re.compile(r'```(\w*)\n?(.*?)```', re.DOTALL)
            for match in code_pattern.finditer(text):
                code_blocks.append({
                    "language": match.group(1) or "text",
                    "code": match.group(2)
                })

            # 텍스트 정리
            clean_text = self._clean_slack_text(text)

            # 타임스탬프 포맷
            try:
                timestamp = datetime.fromtimestamp(float(ts))
                time_str = timestamp.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = ts

            sections.append(ParsedSection(
                name=f"Message {i+1}" if i > 0 else "Original Message",
                content=f"[{time_str}] <{user}>: {clean_text}",
                level=1 if i == 0 else 2
            ))

        # 구조화된 데이터
        structured = {
            "message_count": len(messages),
            "participants": list(users_mentioned),
            "channels_mentioned": list(channels_mentioned),
            "links": links,
        }

        # 메타데이터에서 추가 정보
        if "channel" in metadata:
            structured["channel"] = metadata["channel"]
        if "thread_ts" in metadata:
            structured["thread_id"] = metadata["thread_ts"]

        return ParsedContent(
            sections=sections,
            code_blocks=code_blocks,
            structured_data=structured
        )

    async def _parse_single_message(
        self,
        message: Dict,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """단일 메시지 파싱"""
        text = message.get("text", "")
        user = message.get("user", "unknown")
        ts = message.get("ts", "")

        clean_text = self._clean_slack_text(text)

        # 코드 블록 추출
        code_blocks = []
        code_pattern = re.compile(r'```(\w*)\n?(.*?)```', re.DOTALL)
        for match in code_pattern.finditer(text):
            code_blocks.append({
                "language": match.group(1) or "text",
                "code": match.group(2)
            })

        # 타임스탬프 포맷
        try:
            timestamp = datetime.fromtimestamp(float(ts))
            time_str = timestamp.strftime("%Y-%m-%d %H:%M")
        except:
            time_str = ts

        sections = [ParsedSection(
            name="Message",
            content=f"[{time_str}] <{user}>: {clean_text}",
            level=0
        )]

        # 리액션 정보
        reactions = message.get("reactions", [])
        if reactions:
            reaction_text = ", ".join(
                f":{r['name']}: ({r.get('count', 1)})"
                for r in reactions
            )
            sections.append(ParsedSection(
                name="Reactions",
                content=reaction_text,
                level=1
            ))

        # 첨부 파일
        attachments = message.get("attachments", [])
        if attachments:
            attachment_text = "\n".join(
                f"- {a.get('title', 'Attachment')}: {a.get('text', '')}"
                for a in attachments
            )
            sections.append(ParsedSection(
                name="Attachments",
                content=attachment_text,
                level=1
            ))

        structured = {
            "user": user,
            "timestamp": ts,
            "has_reactions": len(reactions) > 0,
            "has_attachments": len(attachments) > 0,
        }

        return ParsedContent(
            sections=sections,
            code_blocks=code_blocks,
            structured_data=structured
        )

    async def _parse_text(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """일반 텍스트 파싱"""
        clean_text = self._clean_slack_text(content)

        sections = [ParsedSection(
            name="Content",
            content=clean_text,
            level=0
        )]

        return ParsedContent(
            sections=sections,
            structured_data={}
        )

    def _clean_slack_text(self, text: str) -> str:
        """Slack 텍스트 정리"""
        # 사용자 멘션 정리
        text = re.sub(r'<@(\w+)>', r'@\1', text)

        # 채널 멘션 정리
        text = re.sub(r'<#(\w+)\|([^>]+)>', r'#\2', text)
        text = re.sub(r'<#(\w+)>', r'#\1', text)

        # 링크 정리
        text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
        text = re.sub(r'<(https?://[^>]+)>', r'\1', text)

        # 특수 토큰
        text = re.sub(r'<!here>', '@here', text)
        text = re.sub(r'<!channel>', '@channel', text)
        text = re.sub(r'<!everyone>', '@everyone', text)

        return text.strip()


@ParserRegistry.register(SourceType.SLACK, DocumentType.ANNOUNCEMENT)
class SlackAnnouncementParser(SlackParser):
    """Slack 공지 전용 파서"""

    async def parse(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> ParsedContent:
        """공지 특화 파싱"""
        base_result = await super().parse(content, metadata)

        # 공지 특화 구조 추출
        structured = base_result.structured_data.copy()

        full_text = " ".join(
            s.content for s in base_result.sections
        ).lower()

        # 중요도 추출
        if any(kw in full_text for kw in ["urgent", "긴급", "important", "중요"]):
            structured["priority"] = "high"
        elif any(kw in full_text for kw in ["fyi", "참고", "info"]):
            structured["priority"] = "low"
        else:
            structured["priority"] = "normal"

        # 액션 아이템 추출
        action_items = []
        action_patterns = [
            r'(?:please|요청|부탁)[:\s]+(.+?)(?:\.|$)',
            r'(?:action|액션)[:\s]+(.+?)(?:\.|$)',
            r'(?:todo|할일)[:\s]+(.+?)(?:\.|$)',
        ]
        for pattern in action_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            action_items.extend(matches)

        if action_items:
            structured["action_items"] = action_items

        # 마감일 추출
        deadline_patterns = [
            r'(?:by|until|까지|마감)[:\s]*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(?:deadline|마감일)[:\s]*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        ]
        for pattern in deadline_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                structured["deadline"] = match.group(1)
                break

        return ParsedContent(
            sections=base_result.sections,
            code_blocks=base_result.code_blocks,
            tables=base_result.tables,
            structured_data=structured
        )
