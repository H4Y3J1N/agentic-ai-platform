"""
Parsers Package

문서 타입별 파서 모듈
"""

from .base import (
    Parser,
    ParserRegistry,
    RichText,
)

from .notion_parser import (
    NotionParser,
    NotionMeetingNoteParser,
)

from .slack_parser import (
    SlackParser,
    SlackAnnouncementParser,
)


__all__ = [
    # Base
    "Parser",
    "ParserRegistry",
    "RichText",
    # Notion
    "NotionParser",
    "NotionMeetingNoteParser",
    # Slack
    "SlackParser",
    "SlackAnnouncementParser",
]
