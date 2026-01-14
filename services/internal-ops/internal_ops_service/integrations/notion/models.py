"""
Notion Data Models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class BlockType(Enum):
    """Notion block types"""
    PARAGRAPH = "paragraph"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    BULLETED_LIST = "bulleted_list_item"
    NUMBERED_LIST = "numbered_list_item"
    TODO = "to_do"
    TOGGLE = "toggle"
    CODE = "code"
    QUOTE = "quote"
    CALLOUT = "callout"
    DIVIDER = "divider"
    TABLE = "table"
    TABLE_ROW = "table_row"
    IMAGE = "image"
    FILE = "file"
    BOOKMARK = "bookmark"
    EMBED = "embed"
    CHILD_PAGE = "child_page"
    CHILD_DATABASE = "child_database"
    UNSUPPORTED = "unsupported"


class RichText:
    """Rich text content helper"""

    @staticmethod
    def extract_text(rich_texts: List[Dict]) -> str:
        """Extract plain text from rich text array"""
        if not rich_texts:
            return ""
        return "".join(rt.get("plain_text", "") for rt in rich_texts)


@dataclass
class NotionBlock:
    """Notion block (content unit)"""
    id: str
    type: BlockType
    created_time: datetime
    last_edited_time: datetime
    has_children: bool
    content: str
    raw_data: Dict[str, Any]
    children: List["NotionBlock"] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, data: Dict) -> "NotionBlock":
        block_type_str = data.get("type", "unsupported")
        try:
            block_type = BlockType(block_type_str)
        except ValueError:
            block_type = BlockType.UNSUPPORTED

        content = cls._extract_content(data, block_type_str)

        created_time = datetime.now()
        last_edited_time = datetime.now()

        if data.get("created_time"):
            try:
                created_time = datetime.fromisoformat(
                    data["created_time"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        if data.get("last_edited_time"):
            try:
                last_edited_time = datetime.fromisoformat(
                    data["last_edited_time"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        return cls(
            id=data.get("id", ""),
            type=block_type,
            created_time=created_time,
            last_edited_time=last_edited_time,
            has_children=data.get("has_children", False),
            content=content,
            raw_data=data
        )

    @classmethod
    def _extract_content(cls, data: Dict, block_type: str) -> str:
        """Extract plain text content from block"""
        type_data = data.get(block_type, {})

        # Most text blocks have "rich_text" field
        if "rich_text" in type_data:
            return RichText.extract_text(type_data["rich_text"])

        # Child page/database
        if block_type == "child_page":
            return type_data.get("title", "")
        if block_type == "child_database":
            return type_data.get("title", "")

        # Callout/quote with text field
        if "text" in type_data:
            return RichText.extract_text(type_data["text"])

        return ""

    def to_text(self, include_children: bool = True, indent: int = 0) -> str:
        """Convert block tree to plain text"""
        prefix = "  " * indent
        lines = []

        if self.content:
            lines.append(f"{prefix}{self.content}")

        if include_children:
            for child in self.children:
                child_text = child.to_text(include_children=True, indent=indent + 1)
                if child_text:
                    lines.append(child_text)

        return "\n".join(lines)


@dataclass
class NotionPage:
    """Notion page"""
    id: str
    title: str
    url: str
    created_time: datetime
    last_edited_time: datetime
    parent_type: str
    parent_id: Optional[str]
    properties: Dict[str, Any]
    icon: Optional[str] = None
    cover: Optional[str] = None
    archived: bool = False

    @classmethod
    def from_api_response(cls, data: Dict) -> "NotionPage":
        title = cls._extract_title(data.get("properties", {}))

        # Parse parent
        parent = data.get("parent", {})
        parent_type = "workspace"
        parent_id = None

        if parent:
            if "page_id" in parent:
                parent_type = "page_id"
                parent_id = parent["page_id"]
            elif "database_id" in parent:
                parent_type = "database_id"
                parent_id = parent["database_id"]
            elif "workspace" in parent:
                parent_type = "workspace"

        created_time = datetime.now()
        last_edited_time = datetime.now()

        if data.get("created_time"):
            try:
                created_time = datetime.fromisoformat(
                    data["created_time"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        if data.get("last_edited_time"):
            try:
                last_edited_time = datetime.fromisoformat(
                    data["last_edited_time"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        return cls(
            id=data.get("id", ""),
            title=title,
            url=data.get("url", ""),
            created_time=created_time,
            last_edited_time=last_edited_time,
            parent_type=parent_type,
            parent_id=parent_id,
            properties=data.get("properties", {}),
            icon=cls._extract_icon(data.get("icon")),
            cover=cls._extract_cover(data.get("cover")),
            archived=data.get("archived", False)
        )

    @classmethod
    def _extract_title(cls, properties: Dict) -> str:
        """Extract title from page properties"""
        # Try common title property names
        for key in ["title", "Title", "Name", "name"]:
            if key in properties:
                title_prop = properties[key]
                if isinstance(title_prop, dict) and title_prop.get("type") == "title":
                    return RichText.extract_text(title_prop.get("title", []))

        # Fallback: first title-type property
        for prop in properties.values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                return RichText.extract_text(prop.get("title", []))

        return "Untitled"

    @classmethod
    def _extract_icon(cls, icon_data: Optional[Dict]) -> Optional[str]:
        if not icon_data:
            return None
        icon_type = icon_data.get("type")
        if icon_type == "emoji":
            return icon_data.get("emoji")
        elif icon_type == "external":
            return icon_data.get("external", {}).get("url")
        return None

    @classmethod
    def _extract_cover(cls, cover_data: Optional[Dict]) -> Optional[str]:
        if not cover_data:
            return None
        cover_type = cover_data.get("type")
        if cover_type == "external":
            return cover_data.get("external", {}).get("url")
        return None


@dataclass
class NotionDatabase:
    """Notion database"""
    id: str
    title: str
    description: str
    url: str
    created_time: datetime
    last_edited_time: datetime
    properties: Dict[str, Any]

    @classmethod
    def from_api_response(cls, data: Dict) -> "NotionDatabase":
        created_time = datetime.now()
        last_edited_time = datetime.now()

        if data.get("created_time"):
            try:
                created_time = datetime.fromisoformat(
                    data["created_time"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        if data.get("last_edited_time"):
            try:
                last_edited_time = datetime.fromisoformat(
                    data["last_edited_time"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        return cls(
            id=data.get("id", ""),
            title=RichText.extract_text(data.get("title", [])),
            description=RichText.extract_text(data.get("description", [])),
            url=data.get("url", ""),
            created_time=created_time,
            last_edited_time=last_edited_time,
            properties=data.get("properties", {})
        )
