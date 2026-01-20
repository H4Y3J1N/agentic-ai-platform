"""
Slack Data Models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """Slack message types"""
    MESSAGE = "message"
    THREAD_REPLY = "thread_reply"
    BOT_MESSAGE = "bot_message"
    FILE_SHARE = "file_share"


@dataclass
class SlackUser:
    """Slack user information"""
    id: str
    name: str
    real_name: str
    display_name: str
    email: Optional[str] = None
    is_bot: bool = False

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "SlackUser":
        """Create SlackUser from Slack API response"""
        profile = data.get("profile", {})
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            real_name=profile.get("real_name", data.get("real_name", "")),
            display_name=profile.get("display_name", ""),
            email=profile.get("email"),
            is_bot=data.get("is_bot", False)
        )


@dataclass
class SlackChannel:
    """Slack channel information"""
    id: str
    name: str
    is_private: bool
    is_archived: bool
    topic: str
    purpose: str
    member_count: int = 0
    created: Optional[datetime] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "SlackChannel":
        """Create SlackChannel from Slack API response"""
        created_ts = data.get("created")
        created = datetime.fromtimestamp(created_ts) if created_ts else None

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            is_private=data.get("is_private", False),
            is_archived=data.get("is_archived", False),
            topic=data.get("topic", {}).get("value", ""),
            purpose=data.get("purpose", {}).get("value", ""),
            member_count=data.get("num_members", 0),
            created=created
        )


@dataclass
class SlackMessage:
    """Slack message"""
    ts: str  # Timestamp (serves as message ID)
    channel_id: str
    user_id: Optional[str]
    text: str
    thread_ts: Optional[str] = None
    reply_count: int = 0
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    subtype: Optional[str] = None

    @property
    def timestamp(self) -> datetime:
        """Convert Slack ts to datetime"""
        return datetime.fromtimestamp(float(self.ts))

    @property
    def is_thread_parent(self) -> bool:
        """Check if this message has replies (is a thread parent)"""
        return self.reply_count > 0

    @property
    def is_thread_reply(self) -> bool:
        """Check if this message is a thread reply"""
        return self.thread_ts is not None and self.thread_ts != self.ts

    @property
    def message_type(self) -> MessageType:
        """Determine message type"""
        if self.subtype == "bot_message":
            return MessageType.BOT_MESSAGE
        if self.subtype == "file_share":
            return MessageType.FILE_SHARE
        if self.is_thread_reply:
            return MessageType.THREAD_REPLY
        return MessageType.MESSAGE

    @classmethod
    def from_api_response(cls, data: Dict[str, Any], channel_id: str) -> "SlackMessage":
        """Create SlackMessage from Slack API response"""
        return cls(
            ts=data.get("ts", ""),
            channel_id=channel_id,
            user_id=data.get("user"),
            text=data.get("text", ""),
            thread_ts=data.get("thread_ts"),
            reply_count=data.get("reply_count", 0),
            reactions=data.get("reactions", []),
            files=data.get("files", []),
            attachments=data.get("attachments", []),
            subtype=data.get("subtype")
        )


@dataclass
class SlackThread:
    """Slack thread (parent message + replies)"""
    parent: SlackMessage
    replies: List[SlackMessage] = field(default_factory=list)

    @property
    def all_messages(self) -> List[SlackMessage]:
        """Get all messages in thread order"""
        return [self.parent] + self.replies

    @property
    def total_count(self) -> int:
        """Total messages in thread"""
        return 1 + len(self.replies)

    def to_text(self, include_usernames: bool = True) -> str:
        """Convert thread to readable text"""
        lines = []
        for msg in self.all_messages:
            prefix = f"[{msg.user_id}] " if include_usernames and msg.user_id else ""
            lines.append(f"{prefix}{msg.text}")
        return "\n".join(lines)


@dataclass
class SyncedSlackDocument:
    """Slack document ready for RAG indexing"""
    id: str  # channel_id + "_" + ts
    channel_id: str
    channel_name: str
    message_ts: str
    user_id: Optional[str]
    user_name: Optional[str]
    content: str
    timestamp: datetime
    thread_ts: Optional[str] = None
    is_thread: bool = False
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for indexing"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": {
                "source": "slack",
                "channel_id": self.channel_id,
                "channel_name": self.channel_name,
                "message_ts": self.message_ts,
                "user_id": self.user_id,
                "user_name": self.user_name,
                "timestamp": self.timestamp.isoformat(),
                "thread_ts": self.thread_ts,
                "is_thread": self.is_thread,
                "url": self.url,
                **self.metadata
            }
        }
