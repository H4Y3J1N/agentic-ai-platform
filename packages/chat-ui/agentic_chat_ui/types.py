"""
Chat UI Types

채팅 관련 데이터 타입 정의
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class MessageRole(str, Enum):
    """메시지 역할"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class ChatMessage:
    """채팅 메시지"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 선택적 필드
    message_id: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None  # RAG 소스
    is_streaming: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "message_id": self.message_id,
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """딕셔너리에서 생성"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            metadata=data.get("metadata", {}),
            message_id=data.get("message_id"),
            sources=data.get("sources"),
        )


@dataclass
class ChatSession:
    """채팅 세션"""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 사용자 정보
    user_id: Optional[str] = None
    user_name: Optional[str] = None

    def add_message(self, message: ChatMessage):
        """메시지 추가"""
        self.messages.append(message)

    def add_user_message(self, content: str, **kwargs):
        """사용자 메시지 추가"""
        self.add_message(ChatMessage(
            role=MessageRole.USER,
            content=content,
            **kwargs
        ))

    def add_assistant_message(self, content: str, **kwargs):
        """어시스턴트 메시지 추가"""
        self.add_message(ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            **kwargs
        ))

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """LLM용 히스토리 포맷"""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.messages
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM)
        ]

    def clear(self):
        """대화 초기화"""
        self.messages.clear()


@dataclass
class StreamChunk:
    """스트리밍 청크"""
    content: str
    is_final: bool = False
    chunk_type: str = "text"  # text, source, error
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChatConfig(BaseModel):
    """채팅 설정"""
    # API 설정
    api_base_url: str = "http://localhost:8000"
    chat_endpoint: str = "/agent/chat"
    stream_endpoint: str = "/agent/chat/stream"

    # UI 설정
    title: str = "AI Chat Assistant"
    placeholder: str = "메시지를 입력하세요..."
    max_history: int = 50

    # 스트리밍 설정
    enable_streaming: bool = True
    stream_timeout: float = 60.0

    # 스타일 설정
    user_avatar: str = "👤"
    assistant_avatar: str = "🤖"
    show_timestamps: bool = False
    show_sources: bool = True


class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """채팅 응답"""
    message: str
    session_id: str
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
