"""
Streamlit Chat Components

재사용 가능한 Streamlit 채팅 UI 컴포넌트
"""

from typing import List, Optional, Callable, Generator
from datetime import datetime
import uuid

try:
    import streamlit as st
except ImportError:
    raise ImportError("streamlit is required. Install with: pip install streamlit")

from .types import ChatMessage, MessageRole, ChatSession, ChatConfig
from .sse_client import sync_stream_chat_response


def render_chat_message(
    message: ChatMessage,
    config: Optional[ChatConfig] = None
):
    """
    단일 채팅 메시지 렌더링

    Args:
        message: 채팅 메시지
        config: 채팅 설정
    """
    config = config or ChatConfig()

    # 아바타 결정
    if message.role == MessageRole.USER:
        avatar = config.user_avatar
    elif message.role == MessageRole.ASSISTANT:
        avatar = config.assistant_avatar
    elif message.role == MessageRole.ERROR:
        avatar = "⚠️"
    else:
        avatar = "💬"

    # 역할 이름
    role_name = "user" if message.role == MessageRole.USER else "assistant"

    with st.chat_message(role_name, avatar=avatar):
        # 메시지 내용
        st.markdown(message.content)

        # 타임스탬프 (선택적)
        if config.show_timestamps:
            st.caption(message.timestamp.strftime("%Y-%m-%d %H:%M"))

        # 소스 정보 (RAG 결과)
        if config.show_sources and message.sources:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(message.sources, 1):
                    title = source.get("title", "Untitled")
                    url = source.get("url", "")
                    score = source.get("score", source.get("relevance_score", 0))

                    if url:
                        st.markdown(f"**[{i}] [{title}]({url})** (score: {score:.3f})")
                    else:
                        st.markdown(f"**[{i}] {title}** (score: {score:.3f})")


def render_chat_history(
    session: ChatSession,
    config: Optional[ChatConfig] = None
):
    """
    채팅 히스토리 전체 렌더링

    Args:
        session: 채팅 세션
        config: 채팅 설정
    """
    config = config or ChatConfig()

    for message in session.messages:
        render_chat_message(message, config)


def render_chat_input(
    key: str = "chat_input",
    placeholder: str = "메시지를 입력하세요...",
    disabled: bool = False
) -> Optional[str]:
    """
    채팅 입력 필드 렌더링

    Args:
        key: Streamlit 위젯 키
        placeholder: 플레이스홀더 텍스트
        disabled: 비활성화 여부

    Returns:
        입력된 메시지 (없으면 None)
    """
    return st.chat_input(placeholder, key=key, disabled=disabled)


def create_chat_container():
    """
    채팅 컨테이너 생성

    Returns:
        Streamlit 컨테이너 객체
    """
    return st.container()


class ChatUI:
    """
    통합 채팅 UI 클래스

    Streamlit 앱에서 쉽게 채팅 인터페이스를 구현할 수 있는 클래스
    """

    def __init__(self, config: Optional[ChatConfig] = None):
        """
        Args:
            config: 채팅 설정
        """
        self.config = config or ChatConfig()
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Streamlit 세션 상태 초기화"""
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = ChatSession(
                session_id=str(uuid.uuid4())
            )

        if "is_streaming" not in st.session_state:
            st.session_state.is_streaming = False

    @property
    def session(self) -> ChatSession:
        """현재 세션"""
        return st.session_state.chat_session

    def render_header(self):
        """헤더 렌더링"""
        st.title(self.config.title)

    def render_sidebar(
        self,
        on_clear: Optional[Callable] = None,
        extra_content: Optional[Callable] = None
    ):
        """
        사이드바 렌더링

        Args:
            on_clear: 대화 초기화 콜백
            extra_content: 추가 컨텐츠 렌더링 함수
        """
        with st.sidebar:
            st.header("Settings")

            # 대화 초기화 버튼
            if st.button("🗑️ 대화 초기화", use_container_width=True):
                self.session.clear()
                if on_clear:
                    on_clear()
                st.rerun()

            # 세션 정보
            st.divider()
            st.caption(f"Session ID: {self.session.session_id[:8]}...")
            st.caption(f"Messages: {len(self.session.messages)}")

            # 추가 컨텐츠
            if extra_content:
                st.divider()
                extra_content()

    def render_messages(self):
        """메시지 히스토리 렌더링"""
        render_chat_history(self.session, self.config)

    def render_input(self) -> Optional[str]:
        """
        입력 필드 렌더링

        Returns:
            입력된 메시지
        """
        return render_chat_input(
            placeholder=self.config.placeholder,
            disabled=st.session_state.is_streaming
        )

    def add_user_message(self, content: str):
        """사용자 메시지 추가"""
        self.session.add_user_message(content)

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List] = None
    ):
        """어시스턴트 메시지 추가"""
        self.session.add_assistant_message(
            content,
            sources=sources
        )

    def stream_response(
        self,
        user_message: str,
        context: Optional[dict] = None
    ) -> str:
        """
        SSE 스트리밍 응답 처리

        Args:
            user_message: 사용자 메시지
            context: 추가 컨텍스트

        Returns:
            전체 응답 텍스트
        """
        api_url = f"{self.config.api_base_url}{self.config.stream_endpoint}"

        st.session_state.is_streaming = True
        full_response = ""

        try:
            # 어시스턴트 메시지 영역
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                message_placeholder = st.empty()

                # SSE 스트리밍
                for chunk in sync_stream_chat_response(
                    api_url=api_url,
                    message=user_message,
                    session_id=self.session.session_id,
                    context=context,
                    timeout=self.config.stream_timeout
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                # 최종 응답
                message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"오류가 발생했습니다: {str(e)}"
            st.error(full_response)

        finally:
            st.session_state.is_streaming = False

        return full_response

    def handle_input(
        self,
        context: Optional[dict] = None,
        on_response: Optional[Callable[[str], None]] = None
    ):
        """
        입력 처리 및 응답 생성

        Args:
            context: 추가 컨텍스트
            on_response: 응답 완료 콜백
        """
        user_input = self.render_input()

        if user_input:
            # 사용자 메시지 추가 및 표시
            self.add_user_message(user_input)
            render_chat_message(
                ChatMessage(role=MessageRole.USER, content=user_input),
                self.config
            )

            # 스트리밍 응답
            response = self.stream_response(user_input, context)

            # 어시스턴트 메시지 저장
            self.add_assistant_message(response)

            # 콜백
            if on_response:
                on_response(response)

    def run(
        self,
        context: Optional[dict] = None,
        sidebar_extra: Optional[Callable] = None
    ):
        """
        전체 채팅 UI 실행

        Args:
            context: 추가 컨텍스트
            sidebar_extra: 사이드바 추가 컨텐츠
        """
        # 헤더
        self.render_header()

        # 사이드바
        self.render_sidebar(extra_content=sidebar_extra)

        # 메시지 히스토리
        self.render_messages()

        # 입력 처리
        self.handle_input(context=context)
