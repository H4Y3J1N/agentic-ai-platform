"""
Internal Ops Chat Application

Streamlit 기반 사내 AI 챗봇 웹 인터페이스

실행:
    streamlit run app/streamlit_app.py --server.port 8501
"""

import os
import sys
from pathlib import Path

# Add paths
SERVICE_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = SERVICE_ROOT.parent.parent
sys.path.insert(0, str(SERVICE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "chat-ui"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))

import streamlit as st

from agentic_chat_ui import ChatUI, ChatConfig, ChatMessage, MessageRole

# Page config
st.set_page_config(
    page_title="Internal Ops Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
    }
    .stChatInput {
        padding-bottom: 1rem;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    .source-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #f0f2f6;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def get_api_config() -> ChatConfig:
    """API 설정 로드"""
    api_host = os.environ.get("API_HOST", "localhost")
    api_port = os.environ.get("API_PORT", "8000")

    return ChatConfig(
        api_base_url=f"http://{api_host}:{api_port}",
        chat_endpoint="/agent/chat",
        stream_endpoint="/agent/chat/stream",
        title="🤖 Internal Ops Assistant",
        placeholder="질문을 입력하세요... (예: 일본 프로젝트 진행상황 알려줘)",
        max_history=50,
        enable_streaming=True,
        stream_timeout=120.0,
        user_avatar="👤",
        assistant_avatar="🤖",
        show_timestamps=False,
        show_sources=True
    )


def render_sidebar_content():
    """사이드바 추가 컨텐츠"""
    st.subheader("📋 Quick Actions")

    # 빠른 질문 버튼
    quick_questions = [
        "프로젝트 현황 요약해줘",
        "최근 미팅 내용 알려줘",
        "진행 중인 태스크 목록",
    ]

    for q in quick_questions:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            st.session_state.quick_question = q

    st.divider()

    # 검색 모드 설정
    st.subheader("⚙️ Search Settings")

    st.checkbox(
        "Query Rewriting",
        value=True,
        key="enable_rewriting",
        help="LLM을 사용하여 쿼리를 다양하게 변형합니다"
    )

    st.checkbox(
        "Hybrid Search",
        value=True,
        key="enable_hybrid",
        help="키워드 + 시맨틱 검색을 결합합니다"
    )

    st.checkbox(
        "Reranking",
        value=True,
        key="enable_reranking",
        help="Cross-encoder로 결과를 재순위화합니다"
    )

    st.divider()

    # 데이터 소스 선택
    st.subheader("📂 Data Sources")

    st.checkbox("Notion", value=True, key="source_notion")
    st.checkbox("Slack", value=True, key="source_slack")


def get_context() -> dict:
    """검색 컨텍스트 생성"""
    return {
        "query_rewriting": st.session_state.get("enable_rewriting", True),
        "hybrid_search": st.session_state.get("enable_hybrid", True),
        "reranking": st.session_state.get("enable_reranking", True),
        "sources": {
            "notion": st.session_state.get("source_notion", True),
            "slack": st.session_state.get("source_slack", True),
        }
    }


def main():
    """메인 앱 실행"""
    # 설정 로드
    config = get_api_config()

    # ChatUI 인스턴스
    chat_ui = ChatUI(config)

    # 헤더
    chat_ui.render_header()

    # 부제목
    st.caption("Notion과 Slack 데이터를 활용한 사내 AI 어시스턴트")

    # 사이드바
    chat_ui.render_sidebar(extra_content=render_sidebar_content)

    # Quick question 처리
    if "quick_question" in st.session_state:
        quick_q = st.session_state.pop("quick_question")
        chat_ui.add_user_message(quick_q)
        st.rerun()

    # 메시지 히스토리
    chat_ui.render_messages()

    # 입력 및 응답 처리
    chat_ui.handle_input(context=get_context())


if __name__ == "__main__":
    main()
