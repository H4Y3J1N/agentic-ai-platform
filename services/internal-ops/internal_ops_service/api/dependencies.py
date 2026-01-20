"""
API Dependencies - FastAPI Dependency Injection
"""

from fastapi import Header, HTTPException
from typing import Optional
from dataclasses import dataclass
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class User:
    """Current user context"""
    id: str
    name: str = "Anonymous"


# Internal Ops Agent 싱글톤
_internal_ops_agent = None


def get_agent_executor():
    """Get or create InternalOpsAgent instance"""
    global _internal_ops_agent

    if _internal_ops_agent is None:
        from ..agents import InternalOpsAgent

        # RAG 전략 설정 (환경변수로 오버라이드 가능)
        # "single_shot": 빠름, 기본 RAG
        # "corrective": 품질 평가 + 재검색 (기본값, 균형)
        # "self_rag": 자기 검증 + 수정 (정확)
        rag_strategy = os.environ.get("RAG_STRATEGY", "corrective")

        config = {
            # RAG Strategy
            "rag_strategy": rag_strategy,

            # Search 설정
            "use_enhanced_search": True,
            "query_rewriting": True,
            "hybrid_search": True,
            "reranking": True,

            # Corrective RAG 설정
            "max_retries": int(os.environ.get("RAG_MAX_RETRIES", "2")),
            "quality_threshold": float(os.environ.get("RAG_QUALITY_THRESHOLD", "0.7")),

            # Self-RAG 설정
            "critique_threshold": float(os.environ.get("RAG_CRITIQUE_THRESHOLD", "0.7")),
            "enable_self_critique": True,

            # 디버깅
            "verbose": os.environ.get("RAG_VERBOSE", "").lower() == "true",
        }

        _internal_ops_agent = InternalOpsAgent(config)
        logger.info(f"InternalOpsAgent initialized with RAG strategy: {rag_strategy}")

    return _internal_ops_agent


async def get_current_user(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    authorization: Optional[str] = Header(None)
) -> User:
    """Extract current user from request headers"""
    user_id = x_user_id or "anonymous"
    return User(id=user_id)
