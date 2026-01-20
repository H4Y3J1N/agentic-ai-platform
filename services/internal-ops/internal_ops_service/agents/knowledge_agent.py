"""
Knowledge Agent - Answers questions using RAG over internal knowledge bases

Advanced RAG 전략을 사용하여 문서 기반 질의응답 수행:
- SingleShotRAG: 기본 RAG (검색 → 답변)
- CorrectiveRAG: 품질 평가 + 자동 재검색
- SelfRAG: 자기 검증 + 답변 수정
"""

from typing import List, Dict, Any, Optional, AsyncIterator
import os
import sys
import logging
from pathlib import Path

# Add core package to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))

from agentic_core.observability import Tracer, get_tracer
from agentic_core.rag import (
    create_rag_strategy,
    RAGStrategyConfig,
    RAGResult,
    RAGStrategyType,
)

logger = logging.getLogger(__name__)

# Default model - Gemini
DEFAULT_MODEL = "gemini/gemini-1.5-flash"


class KnowledgeAgent:
    """
    Knowledge Agent - Advanced RAG 기반 Q&A

    지원하는 RAG 전략:
    - "single_shot": 기본 RAG (빠름)
    - "corrective": 품질 평가 후 필요시 재검색 (균형)
    - "self_rag": 자기 검증 후 답변 수정 (정확)

    Config options:
        rag_strategy: str = "corrective"  # 사용할 RAG 전략
        model: str = "gemini/gemini-1.5-flash"
        use_enhanced_search: bool = True
        query_rewriting: bool = True
        hybrid_search: bool = True
        reranking: bool = True
        max_retries: int = 2  # Corrective RAG 재시도 횟수
        critique_threshold: float = 0.7  # Self-RAG 검증 임계값
    """

    NAME = "knowledge_agent"
    DESCRIPTION = "Answers questions using advanced RAG strategies"
    CAPABILITIES = [
        "notion_search",
        "knowledge_qa",
        "document_retrieval",
        "corrective_rag",
        "self_rag"
    ]

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Search tool (lazy init)
        self._search_tool = None

        # RAG strategy (lazy init)
        self._rag_strategy = None

        # Initialize Langfuse tracer
        self._tracer: Tracer = get_tracer()

        # Configuration
        self._model = self.config.get("model", DEFAULT_MODEL)
        self._strategy_type = self.config.get("rag_strategy", "corrective")
        self._use_enhanced_search = self.config.get("use_enhanced_search", True)

        logger.info(f"KnowledgeAgent configured with strategy: {self._strategy_type}")

    async def _ensure_initialized(self):
        """Lazy initialization of search tool and RAG strategy"""
        if self._search_tool is None:
            if self._use_enhanced_search:
                from ..tools import create_enhanced_search_tool

                self._search_tool = create_enhanced_search_tool(
                    collection_name="internal_ops_notion",
                    config_dict={
                        "query_rewriting_enabled": self.config.get("query_rewriting", True),
                        "hybrid_search_enabled": self.config.get("hybrid_search", True),
                        "reranking_enabled": self.config.get("reranking", True),
                        "query_rewriting_model": self._model,
                    }
                )
                logger.info("Using EnhancedSearchTool")
            else:
                from ..tools import NotionSearchTool
                self._search_tool = NotionSearchTool(self.config)
                logger.info("Using basic NotionSearchTool")

        if self._rag_strategy is None:
            # RAG Strategy 설정
            strategy_config = RAGStrategyConfig(
                llm_model=self._model,
                llm_temperature=0.3,
                top_k=5,
                max_retries=self.config.get("max_retries", 2),
                quality_threshold=self.config.get("quality_threshold", 0.7),
                critique_threshold=self.config.get("critique_threshold", 0.7),
                enable_self_critique=self.config.get("enable_self_critique", True),
                verbose=self.config.get("verbose", False),
            )

            self._rag_strategy = create_rag_strategy(
                strategy_type=self._strategy_type,
                config=strategy_config
            )
            logger.info(f"RAG Strategy initialized: {self._strategy_type}")

    async def _search_fn(self, query: str, top_k: int) -> List[Dict]:
        """검색 함수 - RAG Strategy에서 사용"""
        results = await self._search_tool.search(query=query, top_k=top_k)
        return results

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute knowledge query using configured RAG strategy.

        Args:
            task: The question or task to answer
            context: Optional context (user_id, domain, etc.)

        Returns:
            AI-generated answer with citations
        """
        await self._ensure_initialized()

        context = context or {}

        # Start Langfuse trace
        user_id = context.get("user_id")
        session_id = context.get("session_id")

        async with self._tracer.trace(
            name="knowledge_agent",
            user_id=user_id,
            session_id=session_id,
            metadata={
                "query_preview": task[:100],
                "rag_strategy": self._strategy_type
            },
            tags=["knowledge_qa", "rag", self._strategy_type]
        ) as trace_ctx:
            try:
                # Execute RAG strategy
                trace_ctx.event("strategy_start", {
                    "strategy": self._strategy_type,
                    "query": task[:100]
                })

                result: RAGResult = await self._rag_strategy.execute(
                    query=task,
                    search_fn=self._search_fn
                )

                # Log results to Langfuse
                trace_ctx.event("strategy_complete", {
                    "strategy": self._strategy_type,
                    "retrieval_quality": result.retrieval_quality.value if result.retrieval_quality else None,
                    "total_searches": result.total_searches,
                    "total_llm_calls": result.total_llm_calls,
                    "execution_time_ms": result.execution_time_ms,
                    "reasoning_steps": [str(s) for s in result.reasoning_steps]
                })

                # Log retrieval
                if result.sources:
                    trace_ctx.log_retrieval(
                        query=task,
                        results=[{
                            "id": s.id,
                            "content": s.content[:200],
                            "score": s.score,
                            "metadata": s.metadata
                        } for s in result.sources],
                        source="notion",
                        top_k=len(result.sources)
                    )

                trace_ctx.log_output(result.answer)

                return result.answer

            except Exception as e:
                logger.error(f"RAG execution failed: {e}")
                trace_ctx.log_error(str(e))
                return self._error_response(str(e))

    async def execute_stream(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Execute knowledge query with streaming response.

        Args:
            task: The question or task to answer
            context: Optional context

        Yields:
            Chunks of the AI-generated answer
        """
        await self._ensure_initialized()

        try:
            async for chunk in self._rag_strategy.execute_stream(
                query=task,
                search_fn=self._search_fn
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Streaming execution failed: {e}")
            yield f"\n\nError: {e}"

    async def execute_with_details(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """
        Execute and return full RAGResult with reasoning steps.

        Useful for debugging and understanding the RAG process.

        Args:
            task: The question
            context: Optional context

        Returns:
            RAGResult with answer, sources, reasoning_steps, etc.
        """
        await self._ensure_initialized()

        result = await self._rag_strategy.execute(
            query=task,
            search_fn=self._search_fn
        )

        return result

    def set_strategy(self, strategy_type: str):
        """
        Change RAG strategy at runtime.

        Args:
            strategy_type: "single_shot", "corrective", or "self_rag"
        """
        self._strategy_type = strategy_type
        self._rag_strategy = None  # Will be re-initialized on next call
        logger.info(f"RAG strategy changed to: {strategy_type}")

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy information"""
        return {
            "strategy_type": self._strategy_type,
            "model": self._model,
            "use_enhanced_search": self._use_enhanced_search,
            "config": {
                "max_retries": self.config.get("max_retries", 2),
                "quality_threshold": self.config.get("quality_threshold", 0.7),
                "critique_threshold": self.config.get("critique_threshold", 0.7),
            }
        }

    def _error_response(self, error: str) -> str:
        """Response when an error occurs"""
        return f"""I encountered an error while processing your question.

Error: {error}

Please try again or contact support if the issue persists."""
