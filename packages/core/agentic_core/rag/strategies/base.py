"""
RAG Strategy Base Classes

고급 RAG 파이프라인을 위한 추상 클래스 및 공통 타입 정의
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable, AsyncIterator
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RAGStrategyType(str, Enum):
    """RAG 전략 타입"""
    SINGLE_SHOT = "single_shot"      # 기본: 검색 → 답변
    CORRECTIVE = "corrective"         # CRAG: 검색 → 품질평가 → (재검색) → 답변
    SELF_RAG = "self_rag"            # Self-RAG: 검색 → 답변 → 자기검증 → (수정)
    AGENTIC = "agentic"              # Agentic: 질문분해 → 멀티검색 → 종합


class RetrievalQuality(str, Enum):
    """검색 결과 품질 등급"""
    EXCELLENT = "excellent"   # 충분한 정보, 높은 관련성
    GOOD = "good"            # 대체로 충분
    AMBIGUOUS = "ambiguous"  # 애매함, 추가 검색 고려
    POOR = "poor"            # 불충분, 재검색 필요


@dataclass
class Document:
    """검색된 문서"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 추가 점수 (선택)
    rerank_score: Optional[float] = None
    relevance_label: Optional[str] = None  # "relevant", "irrelevant", "ambiguous"


@dataclass
class ReasoningStep:
    """추론 단계 기록"""
    step_name: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.step_name}] {self.description}"


@dataclass
class RAGResult:
    """RAG 실행 결과"""
    # 핵심 결과
    answer: str
    sources: List[Document] = field(default_factory=list)

    # 추론 과정 (디버깅/설명용)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)

    # 품질 메트릭
    retrieval_quality: Optional[RetrievalQuality] = None
    confidence_score: Optional[float] = None

    # 메타데이터
    strategy_type: Optional[RAGStrategyType] = None
    total_searches: int = 1
    total_llm_calls: int = 1
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step_name: str, description: str, **kwargs):
        """추론 단계 추가"""
        self.reasoning_steps.append(ReasoningStep(
            step_name=step_name,
            description=description,
            metadata=kwargs
        ))

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "answer": self.answer,
            "sources": [
                {
                    "id": s.id,
                    "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                    "score": s.score,
                    "metadata": s.metadata
                }
                for s in self.sources
            ],
            "reasoning_steps": [str(s) for s in self.reasoning_steps],
            "retrieval_quality": self.retrieval_quality.value if self.retrieval_quality else None,
            "confidence_score": self.confidence_score,
            "strategy_type": self.strategy_type.value if self.strategy_type else None,
            "total_searches": self.total_searches,
            "total_llm_calls": self.total_llm_calls,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class RAGStrategyConfig:
    """RAG 전략 설정"""
    # LLM 설정
    llm_model: str = "gemini/gemini-1.5-flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # 검색 설정
    top_k: int = 5
    min_relevance_score: float = 0.3

    # Corrective RAG 설정
    quality_threshold: float = 0.7      # 품질 평가 임계값
    max_retries: int = 2                # 최대 재검색 횟수

    # Self-RAG 설정
    enable_self_critique: bool = True   # 자기 검증 활성화
    critique_threshold: float = 0.7     # 검증 통과 임계값

    # Agentic RAG 설정
    max_sub_queries: int = 3            # 최대 서브쿼리 수
    enable_query_decomposition: bool = True

    # 일반 설정
    timeout: float = 60.0
    verbose: bool = False


# Type aliases for callbacks
SearchFn = Callable[[str, int], Awaitable[List[Document]]]
LLMFn = Callable[[str], Awaitable[str]]
LLMStreamFn = Callable[[str], AsyncIterator[str]]


class RAGStrategy(ABC):
    """
    RAG Strategy 추상 베이스 클래스

    모든 RAG 전략은 이 클래스를 상속받아 구현합니다.
    """

    strategy_type: RAGStrategyType = None

    def __init__(self, config: Optional[RAGStrategyConfig] = None):
        self.config = config or RAGStrategyConfig()
        self._llm = None

    async def _ensure_llm(self):
        """LLM 클라이언트 초기화"""
        if self._llm is None:
            try:
                from litellm import acompletion
                self._llm = acompletion
            except ImportError:
                raise ImportError(
                    "litellm is required for RAG strategies. "
                    "Install with: pip install litellm"
                )

    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """LLM 호출 헬퍼"""
        await self._ensure_llm()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._llm(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            timeout=self.config.timeout
        )

        return response.choices[0].message.content.strip()

    @abstractmethod
    async def execute(
        self,
        query: str,
        search_fn: SearchFn,
        **kwargs
    ) -> RAGResult:
        """
        RAG 파이프라인 실행

        Args:
            query: 사용자 질문
            search_fn: 검색 함수 (query, top_k) -> List[Document]
            **kwargs: 추가 파라미터

        Returns:
            RAGResult: 실행 결과
        """
        pass

    async def execute_stream(
        self,
        query: str,
        search_fn: SearchFn,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        스트리밍 RAG 실행 (기본 구현은 non-streaming 후 청크로 반환)

        서브클래스에서 오버라이드하여 진짜 스트리밍 구현 가능
        """
        result = await self.execute(query, search_fn, **kwargs)

        # 청크 단위로 반환
        chunk_size = 50
        for i in range(0, len(result.answer), chunk_size):
            yield result.answer[i:i + chunk_size]

    def _format_docs_for_context(self, docs: List[Document]) -> str:
        """문서를 컨텍스트 문자열로 포맷"""
        if not docs:
            return "No relevant documents found."

        sections = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Untitled")
            score_info = f"score: {doc.score:.3f}"
            if doc.rerank_score is not None:
                score_info = f"rerank: {doc.rerank_score:.3f}, {score_info}"

            section = f"[{i}] {title} ({score_info})\n{doc.content}"
            sections.append(section)

        return "\n\n---\n\n".join(sections)

    def _log(self, message: str):
        """Verbose 로깅"""
        if self.config.verbose:
            logger.info(f"[{self.strategy_type.value}] {message}")
