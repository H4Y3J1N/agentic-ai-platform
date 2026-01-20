"""
Query Processor

쿼리 재작성, 확장, 분해 등 쿼리 전처리 모듈
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import asyncio
import re

logger = logging.getLogger(__name__)


@dataclass
class QueryProcessorConfig:
    """Query Processor 설정"""
    # Query Rewriting
    rewriting_enabled: bool = True
    max_rewrites: int = 3           # 생성할 쿼리 변형 수
    include_original: bool = True   # 원본 쿼리 포함 여부

    # LLM 설정
    llm_model: str = "gemini/gemini-1.5-flash"
    temperature: float = 0.3
    timeout: float = 30.0


@dataclass
class ProcessedQuery:
    """처리된 쿼리"""
    original: str
    rewritten: List[str] = field(default_factory=list)
    expanded: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)

    def get_all_queries(self) -> List[str]:
        """모든 쿼리 반환 (중복 제거)"""
        all_queries = []

        # 원본 쿼리 (include_original일 경우)
        if self.original:
            all_queries.append(self.original)

        # 재작성된 쿼리
        all_queries.extend(self.rewritten)

        # 확장된 쿼리
        all_queries.extend(self.expanded)

        # 서브 쿼리
        all_queries.extend(self.sub_queries)

        # 중복 제거 (순서 유지)
        seen = set()
        unique_queries = []
        for q in all_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        return unique_queries


class QueryProcessor(ABC):
    """Query Processor 베이스 클래스"""

    def __init__(self, config: Optional[QueryProcessorConfig] = None):
        self.config = config or QueryProcessorConfig()

    @abstractmethod
    async def process(self, query: str) -> ProcessedQuery:
        """쿼리 처리"""
        pass


class LLMQueryRewriter(QueryProcessor):
    """LLM 기반 Query Rewriter"""

    REWRITE_PROMPT = """You are a search query optimizer. Given a user's search query, generate {max_rewrites} alternative versions of the query that might help find relevant information.

Rules:
1. Each alternative should capture the same intent but use different wording
2. Include variations with synonyms, rephrased questions, or expanded context
3. Keep queries concise and search-friendly
4. Output ONLY the alternative queries, one per line
5. Do NOT include numbering or bullet points

User Query: {query}

Alternative Queries:"""

    def __init__(
        self,
        config: Optional[QueryProcessorConfig] = None,
        llm_client: Optional[Any] = None
    ):
        super().__init__(config)
        self._llm_client = llm_client

    async def _ensure_llm(self):
        """LLM 클라이언트 초기화"""
        if self._llm_client is None:
            try:
                from litellm import acompletion
                self._acompletion = acompletion
            except ImportError:
                raise ImportError(
                    "litellm is required for LLMQueryRewriter. "
                    "Install with: pip install litellm"
                )

    async def process(self, query: str) -> ProcessedQuery:
        """쿼리 재작성"""
        result = ProcessedQuery(original=query)

        if not self.config.rewriting_enabled:
            return result

        await self._ensure_llm()

        try:
            prompt = self.REWRITE_PROMPT.format(
                max_rewrites=self.config.max_rewrites,
                query=query
            )

            response = await self._acompletion(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )

            # 응답 파싱
            content = response.choices[0].message.content.strip()
            rewrites = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith(("-", "*", "•"))
            ]

            # 숫자 접두사 제거 (예: "1. query" -> "query")
            cleaned_rewrites = []
            for rw in rewrites:
                # 숫자+점+공백 패턴 제거
                cleaned = re.sub(r"^\d+\.\s*", "", rw)
                if cleaned:
                    cleaned_rewrites.append(cleaned)

            result.rewritten = cleaned_rewrites[:self.config.max_rewrites]

            logger.info(
                f"Query rewritten: '{query}' -> {len(result.rewritten)} variations"
            )

        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            # 실패해도 원본 쿼리는 반환

        return result


class MultiQueryDecomposer(QueryProcessor):
    """복합 쿼리 분해기

    복잡한 쿼리를 여러 개의 단순한 서브 쿼리로 분해
    """

    DECOMPOSE_PROMPT = """Break down the following complex question into simpler sub-questions that, when answered together, would address the original question.

Rules:
1. Create 2-4 focused sub-questions
2. Each sub-question should be answerable independently
3. Together they should cover all aspects of the original question
4. Output ONLY the sub-questions, one per line

Original Question: {query}

Sub-questions:"""

    def __init__(
        self,
        config: Optional[QueryProcessorConfig] = None,
        llm_client: Optional[Any] = None
    ):
        super().__init__(config)
        self._llm_client = llm_client

    async def _ensure_llm(self):
        """LLM 클라이언트 초기화"""
        if self._llm_client is None:
            try:
                from litellm import acompletion
                self._acompletion = acompletion
            except ImportError:
                raise ImportError(
                    "litellm is required for MultiQueryDecomposer. "
                    "Install with: pip install litellm"
                )

    async def process(self, query: str) -> ProcessedQuery:
        """쿼리 분해"""
        result = ProcessedQuery(original=query)

        await self._ensure_llm()

        try:
            prompt = self.DECOMPOSE_PROMPT.format(query=query)

            response = await self._acompletion(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )

            content = response.choices[0].message.content.strip()
            sub_queries = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith(("-", "*", "•"))
            ]

            # 숫자 접두사 제거
            cleaned_queries = []
            for sq in sub_queries:
                cleaned = re.sub(r"^\d+\.\s*", "", sq)
                if cleaned:
                    cleaned_queries.append(cleaned)

            result.sub_queries = cleaned_queries[:4]

            logger.info(
                f"Query decomposed: '{query}' -> {len(result.sub_queries)} sub-queries"
            )

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")

        return result


class CompositeQueryProcessor(QueryProcessor):
    """여러 Query Processor를 조합"""

    def __init__(
        self,
        processors: List[QueryProcessor],
        config: Optional[QueryProcessorConfig] = None
    ):
        super().__init__(config)
        self.processors = processors

    async def process(self, query: str) -> ProcessedQuery:
        """모든 프로세서 실행 후 결과 병합"""
        final_result = ProcessedQuery(original=query)

        for processor in self.processors:
            try:
                result = await processor.process(query)
                final_result.rewritten.extend(result.rewritten)
                final_result.expanded.extend(result.expanded)
                final_result.sub_queries.extend(result.sub_queries)
            except Exception as e:
                logger.warning(f"Processor {type(processor).__name__} failed: {e}")

        return final_result


def create_query_processor(
    rewriting_enabled: bool = True,
    llm_model: str = "gemini/gemini-1.5-flash",
    **kwargs
) -> QueryProcessor:
    """
    Query Processor 팩토리 함수

    Args:
        rewriting_enabled: 쿼리 재작성 활성화
        llm_model: 사용할 LLM 모델
        **kwargs: 추가 설정

    Returns:
        QueryProcessor 인스턴스
    """
    config = QueryProcessorConfig(
        rewriting_enabled=rewriting_enabled,
        llm_model=llm_model,
        **{k: v for k, v in kwargs.items() if k in QueryProcessorConfig.__dataclass_fields__}
    )

    return LLMQueryRewriter(config)
