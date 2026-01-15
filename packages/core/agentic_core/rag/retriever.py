"""
RAG Retriever

벡터 검색 및 컨텍스트 생성 모듈
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import logging

from .stores.base import VectorStore, SearchResult
from .embedder import Embedder

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    """검색 모드"""
    SIMILARITY = "similarity"         # 단순 유사도 검색
    MMR = "mmr"                       # Maximal Marginal Relevance (다양성)
    THRESHOLD = "threshold"           # 최소 점수 기준


@dataclass
class RetrievalConfig:
    """검색 설정"""
    top_k: int = 5
    min_score: float = 0.0            # 최소 유사도 점수
    mode: RetrievalMode = RetrievalMode.SIMILARITY
    mmr_lambda: float = 0.5           # MMR 다양성 파라미터 (0=다양성, 1=관련성)
    rerank: bool = False              # 재순위 적용
    include_metadata: bool = True     # 메타데이터 포함


@dataclass
class RetrievedContext:
    """검색된 컨텍스트"""
    results: List[SearchResult]
    query: str
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self, separator: str = "\n\n---\n\n") -> str:
        """텍스트로 변환"""
        texts = []
        for i, result in enumerate(self.results):
            header = f"[Source {i + 1}]"
            if result.metadata.get("title"):
                header += f" {result.metadata['title']}"
            texts.append(f"{header}\n{result.content}")
        return separator.join(texts)

    def to_messages(self) -> List[Dict[str, str]]:
        """메시지 형태로 변환 (LLM 컨텍스트용)"""
        messages = []
        for result in self.results:
            messages.append({
                "role": "system",
                "content": f"Reference document:\n{result.content}"
            })
        return messages


class Retriever:
    """RAG 검색기"""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        config: Optional[RetrievalConfig] = None
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._reranker: Optional[Callable] = None

    def set_reranker(self, reranker: Callable[[str, List[SearchResult]], List[SearchResult]]) -> None:
        """재순위 함수 설정"""
        self._reranker = reranker

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> RetrievedContext:
        """
        쿼리에 대한 관련 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수 (None이면 config 값 사용)
            filters: 메타데이터 필터
            min_score: 최소 유사도 점수

        Returns:
            검색된 컨텍스트
        """
        effective_top_k = top_k or self.config.top_k
        effective_min_score = min_score if min_score is not None else self.config.min_score

        # 쿼리 임베딩
        query_result = await self.embedder.embed(query)
        query_embedding = query_result.embedding

        # 벡터 검색
        if self.config.mode == RetrievalMode.MMR:
            results = await self._mmr_search(
                query_embedding,
                effective_top_k,
                filters,
                effective_min_score
            )
        else:
            results = await self.store.search(
                query_embedding=query_embedding,
                top_k=effective_top_k * 2 if self.config.rerank else effective_top_k,
                filters=filters,
                min_score=effective_min_score
            )

        # 재순위
        if self.config.rerank and self._reranker:
            results = self._reranker(query, results)
            results = results[:effective_top_k]

        # 최소 점수 필터링
        if effective_min_score > 0:
            results = [r for r in results if r.score >= effective_min_score]

        # 토큰 수 추정
        total_tokens = sum(len(r.content.split()) * 1.3 for r in results)

        return RetrievedContext(
            results=results,
            query=query,
            total_tokens=int(total_tokens),
            metadata={
                "top_k": effective_top_k,
                "min_score": effective_min_score,
                "mode": self.config.mode.value,
                "result_count": len(results)
            }
        )

    async def _mmr_search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        min_score: float
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance 검색

        다양성과 관련성의 균형을 맞춘 검색
        """
        # 초기 후보 검색 (top_k의 3배)
        candidates = await self.store.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,
            filters=filters,
            min_score=min_score
        )

        if not candidates:
            return []

        selected = []
        remaining = candidates.copy()

        # MMR 선택
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = float("-inf")

            for i, candidate in enumerate(remaining):
                # 관련성 점수
                relevance = candidate.score

                # 다양성 점수 (선택된 것들과의 최대 유사도)
                if selected:
                    max_sim = max(
                        self._cosine_similarity(
                            candidate.content,
                            s.content
                        )
                        for s in selected
                    )
                else:
                    max_sim = 0.0

                # MMR 점수
                mmr_score = (
                    self.config.mmr_lambda * relevance -
                    (1 - self.config.mmr_lambda) * max_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """
        텍스트 간 간단한 유사도 계산 (단어 기반)

        실제 구현에서는 임베딩 기반 유사도 사용 권장
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def retrieve_with_context_window(
        self,
        query: str,
        max_tokens: int = 4000,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievedContext:
        """
        토큰 제한 내에서 최대한 많은 컨텍스트 검색

        Args:
            query: 검색 쿼리
            max_tokens: 최대 토큰 수
            filters: 메타데이터 필터

        Returns:
            검색된 컨텍스트
        """
        # 초기 검색 (많이 가져옴)
        context = await self.retrieve(
            query=query,
            top_k=20,
            filters=filters
        )

        # 토큰 제한에 맞게 조정
        results = []
        total_tokens = 0

        for result in context.results:
            result_tokens = int(len(result.content.split()) * 1.3)
            if total_tokens + result_tokens <= max_tokens:
                results.append(result)
                total_tokens += result_tokens
            else:
                break

        return RetrievedContext(
            results=results,
            query=query,
            total_tokens=total_tokens,
            metadata={
                **context.metadata,
                "max_tokens": max_tokens,
                "truncated": len(results) < len(context.results)
            }
        )

    async def retrieve_by_document(
        self,
        document_id: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        특정 문서의 청크들 검색

        Args:
            document_id: 문서 ID
            top_k: 반환할 청크 수

        Returns:
            검색 결과 리스트
        """
        effective_top_k = top_k or self.config.top_k

        results = await self.store.search(
            query_embedding=[0.0] * self.embedder.dimensions,  # 더미 임베딩
            top_k=effective_top_k * 10,
            filters={"document_id": document_id}
        )

        # document_id로 필터링된 결과를 chunk_index로 정렬
        sorted_results = sorted(
            results,
            key=lambda r: r.metadata.get("chunk_index", 0)
        )

        return sorted_results[:effective_top_k]


class MultiRetriever:
    """다중 소스 검색기"""

    def __init__(self, retrievers: Dict[str, Retriever]):
        """
        Args:
            retrievers: 소스별 검색기 딕셔너리 {"notion": retriever1, "slack": retriever2}
        """
        self.retrievers = retrievers

    async def retrieve(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        top_k: int = 5
    ) -> RetrievedContext:
        """
        다중 소스에서 검색

        Args:
            query: 검색 쿼리
            sources: 검색할 소스 (None이면 전체)
            top_k: 소스당 반환할 결과 수

        Returns:
            통합된 검색 결과
        """
        import asyncio

        target_sources = sources or list(self.retrievers.keys())
        tasks = []

        for source in target_sources:
            if source in self.retrievers:
                tasks.append(self._retrieve_from_source(
                    source,
                    self.retrievers[source],
                    query,
                    top_k
                ))

        all_results = await asyncio.gather(*tasks)

        # 결과 병합 및 점수순 정렬
        merged_results = []
        for source, results in all_results:
            for result in results:
                result.metadata["source"] = source
                merged_results.append(result)

        merged_results.sort(key=lambda r: r.score, reverse=True)

        return RetrievedContext(
            results=merged_results[:top_k * len(target_sources)],
            query=query,
            metadata={
                "sources": target_sources,
                "per_source_top_k": top_k
            }
        )

    async def _retrieve_from_source(
        self,
        source: str,
        retriever: Retriever,
        query: str,
        top_k: int
    ) -> tuple:
        """개별 소스에서 검색"""
        try:
            context = await retriever.retrieve(query, top_k=top_k)
            return (source, context.results)
        except Exception as e:
            logger.error(f"Failed to retrieve from {source}: {e}")
            return (source, [])
