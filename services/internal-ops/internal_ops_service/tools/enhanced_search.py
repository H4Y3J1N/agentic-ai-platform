"""
Enhanced Search Tool

고급 RAG 기능 통합:
- Query Rewriting (LLM)
- Hybrid Search (BM25 + Semantic with RRF)
- Cross-Encoder Reranking
"""

from typing import List, Dict, Any, Optional, Callable, Awaitable
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field

# Add core package to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))

from agentic_core.rag import (
    # Embedder
    EmbeddingModel,
    create_embedder,
    # Reranker
    create_reranker,
    RerankerConfig,
    RerankResult,
    # Query Processor
    create_query_processor,
    QueryProcessorConfig,
    ProcessedQuery,
    # Hybrid Search
    create_hybrid_searcher,
    HybridSearchConfig,
    HybridSearchResult,
    BM25Index,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSearchConfig:
    """Enhanced Search 설정"""
    # Embedding
    embedding_model: str = EmbeddingModel.GEMINI_EMBEDDING.value

    # ChromaDB
    persist_dir: str = "./data/chroma"
    collection_name: str = "default"

    # Query Rewriting
    query_rewriting_enabled: bool = True
    query_rewriting_model: str = "gemini/gemini-1.5-flash"
    max_rewrites: int = 3
    include_original: bool = True

    # Hybrid Search
    hybrid_search_enabled: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    rrf_k: int = 60

    # Reranking
    reranking_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 5

    # General
    default_top_k: int = 5
    max_top_k: int = 20
    min_relevance_score: float = 0.3


@dataclass
class EnhancedSearchResult:
    """Enhanced Search 결과"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 상세 점수
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # 순위 정보
    original_rank: Optional[int] = None
    final_rank: Optional[int] = None

    # 쿼리 정보
    matched_query: Optional[str] = None


class EnhancedSearchTool:
    """
    Enhanced Search Tool

    고급 RAG 기능을 통합한 검색 도구
    """

    def __init__(self, config: Optional[EnhancedSearchConfig] = None):
        self.config = config or EnhancedSearchConfig()
        self.name = "EnhancedSearchTool"
        self.description = "Advanced search with query rewriting, hybrid search, and reranking"

        # Lazy-initialized components
        self._chroma_client = None
        self._collection = None
        self._embedder = None
        self._reranker = None
        self._query_processor = None
        self._hybrid_searcher = None
        self._bm25_built = False

    async def _ensure_initialized(self):
        """Lazy initialization of all components"""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required")

        # ChromaDB
        persist_dir = self.config.persist_dir
        if persist_dir.startswith("${"):
            persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")

        self._chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collection = self._chroma_client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Embedder
        self._embedder = create_embedder(model=self.config.embedding_model)

        # Query Processor
        if self.config.query_rewriting_enabled:
            self._query_processor = create_query_processor(
                rewriting_enabled=True,
                llm_model=self.config.query_rewriting_model,
                max_rewrites=self.config.max_rewrites,
                include_original=self.config.include_original
            )

        # Hybrid Searcher
        if self.config.hybrid_search_enabled:
            self._hybrid_searcher = create_hybrid_searcher(
                enabled=True,
                semantic_weight=self.config.semantic_weight,
                keyword_weight=self.config.keyword_weight,
                rrf_k=self.config.rrf_k
            )

        # Reranker
        if self.config.reranking_enabled:
            self._reranker = create_reranker(
                model=self.config.reranker_model,
                top_n=self.config.rerank_top_n
            )

        logger.info(
            f"EnhancedSearchTool initialized: "
            f"collection={self.config.collection_name}, "
            f"query_rewriting={self.config.query_rewriting_enabled}, "
            f"hybrid_search={self.config.hybrid_search_enabled}, "
            f"reranking={self.config.reranking_enabled}"
        )

    async def _build_bm25_index(self):
        """BM25 인덱스 구축 (최초 1회)"""
        if self._bm25_built or not self._hybrid_searcher:
            return

        # 컬렉션에서 모든 문서 가져오기
        all_docs = self._collection.get(include=["documents", "metadatas"])

        if all_docs and all_docs["ids"]:
            documents = []
            for i, doc_id in enumerate(all_docs["ids"]):
                documents.append({
                    "id": doc_id,
                    "content": all_docs["documents"][i] if all_docs["documents"] else ""
                })

            self._hybrid_searcher.build_bm25_index(documents)
            self._bm25_built = True
            logger.info(f"BM25 index built with {len(documents)} documents")

    async def _get_query_embedding(self, text: str) -> List[float]:
        """쿼리 임베딩 생성"""
        if hasattr(self._embedder, 'embed_query'):
            result = await self._embedder.embed_query(text)
        else:
            result = await self._embedder.embed(text)
        return result.embedding

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """시맨틱 검색"""
        query_embedding = await self._get_query_embedding(query)

        # Filter 구성
        where_clause = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    conditions.append({key: {"$eq": value}})

            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}

        # ChromaDB 쿼리
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.config.max_top_k),
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        # 결과 포맷팅
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance  # cosine distance to similarity

                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "score": score,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })

        return formatted

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        use_rewriting: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        use_reranking: Optional[bool] = None
    ) -> List[EnhancedSearchResult]:
        """
        Enhanced Search 실행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filters: 메타데이터 필터
            use_rewriting: Query rewriting 사용 여부 (None이면 config 따름)
            use_hybrid: Hybrid search 사용 여부
            use_reranking: Reranking 사용 여부

        Returns:
            검색 결과 목록
        """
        await self._ensure_initialized()

        # 옵션 결정
        do_rewriting = use_rewriting if use_rewriting is not None else self.config.query_rewriting_enabled
        do_hybrid = use_hybrid if use_hybrid is not None else self.config.hybrid_search_enabled
        do_reranking = use_reranking if use_reranking is not None else self.config.reranking_enabled

        # 1. Query Rewriting
        queries = [query]
        if do_rewriting and self._query_processor:
            try:
                processed = await self._query_processor.process(query)
                queries = processed.get_all_queries()
                logger.info(f"Query rewritten: {query} -> {len(queries)} variations")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}")

        # 2. 검색 실행 (각 쿼리에 대해)
        all_results: Dict[str, Dict[str, Any]] = {}

        for q in queries:
            if do_hybrid and self._hybrid_searcher:
                # Hybrid Search
                await self._build_bm25_index()

                async def semantic_fn(query: str, top_k: int):
                    return await self._semantic_search(query, top_k, filters)

                hybrid_results = await self._hybrid_searcher.search(
                    query=q,
                    semantic_search_fn=semantic_fn,
                    top_k=top_k * 2  # 더 많이 가져와서 융합
                )

                for hr in hybrid_results:
                    if hr.id not in all_results or hr.score > all_results[hr.id].get("score", 0):
                        all_results[hr.id] = {
                            "id": hr.id,
                            "content": hr.content,
                            "score": hr.score,
                            "semantic_score": hr.semantic_score,
                            "keyword_score": hr.keyword_score,
                            "metadata": hr.metadata,
                            "matched_query": q
                        }
            else:
                # Semantic only
                semantic_results = await self._semantic_search(q, top_k * 2, filters)

                for sr in semantic_results:
                    if sr["id"] not in all_results or sr["score"] > all_results[sr["id"]].get("score", 0):
                        all_results[sr["id"]] = {
                            **sr,
                            "semantic_score": sr["score"],
                            "keyword_score": None,
                            "matched_query": q
                        }

        # 점수 기준 정렬
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        # 3. Reranking
        if do_reranking and self._reranker and sorted_results:
            try:
                reranked = await self._reranker.rerank(
                    query=query,  # 원본 쿼리로 rerank
                    documents=sorted_results,
                    top_n=top_k
                )

                final_results = []
                for i, rr in enumerate(reranked):
                    final_results.append(EnhancedSearchResult(
                        id=rr.id,
                        content=rr.content,
                        score=rr.rerank_score,
                        metadata=rr.metadata,
                        semantic_score=sorted_results[rr.original_rank - 1].get("semantic_score") if rr.original_rank <= len(sorted_results) else None,
                        keyword_score=sorted_results[rr.original_rank - 1].get("keyword_score") if rr.original_rank <= len(sorted_results) else None,
                        rerank_score=rr.rerank_score,
                        original_rank=rr.original_rank,
                        final_rank=i + 1,
                        matched_query=sorted_results[rr.original_rank - 1].get("matched_query") if rr.original_rank <= len(sorted_results) else None
                    ))

                logger.info(f"Search completed: {len(final_results)} results (reranked)")
                return final_results

            except Exception as e:
                logger.warning(f"Reranking failed, returning without rerank: {e}")

        # Reranking 없이 반환
        final_results = []
        for i, r in enumerate(sorted_results[:top_k]):
            if r.get("score", 0) >= self.config.min_relevance_score:
                final_results.append(EnhancedSearchResult(
                    id=r["id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r.get("metadata", {}),
                    semantic_score=r.get("semantic_score"),
                    keyword_score=r.get("keyword_score"),
                    rerank_score=None,
                    original_rank=i + 1,
                    final_rank=i + 1,
                    matched_query=r.get("matched_query")
                ))

        logger.info(f"Search completed: {len(final_results)} results")
        return final_results

    async def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계"""
        await self._ensure_initialized()

        return {
            "collection_name": self._collection.name,
            "document_count": self._collection.count(),
            "metadata": self._collection.metadata,
            "config": {
                "query_rewriting": self.config.query_rewriting_enabled,
                "hybrid_search": self.config.hybrid_search_enabled,
                "reranking": self.config.reranking_enabled
            }
        }


def create_enhanced_search_tool(
    collection_name: str,
    config_dict: Optional[Dict[str, Any]] = None
) -> EnhancedSearchTool:
    """
    Enhanced Search Tool 팩토리 함수

    Args:
        collection_name: ChromaDB 컬렉션 이름
        config_dict: 설정 딕셔너리

    Returns:
        EnhancedSearchTool 인스턴스
    """
    config_dict = config_dict or {}
    config_dict["collection_name"] = collection_name

    config = EnhancedSearchConfig(**{
        k: v for k, v in config_dict.items()
        if k in EnhancedSearchConfig.__dataclass_fields__
    })

    return EnhancedSearchTool(config)
