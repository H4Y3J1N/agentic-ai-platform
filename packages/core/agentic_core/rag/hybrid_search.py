"""
Hybrid Search

BM25 키워드 검색 + 시맨틱 검색을 RRF (Reciprocal Rank Fusion)로 결합
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable
from collections import defaultdict
import logging
import asyncio
import math
import re

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Hybrid Search 설정"""
    enabled: bool = True
    semantic_weight: float = 0.7    # 시맨틱 검색 가중치
    keyword_weight: float = 0.3     # 키워드 검색 가중치
    rrf_k: int = 60                 # RRF 상수 (기본 60)
    min_score: float = 0.0          # 최소 점수


@dataclass
class HybridSearchResult:
    """Hybrid Search 결과"""
    id: str
    content: str
    score: float                    # RRF 융합 점수
    semantic_score: float           # 시맨틱 검색 점수
    keyword_score: float            # 키워드 검색 점수
    semantic_rank: Optional[int]    # 시맨틱 검색 순위
    keyword_rank: Optional[int]     # 키워드 검색 순위
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """BM25 키워드 검색 인덱스"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        BM25 파라미터 초기화

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Document length normalization (0-1, 0.75 typical)
            epsilon: Minimum IDF floor
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> doc count
        self.idf: Dict[str, float] = {}       # term -> IDF score
        self.doc_len: Dict[str, int] = {}     # doc_id -> length
        self.term_freqs: Dict[str, Dict[str, int]] = {}  # doc_id -> {term: freq}
        self.documents: Dict[str, str] = {}   # doc_id -> content

    def _tokenize(self, text: str) -> List[str]:
        """간단한 토크나이저"""
        # 소문자 변환 및 비알파벳 문자 제거
        text = text.lower()
        # 단어 분리
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ):
        """
        문서 인덱싱

        Args:
            documents: [{"id": "...", "content": "..."}, ...]
        """
        # 토큰화 및 통계 수집
        for doc in documents:
            doc_id = doc.get("id", "")
            content = doc.get("content", "")

            if not doc_id or not content:
                continue

            tokens = self._tokenize(content)
            self.doc_len[doc_id] = len(tokens)
            self.documents[doc_id] = content

            # Term frequency
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            self.term_freqs[doc_id] = dict(term_freq)

            # Document frequency (한 문서에서 term이 나타나면 +1)
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # 통계 계산
        self.corpus_size = len(self.doc_len)
        if self.corpus_size > 0:
            self.avgdl = sum(self.doc_len.values()) / self.corpus_size

        # IDF 계산
        self._calculate_idf()

        logger.info(f"BM25 index built: {self.corpus_size} documents")

    def _calculate_idf(self):
        """IDF (Inverse Document Frequency) 계산"""
        for term, freq in self.doc_freqs.items():
            # BM25 IDF formula
            idf = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1
            )
            self.idf[term] = max(idf, self.epsilon)

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        BM25 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수

        Returns:
            점수와 함께 정렬된 결과 목록
        """
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = {}

        for doc_id, term_freqs in self.term_freqs.items():
            score = 0.0
            doc_len = self.doc_len.get(doc_id, 0)

            for token in query_tokens:
                if token not in term_freqs:
                    continue

                tf = term_freqs[token]
                idf = self.idf.get(token, 0)

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scores[doc_id] = score

        # 정렬 및 상위 k개 반환
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            results.append({
                "id": doc_id,
                "content": self.documents.get(doc_id, ""),
                "score": score
            })

        return results


class HybridSearcher:
    """
    Hybrid Search 실행기

    BM25 + Semantic 검색을 RRF로 융합
    """

    def __init__(self, config: Optional[HybridSearchConfig] = None):
        self.config = config or HybridSearchConfig()
        self.bm25_index: Optional[BM25Index] = None

    def build_bm25_index(self, documents: List[Dict[str, Any]]):
        """BM25 인덱스 구축"""
        self.bm25_index = BM25Index()
        self.bm25_index.add_documents(documents)

    def _rrf_score(self, rank: int) -> float:
        """
        RRF (Reciprocal Rank Fusion) 점수 계산

        RRF(d) = Σ 1 / (k + rank(d))

        Args:
            rank: 문서 순위 (1부터 시작)

        Returns:
            RRF 점수
        """
        return 1.0 / (self.config.rrf_k + rank)

    async def search(
        self,
        query: str,
        semantic_search_fn: Callable[[str, int], Awaitable[List[Dict[str, Any]]]],
        top_k: int = 10,
        semantic_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None
    ) -> List[HybridSearchResult]:
        """
        Hybrid 검색 실행

        Args:
            query: 검색 쿼리
            semantic_search_fn: 시맨틱 검색 함수 (async def fn(query, top_k) -> results)
            top_k: 최종 반환할 결과 수
            semantic_top_k: 시맨틱 검색에서 가져올 결과 수
            keyword_top_k: 키워드 검색에서 가져올 결과 수

        Returns:
            RRF로 융합된 결과 목록
        """
        # 검색 범위 설정 (더 많이 가져와서 융합)
        fetch_k = max(top_k * 3, 30)
        semantic_top_k = semantic_top_k or fetch_k
        keyword_top_k = keyword_top_k or fetch_k

        # 1. 시맨틱 검색
        semantic_results = await semantic_search_fn(query, semantic_top_k)

        # 2. 키워드 검색 (BM25)
        keyword_results = []
        if self.bm25_index and self.config.keyword_weight > 0:
            keyword_results = self.bm25_index.search(query, keyword_top_k)

        # Hybrid가 비활성화되어 있거나 키워드 결과가 없으면 시맨틱만 반환
        if not self.config.enabled or not keyword_results:
            return [
                HybridSearchResult(
                    id=r.get("id", ""),
                    content=r.get("content", ""),
                    score=r.get("score", r.get("relevance_score", 0.0)),
                    semantic_score=r.get("score", r.get("relevance_score", 0.0)),
                    keyword_score=0.0,
                    semantic_rank=i + 1,
                    keyword_rank=None,
                    metadata=r.get("metadata", {})
                )
                for i, r in enumerate(semantic_results[:top_k])
            ]

        # 3. RRF 융합
        doc_scores = defaultdict(lambda: {
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "semantic_rank": None,
            "keyword_rank": None,
            "content": "",
            "metadata": {}
        })

        # 시맨틱 검색 결과 처리
        for rank, result in enumerate(semantic_results, 1):
            doc_id = result.get("id", "")
            score = result.get("score", result.get("relevance_score", 0.0))

            doc_scores[doc_id]["semantic_score"] = score
            doc_scores[doc_id]["semantic_rank"] = rank
            doc_scores[doc_id]["content"] = result.get("content", "")
            doc_scores[doc_id]["metadata"] = result.get("metadata", {})

        # 키워드 검색 결과 처리
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result.get("id", "")
            score = result.get("score", 0.0)

            doc_scores[doc_id]["keyword_score"] = score
            doc_scores[doc_id]["keyword_rank"] = rank
            if not doc_scores[doc_id]["content"]:
                doc_scores[doc_id]["content"] = result.get("content", "")

        # 4. RRF 점수 계산
        fused_results = []
        for doc_id, scores in doc_scores.items():
            rrf_score = 0.0

            # 시맨틱 RRF 기여
            if scores["semantic_rank"]:
                rrf_score += self.config.semantic_weight * self._rrf_score(
                    scores["semantic_rank"]
                )

            # 키워드 RRF 기여
            if scores["keyword_rank"]:
                rrf_score += self.config.keyword_weight * self._rrf_score(
                    scores["keyword_rank"]
                )

            if rrf_score >= self.config.min_score:
                fused_results.append(HybridSearchResult(
                    id=doc_id,
                    content=scores["content"],
                    score=rrf_score,
                    semantic_score=scores["semantic_score"],
                    keyword_score=scores["keyword_score"],
                    semantic_rank=scores["semantic_rank"],
                    keyword_rank=scores["keyword_rank"],
                    metadata=scores["metadata"]
                ))

        # 5. RRF 점수로 정렬
        fused_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"Hybrid search: {len(semantic_results)} semantic + "
            f"{len(keyword_results)} keyword -> {len(fused_results[:top_k])} fused"
        )

        return fused_results[:top_k]


def create_hybrid_searcher(
    enabled: bool = True,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    rrf_k: int = 60,
    **kwargs
) -> HybridSearcher:
    """
    Hybrid Searcher 팩토리 함수

    Args:
        enabled: Hybrid search 활성화
        semantic_weight: 시맨틱 검색 가중치
        keyword_weight: 키워드 검색 가중치
        rrf_k: RRF 상수
        **kwargs: 추가 설정

    Returns:
        HybridSearcher 인스턴스
    """
    config = HybridSearchConfig(
        enabled=enabled,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        rrf_k=rrf_k,
        **{k: v for k, v in kwargs.items() if k in HybridSearchConfig.__dataclass_fields__}
    )

    return HybridSearcher(config)
