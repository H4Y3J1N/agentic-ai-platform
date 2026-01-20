"""
Cross-Encoder Reranker

문서 재순위화를 위한 Cross-Encoder 기반 Reranker
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class RerankerModel(str, Enum):
    """지원되는 Reranker 모델"""
    # MS MARCO Cross-Encoders (sentence-transformers)
    MS_MARCO_MINILM_L6 = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MS_MARCO_MINILM_L12 = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    MS_MARCO_TINYBERT = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    # Multilingual
    MMARCO_MINILM_L12 = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # BGE Reranker
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    BGE_RERANKER_LARGE = "BAAI/bge-reranker-large"


@dataclass
class RerankerConfig:
    """Reranker 설정"""
    model: str = RerankerModel.MS_MARCO_MINILM_L6.value
    top_n: int = 5              # 재순위화 후 반환할 결과 수
    batch_size: int = 32        # 배치 처리 크기
    max_length: int = 512       # 최대 입력 길이
    device: Optional[str] = None  # cuda, cpu, or None (auto)


@dataclass
class RerankResult:
    """재순위화 결과"""
    id: str
    content: str
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class Reranker(ABC):
    """Reranker 베이스 클래스"""

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """
        문서 재순위화

        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 목록 (id, content, score, metadata 포함)
            top_n: 반환할 결과 수 (None이면 config 값 사용)

        Returns:
            재순위화된 결과 목록
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """모델 이름"""
        pass


class CrossEncoderReranker(Reranker):
    """Cross-Encoder 기반 Reranker (sentence-transformers)"""

    def __init__(self, config: Optional[RerankerConfig] = None):
        super().__init__(config)
        self._model = None

    def _ensure_model(self):
        """모델 로드 (lazy loading)"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                )

            device = self.config.device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = CrossEncoder(
                self.config.model,
                max_length=self.config.max_length,
                device=device
            )
            logger.info(f"CrossEncoder loaded: {self.config.model} on {device}")

    @property
    def model_name(self) -> str:
        return self.config.model

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """Cross-Encoder로 문서 재순위화"""
        if not documents:
            return []

        self._ensure_model()
        top_n = top_n or self.config.top_n

        # 쿼리-문서 쌍 생성
        pairs = []
        for doc in documents:
            content = doc.get("content", "")
            if content:
                pairs.append((query, content))

        if not pairs:
            return []

        # Cross-Encoder 스코어링 (동기 함수이므로 executor에서 실행)
        loop = asyncio.get_event_loop()

        def _compute_scores():
            scores = []
            # 배치 처리
            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i:i + self.config.batch_size]
                batch_scores = self._model.predict(batch)
                scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else batch_scores)
            return scores

        rerank_scores = await loop.run_in_executor(None, _compute_scores)

        # 결과 생성 및 정렬
        results = []
        for i, (doc, rerank_score) in enumerate(zip(documents, rerank_scores)):
            results.append(RerankResult(
                id=doc.get("id", f"doc_{i}"),
                content=doc.get("content", ""),
                original_score=doc.get("score", doc.get("relevance_score", 0.0)),
                rerank_score=float(rerank_score),
                original_rank=i + 1,
                new_rank=0,  # 정렬 후 설정
                metadata=doc.get("metadata", {})
            ))

        # rerank_score 기준 정렬
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # 새 순위 설정 및 top_n 적용
        for i, result in enumerate(results[:top_n]):
            result.new_rank = i + 1

        logger.info(
            f"Reranked {len(documents)} documents -> top {top_n} "
            f"(model: {self.config.model})"
        )

        return results[:top_n]


class BGEReranker(Reranker):
    """BGE Reranker (FlagEmbedding 기반)"""

    def __init__(self, config: Optional[RerankerConfig] = None):
        super().__init__(config)
        self._model = None

    def _ensure_model(self):
        """모델 로드 (lazy loading)"""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError:
                raise ImportError(
                    "FlagEmbedding is required for BGEReranker. "
                    "Install with: pip install FlagEmbedding"
                )

            self._model = FlagReranker(
                self.config.model,
                use_fp16=True  # 성능 최적화
            )
            logger.info(f"BGE Reranker loaded: {self.config.model}")

    @property
    def model_name(self) -> str:
        return self.config.model

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """BGE Reranker로 문서 재순위화"""
        if not documents:
            return []

        self._ensure_model()
        top_n = top_n or self.config.top_n

        # 쿼리-문서 쌍 생성
        pairs = []
        for doc in documents:
            content = doc.get("content", "")
            if content:
                pairs.append([query, content])

        if not pairs:
            return []

        # BGE 스코어링
        loop = asyncio.get_event_loop()

        def _compute_scores():
            return self._model.compute_score(pairs)

        rerank_scores = await loop.run_in_executor(None, _compute_scores)

        # 결과 생성 및 정렬
        results = []
        for i, (doc, rerank_score) in enumerate(zip(documents, rerank_scores)):
            results.append(RerankResult(
                id=doc.get("id", f"doc_{i}"),
                content=doc.get("content", ""),
                original_score=doc.get("score", doc.get("relevance_score", 0.0)),
                rerank_score=float(rerank_score),
                original_rank=i + 1,
                new_rank=0,
                metadata=doc.get("metadata", {})
            ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)

        for i, result in enumerate(results[:top_n]):
            result.new_rank = i + 1

        logger.info(
            f"Reranked {len(documents)} documents -> top {top_n} "
            f"(model: {self.config.model})"
        )

        return results[:top_n]


def create_reranker(
    model: str = RerankerModel.MS_MARCO_MINILM_L6.value,
    **kwargs
) -> Reranker:
    """
    Reranker 팩토리 함수

    Args:
        model: Reranker 모델 이름
        **kwargs: 추가 설정

    Returns:
        Reranker 인스턴스
    """
    config = RerankerConfig(model=model, **{
        k: v for k, v in kwargs.items()
        if k in RerankerConfig.__dataclass_fields__
    })

    # BGE Reranker
    if "bge-reranker" in model.lower():
        return BGEReranker(config)

    # Default: Cross-Encoder
    return CrossEncoderReranker(config)
