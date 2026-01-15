"""
Vector Store Base

벡터 저장소 추상 인터페이스 정의
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SearchResult:
    """벡터 검색 결과"""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # 유사도 점수 (0~1, 높을수록 유사)
    distance: float = 0.0  # 거리 (낮을수록 유사)

    # 원본 참조
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None

    def __post_init__(self):
        # distance가 있으면 score로 변환 (cosine distance → similarity)
        if self.distance > 0 and self.score == 0:
            self.score = 1.0 - self.distance


@dataclass
class IndexStats:
    """인덱스 통계"""

    collection_name: str
    document_count: int
    embedding_dimension: Optional[int] = None
    index_size_bytes: Optional[int] = None
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """벡터 저장소 추상 인터페이스"""

    @abstractmethod
    async def insert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        벡터 삽입

        Args:
            ids: 문서 ID 리스트
            texts: 텍스트 콘텐츠 리스트
            embeddings: 임베딩 벡터 리스트
            metadatas: 메타데이터 리스트 (선택)
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        벡터 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 결과 수
            filters: 메타데이터 필터
            min_score: 최소 유사도 점수

        Returns:
            검색 결과 리스트 (유사도 순)
        """
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """
        벡터 삭제

        Args:
            ids: 삭제할 문서 ID 리스트
        """
        pass

    @abstractmethod
    async def update(
        self,
        ids: List[str],
        texts: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        벡터 업데이트

        Args:
            ids: 업데이트할 문서 ID 리스트
            texts: 새 텍스트 (선택)
            embeddings: 새 임베딩 (선택)
            metadatas: 새 메타데이터 (선택)
        """
        pass

    @abstractmethod
    async def get(self, ids: List[str]) -> List[SearchResult]:
        """
        ID로 문서 조회

        Args:
            ids: 조회할 문서 ID 리스트

        Returns:
            검색 결과 리스트
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """저장된 문서 수 반환"""
        pass

    @abstractmethod
    async def get_stats(self) -> IndexStats:
        """인덱스 통계 반환"""
        pass

    async def exists(self, id: str) -> bool:
        """문서 존재 여부 확인"""
        results = await self.get([id])
        return len(results) > 0

    async def upsert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        삽입 또는 업데이트 (upsert)

        Args:
            ids: 문서 ID 리스트
            texts: 텍스트 콘텐츠 리스트
            embeddings: 임베딩 벡터 리스트
            metadatas: 메타데이터 리스트 (선택)
        """
        # 기본 구현: 삭제 후 삽입
        await self.delete(ids)
        await self.insert(ids, texts, embeddings, metadatas)

    async def batch_insert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> None:
        """
        배치 삽입

        Args:
            ids: 문서 ID 리스트
            texts: 텍스트 콘텐츠 리스트
            embeddings: 임베딩 벡터 리스트
            metadatas: 메타데이터 리스트 (선택)
            batch_size: 배치 크기
        """
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size] if metadatas else None

            await self.insert(batch_ids, batch_texts, batch_embeddings, batch_metadatas)

    async def clear(self) -> None:
        """모든 문서 삭제 (구현 필요시 오버라이드)"""
        raise NotImplementedError("clear() not implemented for this store")
