"""
ChromaDB Vector Store

경량 벡터 저장소 구현 - 단일 서버 환경에 적합
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import logging

from .base import VectorStore, SearchResult, IndexStats

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """
    ChromaDB 벡터 저장소

    특징:
    - 파일 기반 영속성 (SQLite + Parquet)
    - HNSW 인덱스 (빠른 ANN 검색)
    - 메타데이터 필터링 지원
    - 단일 서버 환경에 최적화
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_dir: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        distance_metric: str = "cosine"
    ):
        """
        Args:
            collection_name: 컬렉션 이름
            persist_dir: 영속화 디렉토리 (None이면 환경변수 또는 기본값)
            embedding_dimension: 임베딩 차원 (자동 감지됨)
            distance_metric: 거리 메트릭 (cosine, l2, ip)
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir or os.environ.get(
            "CHROMA_PERSIST_DIR", "./chroma_data"
        )
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric

        self._client = None
        self._collection = None

    def _ensure_client(self):
        """ChromaDB 클라이언트 초기화 (lazy loading)"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                raise ImportError(
                    "chromadb is required. Install with: pip install chromadb"
                )

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )

            logger.info(
                f"ChromaDB initialized: collection={self.collection_name}, "
                f"persist_dir={self.persist_dir}"
            )

    @property
    def collection(self):
        """컬렉션 접근자"""
        self._ensure_client()
        return self._collection

    async def insert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """벡터 삽입"""
        if not ids:
            return

        # 메타데이터 정규화 (ChromaDB는 None 값 허용 안함)
        clean_metadatas = None
        if metadatas:
            clean_metadatas = [
                {k: v for k, v in m.items() if v is not None}
                for m in metadatas
            ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas
        )

        logger.debug(f"Inserted {len(ids)} documents")

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """벡터 검색"""
        # 필터 정규화
        where_filter = None
        if filters:
            where_filter = {k: v for k, v in filters.items() if v is not None}
            if not where_filter:
                where_filter = None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # cosine distance → similarity (1 - distance)
                score = 1.0 - distance if self.distance_metric == "cosine" else distance

                # 최소 점수 필터
                if min_score is not None and score < min_score:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                search_results.append(SearchResult(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=metadata,
                    score=score,
                    distance=distance,
                    document_id=metadata.get("document_id"),
                    chunk_index=metadata.get("chunk_index")
                ))

        return search_results

    async def delete(self, ids: List[str]) -> None:
        """벡터 삭제"""
        if not ids:
            return
        self.collection.delete(ids=ids)
        logger.debug(f"Deleted {len(ids)} documents")

    async def update(
        self,
        ids: List[str],
        texts: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """벡터 업데이트"""
        if not ids:
            return

        update_kwargs = {"ids": ids}
        if texts:
            update_kwargs["documents"] = texts
        if embeddings:
            update_kwargs["embeddings"] = embeddings
        if metadatas:
            update_kwargs["metadatas"] = [
                {k: v for k, v in m.items() if v is not None}
                for m in metadatas
            ]

        self.collection.update(**update_kwargs)
        logger.debug(f"Updated {len(ids)} documents")

    async def get(self, ids: List[str]) -> List[SearchResult]:
        """ID로 문서 조회"""
        if not ids:
            return []

        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"]
        )

        search_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                search_results.append(SearchResult(
                    id=doc_id,
                    content=results["documents"][i] if results["documents"] else "",
                    metadata=metadata,
                    score=1.0,  # 정확한 ID 매칭
                    document_id=metadata.get("document_id"),
                    chunk_index=metadata.get("chunk_index")
                ))

        return search_results

    async def count(self) -> int:
        """저장된 문서 수 반환"""
        return self.collection.count()

    async def get_stats(self) -> IndexStats:
        """인덱스 통계 반환"""
        count = await self.count()

        return IndexStats(
            collection_name=self.collection_name,
            document_count=count,
            embedding_dimension=self.embedding_dimension,
            last_updated=datetime.now(),
            metadata={
                "persist_dir": self.persist_dir,
                "distance_metric": self.distance_metric
            }
        )

    async def clear(self) -> None:
        """모든 문서 삭제"""
        self._ensure_client()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        logger.info(f"Cleared collection: {self.collection_name}")

    async def upsert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """삽입 또는 업데이트"""
        if not ids:
            return

        clean_metadatas = None
        if metadatas:
            clean_metadatas = [
                {k: v for k, v in m.items() if v is not None}
                for m in metadatas
            ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas
        )
        logger.debug(f"Upserted {len(ids)} documents")

    def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        """필터로 문서 삭제 (동기)"""
        where_filter = {k: v for k, v in filters.items() if v is not None}
        if where_filter:
            self.collection.delete(where=where_filter)
            logger.debug(f"Deleted documents matching filter: {where_filter}")
