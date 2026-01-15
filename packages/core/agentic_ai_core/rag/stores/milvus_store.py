"""
Milvus Vector Store

Milvus Lite를 사용한 벡터 저장소 구현
- 로컬 파일 기반 (Milvus Lite)
- 서버 연결 (Milvus Standalone/Cluster)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from functools import partial

from .base import VectorStore, SearchResult, IndexStats


class MilvusStore(VectorStore):
    """
    Milvus 벡터 저장소

    Milvus Lite (로컬) 또는 Milvus 서버와 연동합니다.
    """

    def __init__(
        self,
        collection_name: str,
        uri: str = "./milvus_lite.db",
        dimension: int = 1536,
        metric_type: str = "COSINE",
        index_type: str = "AUTOINDEX",
        **kwargs,
    ):
        """
        MilvusStore 초기화

        Args:
            collection_name: 컬렉션 이름
            uri: Milvus 연결 URI
                - 로컬 파일: "./milvus.db" (Milvus Lite)
                - 서버: "http://localhost:19530"
            dimension: 벡터 차원
            metric_type: 유사도 메트릭 (COSINE, L2, IP)
            index_type: 인덱스 타입 (AUTOINDEX, IVF_FLAT, HNSW 등)
            **kwargs: 추가 연결 옵션
        """
        self.collection_name = collection_name
        self.uri = uri
        self.dimension = dimension
        self.metric_type = metric_type
        self.index_type = index_type
        self._kwargs = kwargs

        self._client = None
        self._collection_initialized = False

    async def _get_client(self):
        """Milvus 클라이언트 lazy initialization"""
        if self._client is None:
            from pymilvus import MilvusClient

            # 동기 클라이언트를 비동기로 래핑
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None,
                partial(MilvusClient, uri=self.uri, **self._kwargs)
            )

            await self._ensure_collection()

        return self._client

    async def _ensure_collection(self):
        """컬렉션 존재 확인 및 생성"""
        if self._collection_initialized:
            return

        loop = asyncio.get_event_loop()
        client = self._client

        # 컬렉션 존재 확인
        has_collection = await loop.run_in_executor(
            None,
            lambda: client.has_collection(self.collection_name)
        )

        if not has_collection:
            # 컬렉션 생성
            await loop.run_in_executor(
                None,
                lambda: client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.dimension,
                    metric_type=self.metric_type,
                    auto_id=False,
                )
            )

        self._collection_initialized = True

    async def insert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """벡터 삽입"""
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        # 데이터 준비
        data = []
        for i, (id_, text, embedding) in enumerate(zip(ids, texts, embeddings)):
            record = {
                "id": id_,
                "vector": embedding,
                "text": text,
            }
            # 메타데이터 추가
            if metadatas and i < len(metadatas):
                for key, value in metadatas[i].items():
                    # Milvus는 동적 필드 지원
                    if isinstance(value, (str, int, float, bool)):
                        record[key] = value

            data.append(record)

        await loop.run_in_executor(
            None,
            lambda: client.insert(
                collection_name=self.collection_name,
                data=data
            )
        )

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """벡터 검색"""
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        # 필터 표현식 생성
        filter_expr = self._build_filter_expr(filters) if filters else ""

        results = await loop.run_in_executor(
            None,
            lambda: client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr if filter_expr else None,
                output_fields=["*"],
            )
        )

        search_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                # 거리를 유사도로 변환
                distance = hit.get("distance", 0)
                if self.metric_type == "COSINE":
                    score = 1 - distance  # cosine distance to similarity
                elif self.metric_type == "IP":
                    score = distance  # inner product is already similarity
                else:  # L2
                    score = 1 / (1 + distance)

                if min_score and score < min_score:
                    continue

                entity = hit.get("entity", {})
                search_results.append(SearchResult(
                    id=str(hit.get("id", "")),
                    content=entity.get("text", ""),
                    metadata={k: v for k, v in entity.items() if k not in ["id", "vector", "text"]},
                    score=score,
                    distance=distance,
                ))

        return search_results

    async def delete(self, ids: List[str]) -> None:
        """벡터 삭제"""
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        # ID로 삭제
        filter_expr = f'id in {ids}'
        await loop.run_in_executor(
            None,
            lambda: client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
        )

    async def update(
        self,
        ids: List[str],
        texts: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """벡터 업데이트 (upsert 사용)"""
        # Milvus는 upsert를 지원
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        if not texts or not embeddings:
            # 기존 데이터 조회
            existing = await self.get(ids)
            if not texts:
                texts = [r.content for r in existing]
            if not embeddings:
                # 임베딩은 조회 불가, 에러
                raise ValueError("embeddings required for update")

        data = []
        for i, (id_, text, embedding) in enumerate(zip(ids, texts, embeddings)):
            record = {
                "id": id_,
                "vector": embedding,
                "text": text,
            }
            if metadatas and i < len(metadatas):
                for key, value in metadatas[i].items():
                    if isinstance(value, (str, int, float, bool)):
                        record[key] = value
            data.append(record)

        await loop.run_in_executor(
            None,
            lambda: client.upsert(
                collection_name=self.collection_name,
                data=data
            )
        )

    async def get(self, ids: List[str]) -> List[SearchResult]:
        """ID로 문서 조회"""
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        results = await loop.run_in_executor(
            None,
            lambda: client.get(
                collection_name=self.collection_name,
                ids=ids,
                output_fields=["*"]
            )
        )

        return [
            SearchResult(
                id=str(r.get("id", "")),
                content=r.get("text", ""),
                metadata={k: v for k, v in r.items() if k not in ["id", "vector", "text"]},
            )
            for r in results
        ]

    async def count(self) -> int:
        """저장된 문서 수 반환"""
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        stats = await loop.run_in_executor(
            None,
            lambda: client.get_collection_stats(self.collection_name)
        )

        return stats.get("row_count", 0)

    async def get_stats(self) -> IndexStats:
        """인덱스 통계 반환"""
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        stats = await loop.run_in_executor(
            None,
            lambda: client.get_collection_stats(self.collection_name)
        )

        return IndexStats(
            collection_name=self.collection_name,
            document_count=stats.get("row_count", 0),
            embedding_dimension=self.dimension,
            metadata={
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "uri": self.uri,
            }
        )

    async def clear(self) -> None:
        """모든 문서 삭제"""
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        # 컬렉션 삭제 후 재생성
        await loop.run_in_executor(
            None,
            lambda: client.drop_collection(self.collection_name)
        )

        self._collection_initialized = False
        await self._ensure_collection()

    def _build_filter_expr(self, filters: Dict[str, Any]) -> str:
        """Milvus 필터 표현식 생성"""
        conditions = []

        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            elif isinstance(value, bool):
                conditions.append(f'{key} == {str(value).lower()}')
            elif isinstance(value, (int, float)):
                conditions.append(f'{key} == {value}')
            elif isinstance(value, list):
                # in 연산
                if all(isinstance(v, str) for v in value):
                    values_str = ", ".join(f'"{v}"' for v in value)
                    conditions.append(f'{key} in [{values_str}]')
                else:
                    values_str = ", ".join(str(v) for v in value)
                    conditions.append(f'{key} in [{values_str}]')
            elif isinstance(value, dict):
                # 범위 쿼리
                if "$gt" in value:
                    conditions.append(f'{key} > {value["$gt"]}')
                if "$gte" in value:
                    conditions.append(f'{key} >= {value["$gte"]}')
                if "$lt" in value:
                    conditions.append(f'{key} < {value["$lt"]}')
                if "$lte" in value:
                    conditions.append(f'{key} <= {value["$lte"]}')

        return " and ".join(conditions)

    async def close(self) -> None:
        """리소스 정리"""
        if self._client:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._client.close)
            self._client = None

    async def __aenter__(self):
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
