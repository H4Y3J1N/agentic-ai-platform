"""
RAG Indexer

문서 인덱싱 관리 모듈
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncIterator
from datetime import datetime
from enum import Enum
import logging
import asyncio

from .stores.base import VectorStore, IndexStats
from .chunker import Chunker, TextChunk, ChunkerConfig, create_chunker, ChunkingStrategy
from .embedder import Embedder, EmbeddingResult

logger = logging.getLogger(__name__)


class IndexingStatus(str, Enum):
    """인덱싱 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class IndexingResult:
    """인덱싱 결과"""
    document_id: str
    status: IndexingStatus
    chunk_count: int = 0
    chunk_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_ms: int = 0


@dataclass
class IndexerConfig:
    """인덱서 설정"""
    batch_size: int = 50              # 배치 크기
    parallel_embeddings: int = 5      # 동시 임베딩 수
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = 500
    chunk_overlap: int = 50
    skip_existing: bool = True        # 기존 문서 스킵
    store_raw_text: bool = True       # 원본 텍스트 저장


class Indexer:
    """RAG 인덱서"""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunker: Optional[Chunker] = None,
        config: Optional[IndexerConfig] = None
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or IndexerConfig()

        # 청커 설정
        if chunker:
            self.chunker = chunker
        else:
            chunker_config = ChunkerConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            self.chunker = create_chunker(
                self.config.chunking_strategy,
                chunker_config
            )

    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IndexingResult:
        """
        단일 문서 인덱싱

        Args:
            document_id: 문서 ID
            content: 문서 콘텐츠
            metadata: 문서 메타데이터

        Returns:
            인덱싱 결과
        """
        start_time = datetime.now()

        try:
            # 기존 문서 확인
            if self.config.skip_existing:
                existing = await self.store.search(
                    query_embedding=[0.0] * self.embedder.dimensions,
                    top_k=1,
                    filters={"document_id": document_id}
                )
                if existing:
                    logger.info(f"Skipping existing document: {document_id}")
                    return IndexingResult(
                        document_id=document_id,
                        status=IndexingStatus.COMPLETED,
                        chunk_count=0,
                        duration_ms=0
                    )

            # 청킹
            chunks = self.chunker.chunk(content, metadata)

            if not chunks:
                logger.warning(f"No chunks generated for document: {document_id}")
                return IndexingResult(
                    document_id=document_id,
                    status=IndexingStatus.COMPLETED,
                    chunk_count=0,
                    duration_ms=self._calc_duration(start_time)
                )

            # 임베딩 생성
            texts = [chunk.content for chunk in chunks]
            embeddings = await self._generate_embeddings(texts)

            # 청크 ID 생성
            chunk_ids = [
                f"{document_id}_chunk_{i}"
                for i in range(len(chunks))
            ]

            # 메타데이터 구성
            metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "header_context": chunk.metadata.header_context,
                    "token_count": chunk.metadata.token_count,
                    **(metadata or {})
                }
                if chunk.metadata.section_name:
                    chunk_meta["section_name"] = chunk.metadata.section_name
                metadatas.append(chunk_meta)

            # 벡터 저장소에 삽입
            await self.store.insert(
                ids=chunk_ids,
                texts=texts if self.config.store_raw_text else [""] * len(texts),
                embeddings=[e.embedding for e in embeddings],
                metadatas=metadatas
            )

            duration = self._calc_duration(start_time)
            logger.info(
                f"Indexed document {document_id}: "
                f"{len(chunks)} chunks in {duration}ms"
            )

            return IndexingResult(
                document_id=document_id,
                status=IndexingStatus.COMPLETED,
                chunk_count=len(chunks),
                chunk_ids=chunk_ids,
                duration_ms=duration
            )

        except Exception as e:
            logger.error(f"Failed to index document {document_id}: {e}")
            return IndexingResult(
                document_id=document_id,
                status=IndexingStatus.FAILED,
                error_message=str(e),
                duration_ms=self._calc_duration(start_time)
            )

    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[IndexingResult]:
        """
        다중 문서 인덱싱

        Args:
            documents: 문서 리스트 [{"id": str, "content": str, "metadata": dict}, ...]

        Returns:
            인덱싱 결과 리스트
        """
        results = []

        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]

            # 배치 병렬 처리
            batch_tasks = [
                self.index_document(
                    document_id=doc["id"],
                    content=doc["content"],
                    metadata=doc.get("metadata")
                )
                for doc in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append(IndexingResult(
                        document_id=batch[j]["id"],
                        status=IndexingStatus.FAILED,
                        error_message=str(result)
                    ))
                else:
                    results.append(result)

            logger.info(f"Indexed batch {i // self.config.batch_size + 1}: {len(batch)} documents")

        return results

    async def index_stream(
        self,
        documents: AsyncIterator[Dict[str, Any]]
    ) -> AsyncIterator[IndexingResult]:
        """
        스트리밍 인덱싱

        Args:
            documents: 문서 스트림

        Yields:
            인덱싱 결과
        """
        async for doc in documents:
            result = await self.index_document(
                document_id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata")
            )
            yield result

    async def delete_document(self, document_id: str) -> bool:
        """
        문서 삭제

        Args:
            document_id: 문서 ID

        Returns:
            성공 여부
        """
        try:
            # 해당 문서의 모든 청크 조회
            results = await self.store.search(
                query_embedding=[0.0] * self.embedder.dimensions,
                top_k=1000,
                filters={"document_id": document_id}
            )

            if results:
                chunk_ids = [r.id for r in results]
                await self.store.delete(chunk_ids)
                logger.info(f"Deleted document {document_id}: {len(chunk_ids)} chunks")

            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def update_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IndexingResult:
        """
        문서 업데이트 (삭제 후 재인덱싱)

        Args:
            document_id: 문서 ID
            content: 새 콘텐츠
            metadata: 새 메타데이터

        Returns:
            인덱싱 결과
        """
        # 기존 삭제
        await self.delete_document(document_id)

        # 재인덱싱 (skip_existing 무시)
        original_skip = self.config.skip_existing
        self.config.skip_existing = False

        try:
            result = await self.index_document(document_id, content, metadata)
        finally:
            self.config.skip_existing = original_skip

        return result

    async def get_stats(self) -> IndexStats:
        """인덱스 통계 조회"""
        return await self.store.get_stats()

    async def _generate_embeddings(
        self,
        texts: List[str]
    ) -> List[EmbeddingResult]:
        """임베딩 생성 (병렬 처리)"""
        # 배치로 분할하여 병렬 처리
        all_results = []

        for i in range(0, len(texts), self.config.parallel_embeddings):
            batch = texts[i:i + self.config.parallel_embeddings]
            results = await self.embedder.embed_batch(batch)
            all_results.extend(results)

        return all_results

    def _calc_duration(self, start: datetime) -> int:
        """시간 차이 계산 (밀리초)"""
        return int((datetime.now() - start).total_seconds() * 1000)


class IncrementalIndexer(Indexer):
    """증분 인덱서"""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunker: Optional[Chunker] = None,
        config: Optional[IndexerConfig] = None
    ):
        super().__init__(store, embedder, chunker, config)
        self._indexed_hashes: Dict[str, str] = {}

    async def index_if_changed(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IndexingResult:
        """
        변경된 경우에만 인덱싱

        Args:
            document_id: 문서 ID
            content: 문서 콘텐츠
            metadata: 문서 메타데이터

        Returns:
            인덱싱 결과
        """
        import hashlib

        # 콘텐츠 해시 계산
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # 이전 해시와 비교
        if self._indexed_hashes.get(document_id) == content_hash:
            logger.debug(f"Document unchanged, skipping: {document_id}")
            return IndexingResult(
                document_id=document_id,
                status=IndexingStatus.COMPLETED,
                chunk_count=0
            )

        # 인덱싱
        result = await self.update_document(document_id, content, metadata)

        if result.status == IndexingStatus.COMPLETED:
            self._indexed_hashes[document_id] = content_hash

        return result

    def clear_cache(self) -> None:
        """해시 캐시 초기화"""
        self._indexed_hashes.clear()
