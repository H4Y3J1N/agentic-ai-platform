"""
Store Stage

벡터 저장소 및 그래프 저장소에 저장하는 스테이지
"""

from typing import Optional, Protocol, List, Dict, Any
import logging

from .base import Stage
from ..context import PipelineContext
from ...schema import Document, Entity, Relationship, Chunk
from ...rag import VectorStore

logger = logging.getLogger(__name__)


class GraphStore(Protocol):
    """그래프 저장소 프로토콜"""

    async def add_entity(self, entity: Entity) -> str:
        ...

    async def add_relationship(self, relationship: Relationship) -> str:
        ...


class DocumentStore(Protocol):
    """문서 저장소 프로토콜"""

    async def save(self, document: Document) -> str:
        ...


class StoreStage(Stage):
    """저장 스테이지"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        document_store: Optional[DocumentStore] = None,
        store_entities: bool = True,
        store_relationships: bool = True
    ):
        super().__init__("StoreStage")
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.document_store = document_store
        self.store_entities = store_entities
        self.store_relationships = store_relationships

    async def process(self, context: PipelineContext) -> PipelineContext:
        """저장 실행"""
        # 1. 벡터 저장
        if self.vector_store and context.chunks:
            await self._store_vectors(context)

        # 2. 그래프 저장
        if self.graph_store:
            if self.store_entities and context.entities:
                await self._store_entities(context)
            if self.store_relationships and context.relationships:
                await self._store_relationships(context)

        # 3. 문서 저장
        if self.document_store:
            await self._store_document(context)

        # 최종 문서 생성
        document = context.to_document()
        context.document = document

        logger.debug(
            f"Stored document: {len(context.chunks)} chunks, "
            f"{len(context.entities)} entities, "
            f"{len(context.relationships)} relationships"
        )

        return context

    async def _store_vectors(self, context: PipelineContext) -> None:
        """벡터 저장"""
        chunks = context.chunks

        # 임베딩이 있는 청크만 저장
        valid_chunks = [c for c in chunks if c.embedding]

        if not valid_chunks:
            logger.warning("No chunks with embeddings to store")
            return

        ids = [chunk.id for chunk in valid_chunks]
        texts = [chunk.content for chunk in valid_chunks]
        embeddings = [chunk.embedding for chunk in valid_chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "header_context": chunk.header_context,
                "section_name": chunk.section_name or "",
                **{k: v for k, v in chunk.metadata.items() if v is not None}
            }
            for chunk in valid_chunks
        ]

        await self.vector_store.insert(
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.debug(f"Stored {len(valid_chunks)} vectors")

    async def _store_entities(self, context: PipelineContext) -> None:
        """엔티티 저장"""
        for entity in context.entities:
            try:
                await self.graph_store.add_entity(entity)
            except Exception as e:
                logger.warning(f"Failed to store entity {entity.id}: {e}")

        logger.debug(f"Stored {len(context.entities)} entities")

    async def _store_relationships(self, context: PipelineContext) -> None:
        """관계 저장"""
        for relationship in context.relationships:
            try:
                await self.graph_store.add_relationship(relationship)
            except Exception as e:
                logger.warning(f"Failed to store relationship {relationship.id}: {e}")

        logger.debug(f"Stored {len(context.relationships)} relationships")

    async def _store_document(self, context: PipelineContext) -> None:
        """문서 메타데이터 저장"""
        document = context.to_document()
        try:
            await self.document_store.save(document)
            logger.debug(f"Stored document metadata: {document.id}")
        except Exception as e:
            logger.warning(f"Failed to store document {document.id}: {e}")


class InMemoryDocumentStore:
    """인메모리 문서 저장소 (테스트/개발용)"""

    def __init__(self):
        self.documents: Dict[str, Document] = {}

    async def save(self, document: Document) -> str:
        self.documents[document.id] = document
        return document.id

    async def get(self, document_id: str) -> Optional[Document]:
        return self.documents.get(document_id)

    async def delete(self, document_id: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False

    async def list_all(self) -> List[Document]:
        return list(self.documents.values())


class InMemoryGraphStore:
    """인메모리 그래프 저장소 (테스트/개발용)"""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}

    async def add_entity(self, entity: Entity) -> str:
        self.entities[entity.id] = entity
        return entity.id

    async def add_relationship(self, relationship: Relationship) -> str:
        self.relationships[relationship.id] = relationship
        return relationship.id

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        return self.relationships.get(rel_id)

    async def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1
    ) -> List[Entity]:
        """이웃 엔티티 조회"""
        neighbors = []
        for rel in self.relationships.values():
            if rel.source_entity_id == entity_id:
                if rel.target_entity_id in self.entities:
                    neighbors.append(self.entities[rel.target_entity_id])
            elif rel.target_entity_id == entity_id:
                if rel.source_entity_id in self.entities:
                    neighbors.append(self.entities[rel.source_entity_id])
        return neighbors
