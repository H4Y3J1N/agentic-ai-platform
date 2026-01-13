# packages/agentic-ai-core/src/agentic_ai_core/rag/chroma_store.py

"""
ChromaDB Vector Store - Milvus 대체 경량 벡터 저장소
단일 서버 환경에 적합, 파일 기반 영속성 지원
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os


@dataclass
class SearchResult:
    """검색 결과"""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class ChromaStore:
    """ChromaDB 벡터 저장소 - 경량 대안"""

    def __init__(self, config: dict):
        self.persist_dir = config.get("persist_dir", os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data"))
        self.collection_name = config.get("collection_name", "documents")

        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def insert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        """문서 삽입"""
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """벡터 검색"""
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
                search_results.append(SearchResult(
                    id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=results["distances"][0][i] if results["distances"] else 0.0
                ))

        return search_results

    async def delete(self, ids: List[str]):
        """문서 삭제"""
        self.collection.delete(ids=ids)

    async def update(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        """문서 업데이트"""
        self.collection.update(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def count(self) -> int:
        """문서 수 반환"""
        return self.collection.count()
