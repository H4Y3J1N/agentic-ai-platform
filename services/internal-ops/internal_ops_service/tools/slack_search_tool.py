"""
Slack Search Tool - RAG-based semantic search over Slack messages
"""

from typing import List, Dict, Any, Optional
import os
import sys
import logging
from pathlib import Path

# Add core package to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))

from agentic_core.rag import EmbeddingModel, create_embedder
from agentic_core.tools import BaseTool, ToolResult, ToolConfig, ToolCapability, tool

logger = logging.getLogger(__name__)

# Default embedding model - Google Gemini
DEFAULT_EMBEDDING_MODEL = EmbeddingModel.GEMINI_EMBEDDING.value


@tool(
    name="slack_search",
    description="Search Slack messages using natural language",
    capabilities=[ToolCapability.SEARCH, ToolCapability.RETRIEVE],
    tags=["slack", "search", "rag"]
)
class SlackSearchTool(BaseTool):
    """RAG-based semantic search over Slack messages"""

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        super().__init__(config)
        # Support legacy dict config
        self._extra_config = kwargs.get("legacy_config", {})

        self._chroma_client = None
        self._collection = None
        self._embedder = None

    async def initialize(self):
        """Lazy initialization of ChromaDB and Gemini Embedder"""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required")

        persist_dir = self._extra_config.get(
            "persist_dir",
            os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
        )
        collection_name = self._extra_config.get("collection_name", "internal_ops_slack")

        self._chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize Gemini Embedder
        embedding_model = self._extra_config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self._embedder = create_embedder(model=embedding_model)

        self._initialized = True
        logger.info(f"SlackSearchTool initialized with collection '{collection_name}', embedder: {embedding_model}")

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute search (BaseTool interface).

        Args:
            query: Search query string
            top_k: Number of results (default: 5)
            filters: Optional filters dict

        Returns:
            ToolResult with search results
        """
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 5)
        filters = kwargs.get("filters")

        if not query:
            return ToolResult.fail("Query is required")

        try:
            results = await self.search(query=query, top_k=top_k, filters=filters)
            return ToolResult.ok(results, count=len(results))
        except Exception as e:
            return ToolResult.fail(str(e))

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for query text using Gemini Embedder"""
        # Use embed_query for retrieval queries (different task type than document embedding)
        if hasattr(self._embedder, 'embed_query'):
            result = await self._embedder.embed_query(text)
        else:
            result = await self._embedder.embed(text)

        return result.embedding

    async def _get_document_embedding(self, text: str) -> List[float]:
        """Get embedding for document text (uses retrieval_document task type)"""
        result = await self._embedder.embed(text)
        return result.embedding

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Slack messages semantically.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            filters: Optional filters (e.g., {"channel_name": "general"})

        Returns:
            List of matching messages with content and metadata
        """
        await self.initialize()

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Build where clause for filters
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

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 20),
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert distance to similarity score (1 - distance for cosine)
                relevance_score = 1.0 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                formatted_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "channel_name": metadata.get("channel_name", ""),
                    "user_name": metadata.get("user_name", "Unknown"),
                    "url": metadata.get("url", ""),
                    "relevance_score": relevance_score,
                    "timestamp": metadata.get("timestamp"),
                    "is_thread": metadata.get("is_thread", False),
                    "metadata": metadata
                })

        logger.info(f"Slack search for '{query}' returned {len(formatted_results)} results")
        return formatted_results

    async def search_by_channel(
        self,
        channel_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search messages in a specific channel.

        Args:
            channel_name: Channel name (without #)
            query: Search query
            top_k: Number of results

        Returns:
            Matching messages from the channel
        """
        return await self.search(
            query=query,
            top_k=top_k,
            filters={"channel_name": channel_name}
        )

    async def search_by_user(
        self,
        user_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search messages from a specific user.

        Args:
            user_name: User's display name
            query: Search query
            top_k: Number of results

        Returns:
            Matching messages from the user
        """
        return await self.search(
            query=query,
            top_k=top_k,
            filters={"user_name": user_name}
        )

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Slack message collection"""
        await self.initialize()

        return {
            "collection_name": self._collection.name,
            "document_count": self._collection.count(),
            "metadata": self._collection.metadata
        }

    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Add documents to the collection.

        Args:
            documents: List of documents with 'id', 'content', 'metadata'

        Returns:
            Number of documents added
        """
        await self.initialize()

        if not documents:
            return 0

        ids = []
        contents = []
        metadatas = []
        embeddings = []

        for doc in documents:
            doc_id = doc.get("id")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            if not doc_id or not content:
                continue

            # Get embedding (use document embedding for indexing)
            embedding = await self._get_document_embedding(content)

            ids.append(doc_id)
            contents.append(content)
            metadatas.append(metadata)
            embeddings.append(embedding)

        if ids:
            self._collection.upsert(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings
            )

        logger.info(f"Added {len(ids)} documents to Slack collection")
        return len(ids)
