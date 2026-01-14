"""
RAG (Retrieval-Augmented Generation) Package

벡터 검색 및 RAG 파이프라인 모듈
"""

# Stores
from .stores import (
    VectorStore,
    SearchResult,
    IndexStats,
    ChromaStore,
)

# Chunker
from .chunker import (
    Chunker,
    ChunkerConfig,
    ChunkingStrategy,
    TextChunk,
    ChunkMetadata,
    FixedSizeChunker,
    SemanticChunker,
    RecursiveChunker,
    CodeAwareChunker,
    create_chunker,
)

# Embedder
from .embedder import (
    Embedder,
    EmbedderConfig,
    EmbeddingModel,
    EmbeddingResult,
    OpenAIEmbedder,
    OllamaEmbedder,
    SentenceTransformersEmbedder,
    create_embedder,
)

# Retriever
from .retriever import (
    Retriever,
    RetrievalConfig,
    RetrievalMode,
    RetrievedContext,
    MultiRetriever,
)

# Indexer
from .indexer import (
    Indexer,
    IndexerConfig,
    IndexingResult,
    IndexingStatus,
    IncrementalIndexer,
)


__all__ = [
    # Stores
    "VectorStore",
    "SearchResult",
    "IndexStats",
    "ChromaStore",
    # Chunker
    "Chunker",
    "ChunkerConfig",
    "ChunkingStrategy",
    "TextChunk",
    "ChunkMetadata",
    "FixedSizeChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "CodeAwareChunker",
    "create_chunker",
    # Embedder
    "Embedder",
    "EmbedderConfig",
    "EmbeddingModel",
    "EmbeddingResult",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "SentenceTransformersEmbedder",
    "create_embedder",
    # Retriever
    "Retriever",
    "RetrievalConfig",
    "RetrievalMode",
    "RetrievedContext",
    "MultiRetriever",
    # Indexer
    "Indexer",
    "IndexerConfig",
    "IndexingResult",
    "IndexingStatus",
    "IncrementalIndexer",
]
