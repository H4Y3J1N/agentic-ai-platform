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
    GeminiEmbedder,
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

# Reranker
from .reranker import (
    Reranker,
    RerankerConfig,
    RerankerModel,
    RerankResult,
    CrossEncoderReranker,
    BGEReranker,
    create_reranker,
)

# Query Processor
from .query_processor import (
    QueryProcessor,
    QueryProcessorConfig,
    ProcessedQuery,
    LLMQueryRewriter,
    MultiQueryDecomposer,
    CompositeQueryProcessor,
    create_query_processor,
)

# Hybrid Search
from .hybrid_search import (
    HybridSearchConfig,
    HybridSearchResult,
    BM25Index,
    HybridSearcher,
    create_hybrid_searcher,
)

# RAG Strategies
from .strategies import (
    RAGStrategy,
    RAGStrategyType,
    RAGStrategyConfig,
    RAGResult,
    RetrievalQuality,
    SingleShotRAG,
    CorrectiveRAG,
    SelfRAG,
    create_rag_strategy,
    get_available_strategies,
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
    "GeminiEmbedder",
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
    # Reranker
    "Reranker",
    "RerankerConfig",
    "RerankerModel",
    "RerankResult",
    "CrossEncoderReranker",
    "BGEReranker",
    "create_reranker",
    # Query Processor
    "QueryProcessor",
    "QueryProcessorConfig",
    "ProcessedQuery",
    "LLMQueryRewriter",
    "MultiQueryDecomposer",
    "CompositeQueryProcessor",
    "create_query_processor",
    # Hybrid Search
    "HybridSearchConfig",
    "HybridSearchResult",
    "BM25Index",
    "HybridSearcher",
    "create_hybrid_searcher",
    # RAG Strategies
    "RAGStrategy",
    "RAGStrategyType",
    "RAGStrategyConfig",
    "RAGResult",
    "RetrievalQuality",
    "SingleShotRAG",
    "CorrectiveRAG",
    "SelfRAG",
    "create_rag_strategy",
    "get_available_strategies",
]
