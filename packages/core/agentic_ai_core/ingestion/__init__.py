"""
Ingestion Package

데이터 수집 파이프라인 모듈
"""

from .context import (
    SourceItem,
    ProcessingStatus,
    VectorizationDecision,
    StageError,
    PipelineContext,
)

from .pipeline import (
    Pipeline,
    PipelineBuilder,
    PipelineResult,
    BatchResult,
    create_default_pipeline,
    create_lightweight_pipeline,
    create_indexing_pipeline,
)

from .stages import (
    # Base
    Stage,
    ConditionalStage,
    CompositeStage,
    ParallelStage,
    # Stages
    ParseStage,
    ExtractStage,
    InferStage,
    ScoreStage,
    VectorizeStage,
    StoreStage,
    # Stores
    GraphStore,
    DocumentStore,
    InMemoryDocumentStore,
    InMemoryGraphStore,
    # Parse utilities
    BaseParser,
    DefaultParser,
    # Extract utilities
    DEFAULT_ENTITY_PATTERNS,
    # Score utilities
    DEFAULT_WEIGHTS,
)

from .parsers import (
    Parser,
    ParserRegistry,
    RichText,
    NotionParser,
    NotionMeetingNoteParser,
    SlackParser,
    SlackAnnouncementParser,
)


__all__ = [
    # Context
    "SourceItem",
    "ProcessingStatus",
    "VectorizationDecision",
    "StageError",
    "PipelineContext",
    # Pipeline
    "Pipeline",
    "PipelineBuilder",
    "PipelineResult",
    "BatchResult",
    "create_default_pipeline",
    "create_lightweight_pipeline",
    "create_indexing_pipeline",
    # Base Stages
    "Stage",
    "ConditionalStage",
    "CompositeStage",
    "ParallelStage",
    # Stages
    "ParseStage",
    "ExtractStage",
    "InferStage",
    "ScoreStage",
    "VectorizeStage",
    "StoreStage",
    # Stores
    "GraphStore",
    "DocumentStore",
    "InMemoryDocumentStore",
    "InMemoryGraphStore",
    # Parse utilities
    "BaseParser",
    "DefaultParser",
    # Extract utilities
    "DEFAULT_ENTITY_PATTERNS",
    # Score utilities
    "DEFAULT_WEIGHTS",
    # Parsers
    "Parser",
    "ParserRegistry",
    "RichText",
    "NotionParser",
    "NotionMeetingNoteParser",
    "SlackParser",
    "SlackAnnouncementParser",
]
