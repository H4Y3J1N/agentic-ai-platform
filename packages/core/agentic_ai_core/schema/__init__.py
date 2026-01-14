"""
Schema Package

데이터 스키마 정의 모듈 (core)

Note: Entity, Relationship 스키마는 agentic_ai_knowledge 패키지에서 제공
Note: Decision 스키마는 agentic_ai_decision 패키지에서 제공
"""

# Base classes
from .base import (
    SchemaBase,
    Identifiable,
    Timestamped,
    IdentifiableTimestamped,
    Scorable,
    Provenanced,
)

# Document schemas
from .document import (
    SourceType,
    DocumentType,
    ResolutionLevel,
    DocumentMetadata,
    ParsedSection,
    ParsedContent,
    Chunk,
    Document,
)


__all__ = [
    # Base
    "SchemaBase",
    "Identifiable",
    "Timestamped",
    "IdentifiableTimestamped",
    "Scorable",
    "Provenanced",
    # Document
    "SourceType",
    "DocumentType",
    "ResolutionLevel",
    "DocumentMetadata",
    "ParsedSection",
    "ParsedContent",
    "Chunk",
    "Document",
]
