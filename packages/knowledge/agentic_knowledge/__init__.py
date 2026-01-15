"""
Agentic Knowledge Package

지식 그래프 확장 모듈 - Entity, Relationship, Graph, Ontology

사용 시점:
- 엔티티/관계 추출이 필요한 도메인
- 지식 그래프 기반 검색이 필요한 경우
- 온톨로지 기반 스키마 관리가 필요한 경우

단순 RAG만 필요하면 agentic-core만 사용하세요.
"""

# Schema
from .schema import (
    # Entity
    EntityType,
    EntityRef,
    Entity,
    # Relationship
    RelationType,
    RelationshipRef,
    Relationship,
    RELATION_TYPE_METADATA,
)

# Graph
from .graph import (
    # Stores
    GraphStore,
    GraphStats,
    PathResult,
    SubgraphResult,
    TraversalDirection,
    InMemoryGraphStore,
    Neo4jGraphStore,
    # Traversal
    GraphTraverser,
    TraversalStrategy,
    TraversalOptions,
    TraversalResult,
    # Query
    GraphQuery,
    QueryCondition,
    QueryOperator,
    QueryResult,
    PatternMatcher,
    ContextualSearch,
)

# Ontology
from .ontology import (
    OntologyLoader,
    OntologyMerger,
    Ontology,
    OntologyFormat,
    EntityTypeDefinition,
    RelationTypeDefinition,
    OntologyValidator,
    EntityValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    load_default_ontology,
)

# Extraction
from .extraction import (
    ExtractStage,
    EntityExtractor,
    RelationshipExtractor,
    DEFAULT_ENTITY_PATTERNS,
)


__all__ = [
    # Entity
    "EntityType",
    "EntityRef",
    "Entity",
    # Relationship
    "RelationType",
    "RelationshipRef",
    "Relationship",
    "RELATION_TYPE_METADATA",
    # Graph Stores
    "GraphStore",
    "GraphStats",
    "PathResult",
    "SubgraphResult",
    "TraversalDirection",
    "InMemoryGraphStore",
    "Neo4jGraphStore",
    # Graph Traversal
    "GraphTraverser",
    "TraversalStrategy",
    "TraversalOptions",
    "TraversalResult",
    # Graph Query
    "GraphQuery",
    "QueryCondition",
    "QueryOperator",
    "QueryResult",
    "PatternMatcher",
    "ContextualSearch",
    # Ontology
    "OntologyLoader",
    "OntologyMerger",
    "Ontology",
    "OntologyFormat",
    "EntityTypeDefinition",
    "RelationTypeDefinition",
    "OntologyValidator",
    "EntityValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "load_default_ontology",
    # Extraction
    "ExtractStage",
    "EntityExtractor",
    "RelationshipExtractor",
    "DEFAULT_ENTITY_PATTERNS",
]
