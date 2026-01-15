"""
Ontology Package

온톨로지 로딩 및 검증 모듈

사용 시점:
- YAML/JSON으로 엔티티/관계 타입을 정의할 때
- 온톨로지 스키마 검증이 필요할 때
"""

from .loader import (
    OntologyLoader,
    OntologyMerger,
    Ontology,
    OntologyFormat,
    EntityTypeDefinition,
    RelationTypeDefinition,
    DecisionTypeDefinition,
    load_default_ontology,
)

from .validator import (
    OntologyValidator,
    EntityValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)


__all__ = [
    # Loader
    "OntologyLoader",
    "OntologyMerger",
    "Ontology",
    "OntologyFormat",
    "EntityTypeDefinition",
    "RelationTypeDefinition",
    "DecisionTypeDefinition",
    "load_default_ontology",
    # Validator
    "OntologyValidator",
    "EntityValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
]
