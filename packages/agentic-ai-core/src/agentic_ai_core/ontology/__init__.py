"""
Ontology Package

온톨로지 로딩 및 검증 모듈
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
