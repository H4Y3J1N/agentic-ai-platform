"""
Extraction Package

엔티티/관계 추출 모듈

사용 시점:
- 문서에서 엔티티를 자동 추출할 때
- 엔티티 간 관계를 추론할 때
"""

from .extract_stage import (
    ExtractStage,
    DEFAULT_ENTITY_PATTERNS,
)

# Aliases for convenience
EntityExtractor = ExtractStage
RelationshipExtractor = ExtractStage


__all__ = [
    "ExtractStage",
    "EntityExtractor",
    "RelationshipExtractor",
    "DEFAULT_ENTITY_PATTERNS",
]
