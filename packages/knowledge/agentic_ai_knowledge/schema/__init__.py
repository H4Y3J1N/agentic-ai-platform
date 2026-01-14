"""
Knowledge Schema Package

지식 그래프용 스키마 - Entity, Relationship
"""

from .entity import (
    EntityType,
    EntityRef,
    Entity,
)

from .relationship import (
    RelationType,
    RelationshipRef,
    Relationship,
    RELATION_TYPE_METADATA,
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
]
