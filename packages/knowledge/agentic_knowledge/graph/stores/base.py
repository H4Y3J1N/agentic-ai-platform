"""
Graph Store Base

그래프 저장소 추상 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from ...schema import Entity, Relationship, EntityType, RelationType


class TraversalDirection(Enum):
    """탐색 방향"""
    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass
class PathResult:
    """경로 탐색 결과"""
    nodes: List[Entity]
    edges: List[Relationship]
    total_weight: float = 0.0

    @property
    def length(self) -> int:
        return len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.model_dump() for n in self.nodes],
            "edges": [e.model_dump() for e in self.edges],
            "length": self.length,
            "total_weight": self.total_weight
        }


@dataclass
class SubgraphResult:
    """서브그래프 결과"""
    entities: List[Entity]
    relationships: List[Relationship]
    center_entity_id: Optional[str] = None
    depth: int = 0

    @property
    def node_count(self) -> int:
        return len(self.entities)

    @property
    def edge_count(self) -> int:
        return len(self.relationships)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.model_dump() for e in self.entities],
            "relationships": [r.model_dump() for r in self.relationships],
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "center_entity_id": self.center_entity_id,
            "depth": self.depth
        }


@dataclass
class GraphStats:
    """그래프 통계"""
    total_entities: int = 0
    total_relationships: int = 0
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    relationship_type_counts: Dict[str, int] = field(default_factory=dict)
    avg_degree: float = 0.0
    connected_components: int = 0


class GraphStore(ABC):
    """그래프 저장소 추상 베이스 클래스"""

    @abstractmethod
    async def add_entity(self, entity: Entity) -> str:
        """엔티티 추가"""
        pass

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """엔티티 조회"""
        pass

    @abstractmethod
    async def update_entity(self, entity: Entity) -> bool:
        """엔티티 업데이트"""
        pass

    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """엔티티 삭제"""
        pass

    @abstractmethod
    async def find_entities(
        self,
        entity_type: Optional[EntityType] = None,
        name_pattern: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Entity]:
        """엔티티 검색"""
        pass

    @abstractmethod
    async def add_relationship(self, relationship: Relationship) -> str:
        """관계 추가"""
        pass

    @abstractmethod
    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """관계 조회"""
        pass

    @abstractmethod
    async def delete_relationship(self, rel_id: str) -> bool:
        """관계 삭제"""
        pass

    @abstractmethod
    async def find_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        rel_type: Optional[RelationType] = None,
        limit: int = 100
    ) -> List[Relationship]:
        """관계 검색"""
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: str,
        direction: TraversalDirection = TraversalDirection.BOTH,
        rel_types: Optional[List[RelationType]] = None,
        depth: int = 1
    ) -> List[Entity]:
        """이웃 엔티티 조회"""
        pass

    @abstractmethod
    async def get_subgraph(
        self,
        entity_id: str,
        depth: int = 2,
        rel_types: Optional[List[RelationType]] = None,
        max_nodes: int = 100
    ) -> SubgraphResult:
        """서브그래프 추출"""
        pass

    @abstractmethod
    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        rel_types: Optional[List[RelationType]] = None
    ) -> Optional[PathResult]:
        """최단 경로 찾기"""
        pass

    @abstractmethod
    async def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        max_paths: int = 10,
        rel_types: Optional[List[RelationType]] = None
    ) -> List[PathResult]:
        """모든 경로 찾기"""
        pass

    @abstractmethod
    async def get_stats(self) -> GraphStats:
        """그래프 통계"""
        pass

    @abstractmethod
    async def count_entities(self, entity_type: Optional[EntityType] = None) -> int:
        """엔티티 수"""
        pass

    @abstractmethod
    async def count_relationships(self, rel_type: Optional[RelationType] = None) -> int:
        """관계 수"""
        pass

    async def add_entities_batch(self, entities: List[Entity]) -> List[str]:
        """엔티티 배치 추가"""
        return [await self.add_entity(e) for e in entities]

    async def add_relationships_batch(self, relationships: List[Relationship]) -> List[str]:
        """관계 배치 추가"""
        return [await self.add_relationship(r) for r in relationships]

    async def clear(self) -> None:
        """그래프 초기화"""
        pass

    async def close(self) -> None:
        """연결 종료"""
        pass
