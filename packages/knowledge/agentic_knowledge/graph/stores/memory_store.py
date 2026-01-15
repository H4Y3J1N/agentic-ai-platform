"""
In-Memory Graph Store

인메모리 그래프 저장소 구현
"""

from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import re
import logging

from .base import (
    GraphStore,
    GraphStats,
    PathResult,
    SubgraphResult,
    TraversalDirection,
)
from ...schema import Entity, Relationship, EntityType, RelationType

logger = logging.getLogger(__name__)


class InMemoryGraphStore(GraphStore):
    """인메모리 그래프 저장소"""

    def __init__(self):
        # 저장소
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}

        # 인덱스 (빠른 조회용)
        self._entity_by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self._entity_by_name: Dict[str, Set[str]] = defaultdict(set)
        self._outgoing_edges: Dict[str, Set[str]] = defaultdict(set)
        self._incoming_edges: Dict[str, Set[str]] = defaultdict(set)
        self._edges_by_type: Dict[RelationType, Set[str]] = defaultdict(set)

    # ===================
    # Entity Operations
    # ===================

    async def add_entity(self, entity: Entity) -> str:
        """엔티티 추가"""
        self._entities[entity.id] = entity

        # 인덱스 업데이트
        self._entity_by_type[entity.entity_type].add(entity.id)
        name_key = entity.name.lower()
        self._entity_by_name[name_key].add(entity.id)

        logger.debug(f"Added entity: {entity.id} ({entity.name})")
        return entity.id

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """엔티티 조회"""
        return self._entities.get(entity_id)

    async def update_entity(self, entity: Entity) -> bool:
        """엔티티 업데이트"""
        if entity.id not in self._entities:
            return False

        old_entity = self._entities[entity.id]

        # 인덱스 업데이트 (이름이 변경된 경우)
        if old_entity.name != entity.name:
            old_key = old_entity.name.lower()
            new_key = entity.name.lower()
            self._entity_by_name[old_key].discard(entity.id)
            self._entity_by_name[new_key].add(entity.id)

        # 타입이 변경된 경우
        if old_entity.entity_type != entity.entity_type:
            self._entity_by_type[old_entity.entity_type].discard(entity.id)
            self._entity_by_type[entity.entity_type].add(entity.id)

        self._entities[entity.id] = entity
        return True

    async def delete_entity(self, entity_id: str) -> bool:
        """엔티티 삭제"""
        if entity_id not in self._entities:
            return False

        entity = self._entities[entity_id]

        # 관련 관계 삭제
        for rel_id in list(self._outgoing_edges[entity_id]):
            await self.delete_relationship(rel_id)
        for rel_id in list(self._incoming_edges[entity_id]):
            await self.delete_relationship(rel_id)

        # 인덱스 정리
        self._entity_by_type[entity.entity_type].discard(entity_id)
        self._entity_by_name[entity.name.lower()].discard(entity_id)
        del self._outgoing_edges[entity_id]
        del self._incoming_edges[entity_id]

        del self._entities[entity_id]
        return True

    async def find_entities(
        self,
        entity_type: Optional[EntityType] = None,
        name_pattern: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Entity]:
        """엔티티 검색"""
        candidates: Optional[Set[str]] = None

        # 타입 필터
        if entity_type:
            candidates = self._entity_by_type.get(entity_type, set()).copy()

        # 이름 패턴 필터
        if name_pattern:
            pattern = re.compile(name_pattern, re.IGNORECASE)
            name_matches = set()
            for entity_id, entity in self._entities.items():
                if pattern.search(entity.name):
                    name_matches.add(entity_id)

            if candidates is None:
                candidates = name_matches
            else:
                candidates &= name_matches

        # 전체 후보가 없으면 모든 엔티티
        if candidates is None:
            candidates = set(self._entities.keys())

        # 속성 필터
        results = []
        for entity_id in candidates:
            if len(results) >= limit:
                break

            entity = self._entities[entity_id]

            if properties:
                match = all(
                    entity.properties.get(k) == v
                    for k, v in properties.items()
                )
                if not match:
                    continue

            results.append(entity)

        return results

    # ===================
    # Relationship Operations
    # ===================

    async def add_relationship(self, relationship: Relationship) -> str:
        """관계 추가"""
        self._relationships[relationship.id] = relationship

        # 인덱스 업데이트
        self._outgoing_edges[relationship.source_entity_id].add(relationship.id)
        self._incoming_edges[relationship.target_entity_id].add(relationship.id)
        self._edges_by_type[relationship.relation_type].add(relationship.id)

        logger.debug(
            f"Added relationship: {relationship.source_entity_id} "
            f"-[{relationship.relation_type.value}]-> {relationship.target_entity_id}"
        )
        return relationship.id

    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """관계 조회"""
        return self._relationships.get(rel_id)

    async def delete_relationship(self, rel_id: str) -> bool:
        """관계 삭제"""
        if rel_id not in self._relationships:
            return False

        rel = self._relationships[rel_id]

        # 인덱스 정리
        self._outgoing_edges[rel.source_entity_id].discard(rel_id)
        self._incoming_edges[rel.target_entity_id].discard(rel_id)
        self._edges_by_type[rel.relation_type].discard(rel_id)

        del self._relationships[rel_id]
        return True

    async def find_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        rel_type: Optional[RelationType] = None,
        limit: int = 100
    ) -> List[Relationship]:
        """관계 검색"""
        candidates: Optional[Set[str]] = None

        # 소스 필터
        if source_id:
            candidates = self._outgoing_edges.get(source_id, set()).copy()

        # 타겟 필터
        if target_id:
            target_rels = self._incoming_edges.get(target_id, set())
            if candidates is None:
                candidates = target_rels.copy()
            else:
                candidates &= target_rels

        # 타입 필터
        if rel_type:
            type_rels = self._edges_by_type.get(rel_type, set())
            if candidates is None:
                candidates = type_rels.copy()
            else:
                candidates &= type_rels

        # 전체 후보가 없으면 모든 관계
        if candidates is None:
            candidates = set(self._relationships.keys())

        results = []
        for rel_id in candidates:
            if len(results) >= limit:
                break
            results.append(self._relationships[rel_id])

        return results

    # ===================
    # Graph Traversal
    # ===================

    async def get_neighbors(
        self,
        entity_id: str,
        direction: TraversalDirection = TraversalDirection.BOTH,
        rel_types: Optional[List[RelationType]] = None,
        depth: int = 1
    ) -> List[Entity]:
        """이웃 엔티티 조회"""
        if entity_id not in self._entities:
            return []

        visited: Set[str] = {entity_id}
        current_level: Set[str] = {entity_id}
        neighbors: List[Entity] = []

        for _ in range(depth):
            next_level: Set[str] = set()

            for node_id in current_level:
                # Outgoing edges
                if direction in [TraversalDirection.OUTGOING, TraversalDirection.BOTH]:
                    for rel_id in self._outgoing_edges.get(node_id, []):
                        rel = self._relationships[rel_id]

                        if rel_types and rel.relation_type not in rel_types:
                            continue

                        target_id = rel.target_entity_id
                        if target_id not in visited:
                            visited.add(target_id)
                            next_level.add(target_id)
                            neighbors.append(self._entities[target_id])

                # Incoming edges
                if direction in [TraversalDirection.INCOMING, TraversalDirection.BOTH]:
                    for rel_id in self._incoming_edges.get(node_id, []):
                        rel = self._relationships[rel_id]

                        if rel_types and rel.relation_type not in rel_types:
                            continue

                        source_id = rel.source_entity_id
                        if source_id not in visited:
                            visited.add(source_id)
                            next_level.add(source_id)
                            neighbors.append(self._entities[source_id])

            current_level = next_level
            if not current_level:
                break

        return neighbors

    async def get_subgraph(
        self,
        entity_id: str,
        depth: int = 2,
        rel_types: Optional[List[RelationType]] = None,
        max_nodes: int = 100
    ) -> SubgraphResult:
        """서브그래프 추출"""
        if entity_id not in self._entities:
            return SubgraphResult(entities=[], relationships=[], center_entity_id=entity_id)

        visited_nodes: Set[str] = {entity_id}
        visited_edges: Set[str] = set()
        current_level: Set[str] = {entity_id}

        entities = [self._entities[entity_id]]
        relationships = []

        for d in range(depth):
            if len(visited_nodes) >= max_nodes:
                break

            next_level: Set[str] = set()

            for node_id in current_level:
                # Outgoing
                for rel_id in self._outgoing_edges.get(node_id, []):
                    if rel_id in visited_edges:
                        continue

                    rel = self._relationships[rel_id]

                    if rel_types and rel.relation_type not in rel_types:
                        continue

                    visited_edges.add(rel_id)
                    relationships.append(rel)

                    target_id = rel.target_entity_id
                    if target_id not in visited_nodes and len(visited_nodes) < max_nodes:
                        visited_nodes.add(target_id)
                        next_level.add(target_id)
                        entities.append(self._entities[target_id])

                # Incoming
                for rel_id in self._incoming_edges.get(node_id, []):
                    if rel_id in visited_edges:
                        continue

                    rel = self._relationships[rel_id]

                    if rel_types and rel.relation_type not in rel_types:
                        continue

                    visited_edges.add(rel_id)
                    relationships.append(rel)

                    source_id = rel.source_entity_id
                    if source_id not in visited_nodes and len(visited_nodes) < max_nodes:
                        visited_nodes.add(source_id)
                        next_level.add(source_id)
                        entities.append(self._entities[source_id])

            current_level = next_level
            if not current_level:
                break

        return SubgraphResult(
            entities=entities,
            relationships=relationships,
            center_entity_id=entity_id,
            depth=depth
        )

    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        rel_types: Optional[List[RelationType]] = None
    ) -> Optional[PathResult]:
        """BFS로 최단 경로 찾기"""
        if start_id not in self._entities or end_id not in self._entities:
            return None

        if start_id == end_id:
            return PathResult(
                nodes=[self._entities[start_id]],
                edges=[]
            )

        # BFS
        visited: Set[str] = {start_id}
        queue: List[tuple] = [(start_id, [], [])]  # (node_id, path_nodes, path_edges)

        while queue:
            current_id, path_nodes, path_edges = queue.pop(0)

            if len(path_edges) >= max_depth:
                continue

            # 모든 이웃 탐색
            for rel_id in self._outgoing_edges.get(current_id, []):
                rel = self._relationships[rel_id]

                if rel_types and rel.relation_type not in rel_types:
                    continue

                target_id = rel.target_entity_id

                if target_id == end_id:
                    # 경로 발견
                    final_nodes = path_nodes + [self._entities[current_id], self._entities[end_id]]
                    final_edges = path_edges + [rel]
                    return PathResult(
                        nodes=final_nodes,
                        edges=final_edges,
                        total_weight=sum(e.weight for e in final_edges)
                    )

                if target_id not in visited:
                    visited.add(target_id)
                    new_nodes = path_nodes + [self._entities[current_id]]
                    new_edges = path_edges + [rel]
                    queue.append((target_id, new_nodes, new_edges))

            # Incoming도 탐색 (양방향 그래프)
            for rel_id in self._incoming_edges.get(current_id, []):
                rel = self._relationships[rel_id]

                if rel_types and rel.relation_type not in rel_types:
                    continue

                source_id = rel.source_entity_id

                if source_id == end_id:
                    final_nodes = path_nodes + [self._entities[current_id], self._entities[end_id]]
                    final_edges = path_edges + [rel]
                    return PathResult(
                        nodes=final_nodes,
                        edges=final_edges,
                        total_weight=sum(e.weight for e in final_edges)
                    )

                if source_id not in visited:
                    visited.add(source_id)
                    new_nodes = path_nodes + [self._entities[current_id]]
                    new_edges = path_edges + [rel]
                    queue.append((source_id, new_nodes, new_edges))

        return None

    async def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        max_paths: int = 10,
        rel_types: Optional[List[RelationType]] = None
    ) -> List[PathResult]:
        """DFS로 모든 경로 찾기"""
        if start_id not in self._entities or end_id not in self._entities:
            return []

        if start_id == end_id:
            return [PathResult(nodes=[self._entities[start_id]], edges=[])]

        paths: List[PathResult] = []

        def dfs(
            current_id: str,
            visited: Set[str],
            path_nodes: List[Entity],
            path_edges: List[Relationship]
        ):
            if len(paths) >= max_paths:
                return

            if len(path_edges) > max_depth:
                return

            # Outgoing
            for rel_id in self._outgoing_edges.get(current_id, []):
                if len(paths) >= max_paths:
                    return

                rel = self._relationships[rel_id]

                if rel_types and rel.relation_type not in rel_types:
                    continue

                target_id = rel.target_entity_id

                if target_id == end_id:
                    paths.append(PathResult(
                        nodes=path_nodes + [self._entities[end_id]],
                        edges=path_edges + [rel],
                        total_weight=sum(e.weight for e in path_edges) + rel.weight
                    ))
                elif target_id not in visited:
                    visited.add(target_id)
                    dfs(
                        target_id,
                        visited,
                        path_nodes + [self._entities[target_id]],
                        path_edges + [rel]
                    )
                    visited.remove(target_id)

            # Incoming
            for rel_id in self._incoming_edges.get(current_id, []):
                if len(paths) >= max_paths:
                    return

                rel = self._relationships[rel_id]

                if rel_types and rel.relation_type not in rel_types:
                    continue

                source_id = rel.source_entity_id

                if source_id == end_id:
                    paths.append(PathResult(
                        nodes=path_nodes + [self._entities[end_id]],
                        edges=path_edges + [rel],
                        total_weight=sum(e.weight for e in path_edges) + rel.weight
                    ))
                elif source_id not in visited:
                    visited.add(source_id)
                    dfs(
                        source_id,
                        visited,
                        path_nodes + [self._entities[source_id]],
                        path_edges + [rel]
                    )
                    visited.remove(source_id)

        dfs(
            start_id,
            {start_id},
            [self._entities[start_id]],
            []
        )

        return paths

    # ===================
    # Statistics
    # ===================

    async def get_stats(self) -> GraphStats:
        """그래프 통계"""
        entity_counts: Dict[str, int] = {}
        for entity_type, entity_ids in self._entity_by_type.items():
            entity_counts[entity_type.value] = len(entity_ids)

        rel_counts: Dict[str, int] = {}
        for rel_type, rel_ids in self._edges_by_type.items():
            rel_counts[rel_type.value] = len(rel_ids)

        # 평균 차수
        total_degree = sum(
            len(self._outgoing_edges.get(eid, [])) + len(self._incoming_edges.get(eid, []))
            for eid in self._entities
        )
        avg_degree = total_degree / len(self._entities) if self._entities else 0

        # 연결 컴포넌트 수 (간단한 Union-Find)
        components = await self._count_connected_components()

        return GraphStats(
            total_entities=len(self._entities),
            total_relationships=len(self._relationships),
            entity_type_counts=entity_counts,
            relationship_type_counts=rel_counts,
            avg_degree=avg_degree,
            connected_components=components
        )

    async def _count_connected_components(self) -> int:
        """연결 컴포넌트 수 계산"""
        if not self._entities:
            return 0

        visited: Set[str] = set()
        components = 0

        for entity_id in self._entities:
            if entity_id in visited:
                continue

            # BFS로 컴포넌트 탐색
            queue = [entity_id]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)

                for rel_id in self._outgoing_edges.get(node, []):
                    rel = self._relationships[rel_id]
                    if rel.target_entity_id not in visited:
                        queue.append(rel.target_entity_id)

                for rel_id in self._incoming_edges.get(node, []):
                    rel = self._relationships[rel_id]
                    if rel.source_entity_id not in visited:
                        queue.append(rel.source_entity_id)

            components += 1

        return components

    async def count_entities(self, entity_type: Optional[EntityType] = None) -> int:
        """엔티티 수"""
        if entity_type:
            return len(self._entity_by_type.get(entity_type, set()))
        return len(self._entities)

    async def count_relationships(self, rel_type: Optional[RelationType] = None) -> int:
        """관계 수"""
        if rel_type:
            return len(self._edges_by_type.get(rel_type, set()))
        return len(self._relationships)

    # ===================
    # Lifecycle
    # ===================

    async def clear(self) -> None:
        """그래프 초기화"""
        self._entities.clear()
        self._relationships.clear()
        self._entity_by_type.clear()
        self._entity_by_name.clear()
        self._outgoing_edges.clear()
        self._incoming_edges.clear()
        self._edges_by_type.clear()

    async def close(self) -> None:
        """연결 종료 (인메모리는 no-op)"""
        pass
