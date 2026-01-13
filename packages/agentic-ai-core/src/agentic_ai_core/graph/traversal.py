"""
Graph Traversal

그래프 탐색 유틸리티
"""

from typing import List, Dict, Any, Optional, Set, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import heapq
import logging

from .stores.base import GraphStore, PathResult, SubgraphResult, TraversalDirection
from ..schema import Entity, Relationship, EntityType, RelationType

logger = logging.getLogger(__name__)


class TraversalStrategy(Enum):
    """탐색 전략"""
    BFS = "bfs"  # Breadth-First Search
    DFS = "dfs"  # Depth-First Search
    DIJKSTRA = "dijkstra"  # 최단 가중치 경로
    ASTAR = "astar"  # A* 알고리즘


@dataclass
class TraversalOptions:
    """탐색 옵션"""
    max_depth: int = 5
    max_nodes: int = 100
    direction: TraversalDirection = TraversalDirection.BOTH
    rel_types: Optional[List[RelationType]] = None
    entity_types: Optional[List[EntityType]] = None
    min_weight: float = 0.0
    max_weight: float = float('inf')


@dataclass
class TraversalResult:
    """탐색 결과"""
    visited_entities: List[Entity]
    visited_relationships: List[Relationship]
    depth_reached: int
    terminated_early: bool = False
    termination_reason: Optional[str] = None


class GraphTraverser:
    """그래프 탐색기"""

    def __init__(self, store: GraphStore):
        self.store = store

    async def traverse(
        self,
        start_id: str,
        strategy: TraversalStrategy = TraversalStrategy.BFS,
        options: Optional[TraversalOptions] = None,
        visitor: Optional[Callable[[Entity, int], Awaitable[bool]]] = None
    ) -> TraversalResult:
        """
        그래프 탐색

        Args:
            start_id: 시작 엔티티 ID
            strategy: 탐색 전략
            options: 탐색 옵션
            visitor: 방문 콜백 (False 반환 시 탐색 중단)

        Returns:
            탐색 결과
        """
        options = options or TraversalOptions()

        if strategy == TraversalStrategy.BFS:
            return await self._bfs(start_id, options, visitor)
        elif strategy == TraversalStrategy.DFS:
            return await self._dfs(start_id, options, visitor)
        elif strategy == TraversalStrategy.DIJKSTRA:
            return await self._dijkstra(start_id, options, visitor)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    async def _bfs(
        self,
        start_id: str,
        options: TraversalOptions,
        visitor: Optional[Callable[[Entity, int], Awaitable[bool]]] = None
    ) -> TraversalResult:
        """BFS 탐색"""
        start_entity = await self.store.get_entity(start_id)
        if not start_entity:
            return TraversalResult(
                visited_entities=[],
                visited_relationships=[],
                depth_reached=0,
                terminated_early=True,
                termination_reason="Start entity not found"
            )

        visited_entities: List[Entity] = [start_entity]
        visited_relationships: List[Relationship] = []
        visited_ids: Set[str] = {start_id}
        queue: List[tuple] = [(start_id, 0)]  # (entity_id, depth)
        max_depth_reached = 0

        while queue and len(visited_entities) < options.max_nodes:
            current_id, depth = queue.pop(0)

            if depth >= options.max_depth:
                continue

            max_depth_reached = max(max_depth_reached, depth)

            # 이웃 조회
            relationships = await self._get_neighbors_by_direction(
                current_id, options
            )

            for rel in relationships:
                # 가중치 필터
                if rel.weight < options.min_weight or rel.weight > options.max_weight:
                    continue

                # 다음 노드 결정
                next_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == current_id
                    else rel.source_entity_id
                )

                if next_id in visited_ids:
                    continue

                next_entity = await self.store.get_entity(next_id)
                if not next_entity:
                    continue

                # 엔티티 타입 필터
                if options.entity_types and next_entity.entity_type not in options.entity_types:
                    continue

                # 방문자 콜백
                if visitor:
                    should_continue = await visitor(next_entity, depth + 1)
                    if not should_continue:
                        return TraversalResult(
                            visited_entities=visited_entities,
                            visited_relationships=visited_relationships,
                            depth_reached=max_depth_reached,
                            terminated_early=True,
                            termination_reason="Visitor returned False"
                        )

                visited_ids.add(next_id)
                visited_entities.append(next_entity)
                visited_relationships.append(rel)
                queue.append((next_id, depth + 1))

                if len(visited_entities) >= options.max_nodes:
                    break

        return TraversalResult(
            visited_entities=visited_entities,
            visited_relationships=visited_relationships,
            depth_reached=max_depth_reached,
            terminated_early=len(visited_entities) >= options.max_nodes,
            termination_reason="Max nodes reached" if len(visited_entities) >= options.max_nodes else None
        )

    async def _dfs(
        self,
        start_id: str,
        options: TraversalOptions,
        visitor: Optional[Callable[[Entity, int], Awaitable[bool]]] = None
    ) -> TraversalResult:
        """DFS 탐색"""
        start_entity = await self.store.get_entity(start_id)
        if not start_entity:
            return TraversalResult(
                visited_entities=[],
                visited_relationships=[],
                depth_reached=0,
                terminated_early=True,
                termination_reason="Start entity not found"
            )

        visited_entities: List[Entity] = []
        visited_relationships: List[Relationship] = []
        visited_ids: Set[str] = set()
        max_depth_reached = 0
        terminated_early = False
        termination_reason = None

        async def dfs_visit(entity_id: str, depth: int) -> bool:
            nonlocal max_depth_reached, terminated_early, termination_reason

            if entity_id in visited_ids:
                return True
            if depth > options.max_depth:
                return True
            if len(visited_entities) >= options.max_nodes:
                terminated_early = True
                termination_reason = "Max nodes reached"
                return False

            visited_ids.add(entity_id)
            entity = await self.store.get_entity(entity_id)
            if not entity:
                return True

            # 엔티티 타입 필터
            if options.entity_types and entity.entity_type not in options.entity_types:
                return True

            # 방문자 콜백
            if visitor:
                should_continue = await visitor(entity, depth)
                if not should_continue:
                    terminated_early = True
                    termination_reason = "Visitor returned False"
                    return False

            visited_entities.append(entity)
            max_depth_reached = max(max_depth_reached, depth)

            # 이웃 탐색
            relationships = await self._get_neighbors_by_direction(entity_id, options)

            for rel in relationships:
                if rel.weight < options.min_weight or rel.weight > options.max_weight:
                    continue

                next_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == entity_id
                    else rel.source_entity_id
                )

                if next_id not in visited_ids:
                    visited_relationships.append(rel)
                    if not await dfs_visit(next_id, depth + 1):
                        return False

            return True

        await dfs_visit(start_id, 0)

        return TraversalResult(
            visited_entities=visited_entities,
            visited_relationships=visited_relationships,
            depth_reached=max_depth_reached,
            terminated_early=terminated_early,
            termination_reason=termination_reason
        )

    async def _dijkstra(
        self,
        start_id: str,
        options: TraversalOptions,
        visitor: Optional[Callable[[Entity, int], Awaitable[bool]]] = None
    ) -> TraversalResult:
        """다익스트라 탐색 (가중치 기반)"""
        start_entity = await self.store.get_entity(start_id)
        if not start_entity:
            return TraversalResult(
                visited_entities=[],
                visited_relationships=[],
                depth_reached=0,
                terminated_early=True,
                termination_reason="Start entity not found"
            )

        visited_entities: List[Entity] = []
        visited_relationships: List[Relationship] = []
        visited_ids: Set[str] = set()
        distances: Dict[str, float] = {start_id: 0}
        max_depth_reached = 0

        # 우선순위 큐: (distance, depth, entity_id)
        heap: List[tuple] = [(0, 0, start_id)]

        while heap and len(visited_entities) < options.max_nodes:
            distance, depth, current_id = heapq.heappop(heap)

            if current_id in visited_ids:
                continue

            if depth > options.max_depth:
                continue

            visited_ids.add(current_id)
            max_depth_reached = max(max_depth_reached, depth)

            entity = await self.store.get_entity(current_id)
            if not entity:
                continue

            # 엔티티 타입 필터
            if options.entity_types and entity.entity_type not in options.entity_types:
                continue

            # 방문자 콜백
            if visitor:
                should_continue = await visitor(entity, depth)
                if not should_continue:
                    return TraversalResult(
                        visited_entities=visited_entities,
                        visited_relationships=visited_relationships,
                        depth_reached=max_depth_reached,
                        terminated_early=True,
                        termination_reason="Visitor returned False"
                    )

            visited_entities.append(entity)

            # 이웃 탐색
            relationships = await self._get_neighbors_by_direction(current_id, options)

            for rel in relationships:
                if rel.weight < options.min_weight or rel.weight > options.max_weight:
                    continue

                next_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == current_id
                    else rel.source_entity_id
                )

                if next_id in visited_ids:
                    continue

                new_distance = distance + rel.weight
                if next_id not in distances or new_distance < distances[next_id]:
                    distances[next_id] = new_distance
                    visited_relationships.append(rel)
                    heapq.heappush(heap, (new_distance, depth + 1, next_id))

        return TraversalResult(
            visited_entities=visited_entities,
            visited_relationships=visited_relationships,
            depth_reached=max_depth_reached,
            terminated_early=len(visited_entities) >= options.max_nodes,
            termination_reason="Max nodes reached" if len(visited_entities) >= options.max_nodes else None
        )

    async def _get_neighbors_by_direction(
        self,
        entity_id: str,
        options: TraversalOptions
    ) -> List[Relationship]:
        """방향에 따른 이웃 관계 조회"""
        relationships = []

        if options.direction in [TraversalDirection.OUTGOING, TraversalDirection.BOTH]:
            outgoing = await self.store.find_relationships(
                source_id=entity_id,
                rel_type=options.rel_types[0] if options.rel_types and len(options.rel_types) == 1 else None
            )
            if options.rel_types and len(options.rel_types) > 1:
                outgoing = [r for r in outgoing if r.relation_type in options.rel_types]
            relationships.extend(outgoing)

        if options.direction in [TraversalDirection.INCOMING, TraversalDirection.BOTH]:
            incoming = await self.store.find_relationships(
                target_id=entity_id,
                rel_type=options.rel_types[0] if options.rel_types and len(options.rel_types) == 1 else None
            )
            if options.rel_types and len(options.rel_types) > 1:
                incoming = [r for r in incoming if r.relation_type in options.rel_types]
            relationships.extend(incoming)

        return relationships

    async def find_connected_components(self) -> List[List[str]]:
        """연결 컴포넌트 찾기"""
        all_entities = await self.store.find_entities(limit=10000)
        visited: Set[str] = set()
        components: List[List[str]] = []

        for entity in all_entities:
            if entity.id in visited:
                continue

            # BFS로 컴포넌트 탐색
            component: List[str] = []
            queue = [entity.id]

            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue

                visited.add(current_id)
                component.append(current_id)

                # 이웃 추가
                neighbors = await self.store.get_neighbors(
                    current_id,
                    direction=TraversalDirection.BOTH,
                    depth=1
                )
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        queue.append(neighbor.id)

            if component:
                components.append(component)

        return components

    async def find_bridges(self) -> List[Relationship]:
        """브릿지 (절단 간선) 찾기"""
        # Tarjan's algorithm simplified
        all_entities = await self.store.find_entities(limit=10000)
        if not all_entities:
            return []

        visited: Set[str] = set()
        disc: Dict[str, int] = {}
        low: Dict[str, int] = {}
        parent: Dict[str, Optional[str]] = {}
        bridges: List[Relationship] = []
        time = [0]

        async def dfs(entity_id: str):
            visited.add(entity_id)
            disc[entity_id] = time[0]
            low[entity_id] = time[0]
            time[0] += 1

            neighbors_rels = await self._get_neighbors_by_direction(
                entity_id,
                TraversalOptions(direction=TraversalDirection.BOTH)
            )

            for rel in neighbors_rels:
                next_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == entity_id
                    else rel.source_entity_id
                )

                if next_id not in visited:
                    parent[next_id] = entity_id
                    await dfs(next_id)
                    low[entity_id] = min(low[entity_id], low[next_id])

                    if low[next_id] > disc[entity_id]:
                        bridges.append(rel)

                elif next_id != parent.get(entity_id):
                    low[entity_id] = min(low[entity_id], disc[next_id])

        for entity in all_entities:
            if entity.id not in visited:
                parent[entity.id] = None
                await dfs(entity.id)

        return bridges

    async def compute_pagerank(
        self,
        damping: float = 0.85,
        iterations: int = 20
    ) -> Dict[str, float]:
        """PageRank 계산"""
        all_entities = await self.store.find_entities(limit=10000)
        if not all_entities:
            return {}

        entity_ids = [e.id for e in all_entities]
        n = len(entity_ids)

        # 초기화
        ranks: Dict[str, float] = {eid: 1.0 / n for eid in entity_ids}

        for _ in range(iterations):
            new_ranks: Dict[str, float] = {}

            for entity_id in entity_ids:
                # Incoming 링크의 PageRank 합산
                incoming_rels = await self.store.find_relationships(target_id=entity_id)
                rank_sum = 0.0

                for rel in incoming_rels:
                    source_id = rel.source_entity_id
                    # source의 outgoing 링크 수
                    outgoing = await self.store.find_relationships(source_id=source_id)
                    if outgoing:
                        rank_sum += ranks.get(source_id, 0) / len(outgoing)

                new_ranks[entity_id] = (1 - damping) / n + damping * rank_sum

            ranks = new_ranks

        return ranks
