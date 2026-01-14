"""
Graph Query

그래프 쿼리 및 검색 유틸리티
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

from .stores.base import GraphStore, SubgraphResult
from ..schema import Entity, Relationship, EntityType, RelationType

logger = logging.getLogger(__name__)


class QueryOperator(Enum):
    """쿼리 연산자"""
    EQ = "eq"       # 같음
    NE = "ne"       # 다름
    GT = "gt"       # 초과
    GTE = "gte"     # 이상
    LT = "lt"       # 미만
    LTE = "lte"     # 이하
    IN = "in"       # 포함
    NIN = "nin"     # 미포함
    LIKE = "like"   # 패턴 매칭
    EXISTS = "exists"  # 존재 여부


@dataclass
class QueryCondition:
    """쿼리 조건"""
    field: str
    operator: QueryOperator
    value: Any


@dataclass
class QueryResult:
    """쿼리 결과"""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    total_count: int = 0
    execution_time_ms: float = 0.0


class GraphQuery:
    """그래프 쿼리 빌더"""

    def __init__(self, store: GraphStore):
        self.store = store
        self._entity_conditions: List[QueryCondition] = []
        self._relationship_conditions: List[QueryCondition] = []
        self._entity_types: Optional[List[EntityType]] = None
        self._relationship_types: Optional[List[RelationType]] = None
        self._limit: int = 100
        self._offset: int = 0
        self._order_by: Optional[str] = None
        self._order_desc: bool = False

    def filter_entity_type(self, *types: EntityType) -> "GraphQuery":
        """엔티티 타입 필터"""
        self._entity_types = list(types)
        return self

    def filter_relationship_type(self, *types: RelationType) -> "GraphQuery":
        """관계 타입 필터"""
        self._relationship_types = list(types)
        return self

    def where(
        self,
        field: str,
        operator: Union[QueryOperator, str],
        value: Any
    ) -> "GraphQuery":
        """엔티티 조건 추가"""
        if isinstance(operator, str):
            operator = QueryOperator(operator)

        self._entity_conditions.append(QueryCondition(
            field=field,
            operator=operator,
            value=value
        ))
        return self

    def where_relationship(
        self,
        field: str,
        operator: Union[QueryOperator, str],
        value: Any
    ) -> "GraphQuery":
        """관계 조건 추가"""
        if isinstance(operator, str):
            operator = QueryOperator(operator)

        self._relationship_conditions.append(QueryCondition(
            field=field,
            operator=operator,
            value=value
        ))
        return self

    def limit(self, count: int) -> "GraphQuery":
        """결과 제한"""
        self._limit = count
        return self

    def offset(self, count: int) -> "GraphQuery":
        """결과 오프셋"""
        self._offset = count
        return self

    def order_by(self, field: str, desc: bool = False) -> "GraphQuery":
        """정렬"""
        self._order_by = field
        self._order_desc = desc
        return self

    async def execute(self) -> QueryResult:
        """쿼리 실행"""
        import time
        start_time = time.time()

        # 엔티티 조회
        entities = await self._query_entities()

        # 관계 조회
        relationships = await self._query_relationships()

        execution_time = (time.time() - start_time) * 1000

        return QueryResult(
            entities=entities,
            relationships=relationships,
            total_count=len(entities) + len(relationships),
            execution_time_ms=execution_time
        )

    async def _query_entities(self) -> List[Entity]:
        """엔티티 쿼리"""
        # 기본 조회
        if self._entity_types:
            all_entities = []
            for entity_type in self._entity_types:
                entities = await self.store.find_entities(
                    entity_type=entity_type,
                    limit=self._limit * 10  # 필터링 후 limit 적용을 위해 여유있게
                )
                all_entities.extend(entities)
        else:
            all_entities = await self.store.find_entities(limit=self._limit * 10)

        # 조건 필터링
        filtered = []
        for entity in all_entities:
            if self._matches_conditions(entity, self._entity_conditions):
                filtered.append(entity)

        # 정렬
        if self._order_by:
            filtered = self._sort_entities(filtered)

        # 페이지네이션
        start = self._offset
        end = start + self._limit
        return filtered[start:end]

    async def _query_relationships(self) -> List[Relationship]:
        """관계 쿼리"""
        if not self._relationship_conditions and not self._relationship_types:
            return []

        # 기본 조회
        if self._relationship_types:
            all_relationships = []
            for rel_type in self._relationship_types:
                rels = await self.store.find_relationships(
                    rel_type=rel_type,
                    limit=self._limit * 10
                )
                all_relationships.extend(rels)
        else:
            all_relationships = await self.store.find_relationships(limit=self._limit * 10)

        # 조건 필터링
        filtered = []
        for rel in all_relationships:
            if self._matches_relationship_conditions(rel, self._relationship_conditions):
                filtered.append(rel)

        # 페이지네이션
        start = self._offset
        end = start + self._limit
        return filtered[start:end]

    def _matches_conditions(
        self,
        entity: Entity,
        conditions: List[QueryCondition]
    ) -> bool:
        """엔티티가 조건과 일치하는지 확인"""
        for condition in conditions:
            value = self._get_entity_field(entity, condition.field)

            if not self._evaluate_condition(value, condition.operator, condition.value):
                return False

        return True

    def _matches_relationship_conditions(
        self,
        rel: Relationship,
        conditions: List[QueryCondition]
    ) -> bool:
        """관계가 조건과 일치하는지 확인"""
        for condition in conditions:
            value = self._get_relationship_field(rel, condition.field)

            if not self._evaluate_condition(value, condition.operator, condition.value):
                return False

        return True

    def _get_entity_field(self, entity: Entity, field: str) -> Any:
        """엔티티 필드 값 조회"""
        if field == "name":
            return entity.name
        elif field == "entity_type":
            return entity.entity_type
        elif field.startswith("properties."):
            prop_name = field[11:]
            return entity.properties.get(prop_name)
        elif field == "created_at":
            return entity.created_at
        elif field == "updated_at":
            return entity.updated_at
        else:
            return entity.properties.get(field)

    def _get_relationship_field(self, rel: Relationship, field: str) -> Any:
        """관계 필드 값 조회"""
        if field == "relation_type":
            return rel.relation_type
        elif field == "weight":
            return rel.weight
        elif field.startswith("properties."):
            prop_name = field[11:]
            return rel.properties.get(prop_name)
        elif field == "created_at":
            return rel.created_at
        else:
            return rel.properties.get(field)

    def _evaluate_condition(
        self,
        actual: Any,
        operator: QueryOperator,
        expected: Any
    ) -> bool:
        """조건 평가"""
        if operator == QueryOperator.EQ:
            return actual == expected
        elif operator == QueryOperator.NE:
            return actual != expected
        elif operator == QueryOperator.GT:
            return actual is not None and actual > expected
        elif operator == QueryOperator.GTE:
            return actual is not None and actual >= expected
        elif operator == QueryOperator.LT:
            return actual is not None and actual < expected
        elif operator == QueryOperator.LTE:
            return actual is not None and actual <= expected
        elif operator == QueryOperator.IN:
            return actual in expected
        elif operator == QueryOperator.NIN:
            return actual not in expected
        elif operator == QueryOperator.LIKE:
            if actual is None:
                return False
            pattern = expected.replace("%", ".*").replace("_", ".")
            return bool(re.match(pattern, str(actual), re.IGNORECASE))
        elif operator == QueryOperator.EXISTS:
            return (actual is not None) == expected
        else:
            return False

    def _sort_entities(self, entities: List[Entity]) -> List[Entity]:
        """엔티티 정렬"""
        def get_sort_key(entity: Entity):
            value = self._get_entity_field(entity, self._order_by)
            # None 처리
            if value is None:
                return "" if isinstance(value, str) else 0
            return value

        return sorted(entities, key=get_sort_key, reverse=self._order_desc)


class PatternMatcher:
    """그래프 패턴 매칭"""

    def __init__(self, store: GraphStore):
        self.store = store

    async def match_pattern(
        self,
        pattern: str,
        bindings: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Entity]]:
        """
        간단한 패턴 매칭

        Pattern format:
        - (a:Person)-[r:WORKS_FOR]->(b:Organization)
        - (a)-[r]->(b)

        Args:
            pattern: 패턴 문자열
            bindings: 초기 바인딩

        Returns:
            매칭 결과 리스트 (변수 -> 엔티티 매핑)
        """
        bindings = bindings or {}

        # 패턴 파싱
        parsed = self._parse_pattern(pattern)
        if not parsed:
            return []

        # 패턴 실행
        return await self._execute_pattern(parsed, bindings)

    def _parse_pattern(self, pattern: str) -> Optional[Dict[str, Any]]:
        """패턴 파싱"""
        # 간단한 패턴: (a:Type)-[r:RelType]->(b:Type)
        node_pattern = r'\((\w+)(?::(\w+))?\)'
        rel_pattern = r'\[(\w+)(?::(\w+))?\]'

        # 노드 추출
        nodes = re.findall(node_pattern, pattern)
        if len(nodes) < 2:
            return None

        # 관계 추출
        rels = re.findall(rel_pattern, pattern)

        # 방향 결정
        is_outgoing = "->" in pattern
        is_incoming = "<-" in pattern

        return {
            "start_node": {"var": nodes[0][0], "type": nodes[0][1] or None},
            "end_node": {"var": nodes[-1][0], "type": nodes[-1][1] or None},
            "relationship": {
                "var": rels[0][0] if rels else "r",
                "type": rels[0][1] if rels else None
            },
            "direction": "outgoing" if is_outgoing else ("incoming" if is_incoming else "both")
        }

    async def _execute_pattern(
        self,
        parsed: Dict[str, Any],
        bindings: Dict[str, Any]
    ) -> List[Dict[str, Entity]]:
        """패턴 실행"""
        results = []

        # 시작 노드 후보
        start_var = parsed["start_node"]["var"]
        start_type = parsed["start_node"]["type"]

        if start_var in bindings:
            start_candidates = [bindings[start_var]]
        else:
            entity_type = EntityType(start_type.lower()) if start_type else None
            start_candidates = await self.store.find_entities(
                entity_type=entity_type,
                limit=100
            )

        # 각 시작 노드에 대해 매칭
        for start_entity in start_candidates:
            # 관계 조회
            rel_type = None
            if parsed["relationship"]["type"]:
                try:
                    rel_type = RelationType(parsed["relationship"]["type"].lower())
                except ValueError:
                    pass

            if parsed["direction"] == "outgoing":
                relationships = await self.store.find_relationships(
                    source_id=start_entity.id,
                    rel_type=rel_type
                )
            elif parsed["direction"] == "incoming":
                relationships = await self.store.find_relationships(
                    target_id=start_entity.id,
                    rel_type=rel_type
                )
            else:
                outgoing = await self.store.find_relationships(
                    source_id=start_entity.id,
                    rel_type=rel_type
                )
                incoming = await self.store.find_relationships(
                    target_id=start_entity.id,
                    rel_type=rel_type
                )
                relationships = outgoing + incoming

            # 끝 노드 필터
            end_var = parsed["end_node"]["var"]
            end_type = parsed["end_node"]["type"]

            for rel in relationships:
                end_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == start_entity.id
                    else rel.source_entity_id
                )

                end_entity = await self.store.get_entity(end_id)
                if not end_entity:
                    continue

                # 타입 체크
                if end_type:
                    try:
                        expected_type = EntityType(end_type.lower())
                        if end_entity.entity_type != expected_type:
                            continue
                    except ValueError:
                        pass

                # 바인딩 체크
                if end_var in bindings and bindings[end_var].id != end_entity.id:
                    continue

                results.append({
                    start_var: start_entity,
                    end_var: end_entity,
                    parsed["relationship"]["var"]: rel
                })

        return results


class ContextualSearch:
    """컨텍스트 기반 검색"""

    def __init__(self, store: GraphStore):
        self.store = store

    async def search_with_context(
        self,
        query_entities: List[str],
        context_depth: int = 2,
        max_results: int = 50
    ) -> SubgraphResult:
        """
        쿼리 엔티티들의 컨텍스트를 포함한 서브그래프 검색

        Args:
            query_entities: 쿼리 엔티티 ID 리스트
            context_depth: 컨텍스트 탐색 깊이
            max_results: 최대 노드 수

        Returns:
            서브그래프 결과
        """
        all_entities: Dict[str, Entity] = {}
        all_relationships: Dict[str, Relationship] = {}

        for entity_id in query_entities:
            subgraph = await self.store.get_subgraph(
                entity_id=entity_id,
                depth=context_depth,
                max_nodes=max_results // len(query_entities) if query_entities else max_results
            )

            for entity in subgraph.entities:
                all_entities[entity.id] = entity

            for rel in subgraph.relationships:
                all_relationships[rel.id] = rel

        return SubgraphResult(
            entities=list(all_entities.values())[:max_results],
            relationships=list(all_relationships.values()),
            center_entity_id=query_entities[0] if query_entities else None,
            depth=context_depth
        )

    async def find_common_neighbors(
        self,
        entity_ids: List[str],
        depth: int = 1
    ) -> List[Entity]:
        """
        여러 엔티티의 공통 이웃 찾기

        Args:
            entity_ids: 엔티티 ID 리스트
            depth: 탐색 깊이

        Returns:
            공통 이웃 엔티티 리스트
        """
        if not entity_ids:
            return []

        # 첫 번째 엔티티의 이웃
        neighbors_sets: List[set] = []

        for entity_id in entity_ids:
            neighbors = await self.store.get_neighbors(
                entity_id=entity_id,
                depth=depth
            )
            neighbor_ids = {n.id for n in neighbors}
            neighbors_sets.append(neighbor_ids)

        # 교집합
        common_ids = neighbors_sets[0]
        for ns in neighbors_sets[1:]:
            common_ids &= ns

        # 엔티티 조회
        common_entities = []
        for entity_id in common_ids:
            entity = await self.store.get_entity(entity_id)
            if entity:
                common_entities.append(entity)

        return common_entities

    async def find_connecting_entities(
        self,
        source_id: str,
        target_id: str,
        max_intermediates: int = 5
    ) -> List[Entity]:
        """
        두 엔티티를 연결하는 중간 엔티티들 찾기

        Args:
            source_id: 소스 엔티티 ID
            target_id: 타겟 엔티티 ID
            max_intermediates: 최대 중간 엔티티 수

        Returns:
            중간 엔티티 리스트
        """
        path = await self.store.find_path(
            start_id=source_id,
            end_id=target_id,
            max_depth=max_intermediates + 1
        )

        if not path:
            return []

        # 시작과 끝 제외한 중간 노드들
        return path.nodes[1:-1] if len(path.nodes) > 2 else []
