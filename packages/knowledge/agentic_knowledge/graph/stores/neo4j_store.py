"""
Neo4j Graph Store

Neo4j 그래프 데이터베이스 저장소 구현
"""

from typing import List, Dict, Any, Optional
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


class Neo4jGraphStore(GraphStore):
    """Neo4j 그래프 저장소"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None

    async def _ensure_driver(self):
        """드라이버 초기화"""
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase
                self._driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
            except ImportError:
                raise ImportError(
                    "neo4j package required. Install with: pip install neo4j"
                )

    async def _run_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        """쿼리 실행"""
        await self._ensure_driver()
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, params or {})
            return [record.data() async for record in result]

    # ===================
    # Entity Operations
    # ===================

    async def add_entity(self, entity: Entity) -> str:
        """엔티티 추가"""
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.entity_type = $entity_type,
            e.properties = $properties,
            e.source_ids = $source_ids,
            e.created_at = $created_at,
            e.updated_at = $updated_at
        RETURN e.id as id
        """
        params = {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "properties": entity.properties,
            "source_ids": entity.source_ids,
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
        }
        await self._run_query(query, params)
        return entity.id

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """엔티티 조회"""
        query = """
        MATCH (e:Entity {id: $id})
        RETURN e
        """
        results = await self._run_query(query, {"id": entity_id})
        if not results:
            return None

        data = results[0]["e"]
        return self._dict_to_entity(data)

    async def update_entity(self, entity: Entity) -> bool:
        """엔티티 업데이트"""
        query = """
        MATCH (e:Entity {id: $id})
        SET e.name = $name,
            e.entity_type = $entity_type,
            e.properties = $properties,
            e.source_ids = $source_ids,
            e.updated_at = $updated_at
        RETURN e.id as id
        """
        params = {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "properties": entity.properties,
            "source_ids": entity.source_ids,
            "updated_at": entity.updated_at.isoformat(),
        }
        results = await self._run_query(query, params)
        return len(results) > 0

    async def delete_entity(self, entity_id: str) -> bool:
        """엔티티 삭제"""
        query = """
        MATCH (e:Entity {id: $id})
        DETACH DELETE e
        RETURN count(e) as deleted
        """
        results = await self._run_query(query, {"id": entity_id})
        return results[0]["deleted"] > 0 if results else False

    async def find_entities(
        self,
        entity_type: Optional[EntityType] = None,
        name_pattern: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Entity]:
        """엔티티 검색"""
        conditions = []
        params = {"limit": limit}

        if entity_type:
            conditions.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type.value

        if name_pattern:
            conditions.append("e.name =~ $name_pattern")
            params["name_pattern"] = f"(?i).*{name_pattern}.*"

        where_clause = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (e:Entity)
        WHERE {where_clause}
        RETURN e
        LIMIT $limit
        """

        results = await self._run_query(query, params)

        entities = []
        for record in results:
            entity = self._dict_to_entity(record["e"])

            # 속성 필터 (클라이언트 사이드)
            if properties:
                match = all(
                    entity.properties.get(k) == v
                    for k, v in properties.items()
                )
                if not match:
                    continue

            entities.append(entity)

        return entities

    # ===================
    # Relationship Operations
    # ===================

    async def add_relationship(self, relationship: Relationship) -> str:
        """관계 추가"""
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATES_TO {id: $id}]->(target)
        SET r.relation_type = $relation_type,
            r.weight = $weight,
            r.properties = $properties,
            r.source_ids = $source_ids,
            r.created_at = $created_at
        RETURN r.id as id
        """
        params = {
            "id": relationship.id,
            "source_id": relationship.source_entity_id,
            "target_id": relationship.target_entity_id,
            "relation_type": relationship.relation_type.value,
            "weight": relationship.weight,
            "properties": relationship.properties,
            "source_ids": relationship.source_ids,
            "created_at": relationship.created_at.isoformat(),
        }
        await self._run_query(query, params)
        return relationship.id

    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """관계 조회"""
        query = """
        MATCH (source:Entity)-[r:RELATES_TO {id: $id}]->(target:Entity)
        RETURN r, source.id as source_id, target.id as target_id
        """
        results = await self._run_query(query, {"id": rel_id})
        if not results:
            return None

        data = results[0]
        return self._dict_to_relationship(
            data["r"],
            data["source_id"],
            data["target_id"]
        )

    async def delete_relationship(self, rel_id: str) -> bool:
        """관계 삭제"""
        query = """
        MATCH ()-[r:RELATES_TO {id: $id}]->()
        DELETE r
        RETURN count(r) as deleted
        """
        results = await self._run_query(query, {"id": rel_id})
        return results[0]["deleted"] > 0 if results else False

    async def find_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        rel_type: Optional[RelationType] = None,
        limit: int = 100
    ) -> List[Relationship]:
        """관계 검색"""
        conditions = []
        params = {"limit": limit}

        if source_id:
            conditions.append("source.id = $source_id")
            params["source_id"] = source_id

        if target_id:
            conditions.append("target.id = $target_id")
            params["target_id"] = target_id

        if rel_type:
            conditions.append("r.relation_type = $rel_type")
            params["rel_type"] = rel_type.value

        where_clause = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
        WHERE {where_clause}
        RETURN r, source.id as source_id, target.id as target_id
        LIMIT $limit
        """

        results = await self._run_query(query, params)

        return [
            self._dict_to_relationship(
                record["r"],
                record["source_id"],
                record["target_id"]
            )
            for record in results
        ]

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
        rel_filter = ""
        if rel_types:
            types_str = "|".join(t.value for t in rel_types)
            rel_filter = f"[r:RELATES_TO WHERE r.relation_type IN $rel_types]"
        else:
            rel_filter = "[r:RELATES_TO]"

        if direction == TraversalDirection.OUTGOING:
            pattern = f"-{rel_filter}->"
        elif direction == TraversalDirection.INCOMING:
            pattern = f"<-{rel_filter}-"
        else:
            pattern = f"-{rel_filter}-"

        query = f"""
        MATCH (start:Entity {{id: $entity_id}}){pattern}(neighbor:Entity)
        WHERE neighbor.id <> $entity_id
        RETURN DISTINCT neighbor
        """

        params = {"entity_id": entity_id}
        if rel_types:
            params["rel_types"] = [t.value for t in rel_types]

        results = await self._run_query(query, params)

        return [self._dict_to_entity(record["neighbor"]) for record in results]

    async def get_subgraph(
        self,
        entity_id: str,
        depth: int = 2,
        rel_types: Optional[List[RelationType]] = None,
        max_nodes: int = 100
    ) -> SubgraphResult:
        """서브그래프 추출"""
        rel_filter = ""
        if rel_types:
            rel_filter = "WHERE r.relation_type IN $rel_types"

        query = f"""
        MATCH (center:Entity {{id: $entity_id}})
        CALL apoc.path.subgraphAll(center, {{
            maxLevel: $depth,
            limit: $max_nodes
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """

        # Fallback for when APOC is not available
        fallback_query = f"""
        MATCH path = (center:Entity {{id: $entity_id}})-[r:RELATES_TO*1..{depth}]-(connected:Entity)
        {rel_filter}
        WITH collect(DISTINCT center) + collect(DISTINCT connected) as nodes,
             collect(DISTINCT r) as rels
        UNWIND nodes as n
        WITH collect(DISTINCT n)[0..$max_nodes] as limited_nodes, rels
        RETURN limited_nodes as nodes, rels as relationships
        """

        params = {
            "entity_id": entity_id,
            "depth": depth,
            "max_nodes": max_nodes,
        }
        if rel_types:
            params["rel_types"] = [t.value for t in rel_types]

        try:
            results = await self._run_query(query, params)
        except Exception:
            results = await self._run_query(fallback_query, params)

        if not results:
            return SubgraphResult(
                entities=[],
                relationships=[],
                center_entity_id=entity_id,
                depth=depth
            )

        data = results[0]
        entities = [self._dict_to_entity(n) for n in data.get("nodes", [])]

        # 관계 파싱
        relationships = []
        for rel_list in data.get("relationships", []):
            if isinstance(rel_list, list):
                for r in rel_list:
                    # Neo4j relationship 객체에서 데이터 추출
                    relationships.append(self._neo4j_rel_to_relationship(r))
            else:
                relationships.append(self._neo4j_rel_to_relationship(rel_list))

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
        """최단 경로 찾기"""
        rel_filter = ""
        if rel_types:
            rel_filter = "WHERE ALL(r IN relationships(p) WHERE r.relation_type IN $rel_types)"

        query = f"""
        MATCH (start:Entity {{id: $start_id}}), (end:Entity {{id: $end_id}})
        MATCH p = shortestPath((start)-[*1..{max_depth}]-(end))
        {rel_filter}
        RETURN nodes(p) as nodes, relationships(p) as rels
        """

        params = {
            "start_id": start_id,
            "end_id": end_id,
        }
        if rel_types:
            params["rel_types"] = [t.value for t in rel_types]

        results = await self._run_query(query, params)

        if not results:
            return None

        data = results[0]
        nodes = [self._dict_to_entity(n) for n in data["nodes"]]
        edges = [self._neo4j_rel_to_relationship(r) for r in data["rels"]]

        return PathResult(
            nodes=nodes,
            edges=edges,
            total_weight=sum(e.weight for e in edges)
        )

    async def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        max_paths: int = 10,
        rel_types: Optional[List[RelationType]] = None
    ) -> List[PathResult]:
        """모든 경로 찾기"""
        rel_filter = ""
        if rel_types:
            rel_filter = "WHERE ALL(r IN relationships(p) WHERE r.relation_type IN $rel_types)"

        query = f"""
        MATCH (start:Entity {{id: $start_id}}), (end:Entity {{id: $end_id}})
        MATCH p = (start)-[*1..{max_depth}]-(end)
        {rel_filter}
        RETURN nodes(p) as nodes, relationships(p) as rels
        LIMIT $max_paths
        """

        params = {
            "start_id": start_id,
            "end_id": end_id,
            "max_paths": max_paths,
        }
        if rel_types:
            params["rel_types"] = [t.value for t in rel_types]

        results = await self._run_query(query, params)

        paths = []
        for data in results:
            nodes = [self._dict_to_entity(n) for n in data["nodes"]]
            edges = [self._neo4j_rel_to_relationship(r) for r in data["rels"]]

            paths.append(PathResult(
                nodes=nodes,
                edges=edges,
                total_weight=sum(e.weight for e in edges)
            ))

        return paths

    # ===================
    # Statistics
    # ===================

    async def get_stats(self) -> GraphStats:
        """그래프 통계"""
        # 엔티티 수
        entity_query = """
        MATCH (e:Entity)
        RETURN e.entity_type as type, count(*) as count
        """
        entity_results = await self._run_query(entity_query)

        entity_counts = {}
        total_entities = 0
        for record in entity_results:
            entity_counts[record["type"]] = record["count"]
            total_entities += record["count"]

        # 관계 수
        rel_query = """
        MATCH ()-[r:RELATES_TO]->()
        RETURN r.relation_type as type, count(*) as count
        """
        rel_results = await self._run_query(rel_query)

        rel_counts = {}
        total_rels = 0
        for record in rel_results:
            rel_counts[record["type"]] = record["count"]
            total_rels += record["count"]

        # 평균 차수
        degree_query = """
        MATCH (e:Entity)
        RETURN avg(size((e)-[]->())) + avg(size((e)<-[]-())) as avg_degree
        """
        degree_results = await self._run_query(degree_query)
        avg_degree = degree_results[0]["avg_degree"] if degree_results else 0

        return GraphStats(
            total_entities=total_entities,
            total_relationships=total_rels,
            entity_type_counts=entity_counts,
            relationship_type_counts=rel_counts,
            avg_degree=avg_degree or 0,
            connected_components=0  # 별도 계산 필요
        )

    async def count_entities(self, entity_type: Optional[EntityType] = None) -> int:
        """엔티티 수"""
        if entity_type:
            query = """
            MATCH (e:Entity {entity_type: $type})
            RETURN count(e) as count
            """
            results = await self._run_query(query, {"type": entity_type.value})
        else:
            query = "MATCH (e:Entity) RETURN count(e) as count"
            results = await self._run_query(query)

        return results[0]["count"] if results else 0

    async def count_relationships(self, rel_type: Optional[RelationType] = None) -> int:
        """관계 수"""
        if rel_type:
            query = """
            MATCH ()-[r:RELATES_TO {relation_type: $type}]->()
            RETURN count(r) as count
            """
            results = await self._run_query(query, {"type": rel_type.value})
        else:
            query = "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
            results = await self._run_query(query)

        return results[0]["count"] if results else 0

    # ===================
    # Helpers
    # ===================

    def _dict_to_entity(self, data: Dict[str, Any]) -> Entity:
        """딕셔너리를 Entity로 변환"""
        from datetime import datetime

        return Entity(
            id=data.get("id"),
            name=data.get("name", ""),
            entity_type=EntityType(data.get("entity_type", "other")),
            properties=data.get("properties", {}),
            source_ids=data.get("source_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )

    def _dict_to_relationship(
        self,
        data: Dict[str, Any],
        source_id: str,
        target_id: str
    ) -> Relationship:
        """딕셔너리를 Relationship으로 변환"""
        from datetime import datetime

        return Relationship(
            id=data.get("id"),
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=RelationType(data.get("relation_type", "related_to")),
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
            source_ids=data.get("source_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )

    def _neo4j_rel_to_relationship(self, rel) -> Relationship:
        """Neo4j relationship 객체를 Relationship으로 변환"""
        from datetime import datetime

        # Neo4j relationship에서 데이터 추출
        if hasattr(rel, '_properties'):
            data = rel._properties
        else:
            data = dict(rel) if hasattr(rel, '__iter__') else {}

        return Relationship(
            id=data.get("id", ""),
            source_entity_id=data.get("source_id", ""),
            target_entity_id=data.get("target_id", ""),
            relation_type=RelationType(data.get("relation_type", "related_to")),
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
            source_ids=data.get("source_ids", []),
        )

    # ===================
    # Lifecycle
    # ===================

    async def clear(self) -> None:
        """그래프 초기화"""
        await self._run_query("MATCH (n) DETACH DELETE n")

    async def close(self) -> None:
        """연결 종료"""
        if self._driver:
            await self._driver.close()
            self._driver = None
