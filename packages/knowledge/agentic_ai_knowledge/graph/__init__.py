"""
Graph Package

Knowledge Graph 모듈

사용 시점:
- 엔티티 간 관계를 그래프로 저장/조회할 때
- BFS/DFS 탐색, 경로 찾기가 필요할 때
- 서브그래프 추출이 필요할 때

단순 벡터 검색만 필요하면 agentic-ai-core.rag를 사용하세요.
"""

from .stores import (
    # Base
    GraphStore,
    GraphStats,
    PathResult,
    SubgraphResult,
    TraversalDirection,
    # Implementations
    InMemoryGraphStore,
    Neo4jGraphStore,
)

from .traversal import (
    GraphTraverser,
    TraversalStrategy,
    TraversalOptions,
    TraversalResult,
)

from .query import (
    GraphQuery,
    QueryCondition,
    QueryOperator,
    QueryResult,
    PatternMatcher,
    ContextualSearch,
)


__all__ = [
    # Stores
    "GraphStore",
    "GraphStats",
    "PathResult",
    "SubgraphResult",
    "TraversalDirection",
    "InMemoryGraphStore",
    "Neo4jGraphStore",
    # Traversal
    "GraphTraverser",
    "TraversalStrategy",
    "TraversalOptions",
    "TraversalResult",
    # Query
    "GraphQuery",
    "QueryCondition",
    "QueryOperator",
    "QueryResult",
    "PatternMatcher",
    "ContextualSearch",
]
