"""
Hybrid Search

벡터 + 그래프 하이브리드 검색
"""

from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging

from .rewriter import QueryRewriter, QueryAnalysis
from .planner import QueryPlanner, QueryPlan, SearchStep, SearchSource, ExecutionOrder
from .fusion import ResultFusion, FusionStrategy, SearchResultItem

logger = logging.getLogger(__name__)


class VectorSearcher(Protocol):
    """벡터 검색 프로토콜"""
    async def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        ...


class GraphSearcher(Protocol):
    """그래프 검색 프로토콜"""
    async def search(
        self,
        entity_ids: List[str],
        depth: int
    ) -> List[Dict[str, Any]]:
        ...


class Embedder(Protocol):
    """임베딩 생성 프로토콜"""
    async def embed(self, text: str) -> List[float]:
        ...


@dataclass
class HybridSearchResult:
    """하이브리드 검색 결과"""
    query: str
    results: List[SearchResultItem]
    total_count: int
    execution_time_ms: float
    plan: Optional[QueryPlan] = None
    analysis: Optional[QueryAnalysis] = None
    source_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class HybridSearchConfig:
    """하이브리드 검색 설정"""
    enable_rewrite: bool = True
    enable_graph: bool = True
    default_top_k: int = 10
    vector_weight: float = 0.6
    graph_weight: float = 0.3
    keyword_weight: float = 0.1
    min_score: float = 0.0
    fusion_strategy: FusionStrategy = FusionStrategy.RRF


class HybridSearcher:
    """하이브리드 검색기"""

    def __init__(
        self,
        vector_store=None,
        graph_store=None,
        embedder=None,
        config: Optional[HybridSearchConfig] = None
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.config = config or HybridSearchConfig()

        self.rewriter = QueryRewriter()
        self.planner = QueryPlanner()
        self.fusion = ResultFusion()

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> HybridSearchResult:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 최대 결과 수
            filters: 필터 조건
            context: 추가 컨텍스트

        Returns:
            하이브리드 검색 결과
        """
        start_time = datetime.now()
        top_k = top_k or self.config.default_top_k
        filters = filters or {}

        # 1. 쿼리 분석
        analysis = None
        if self.config.enable_rewrite:
            analysis = self.rewriter.analyze(query)

        # 2. 실행 계획 수립
        plan = self.planner.plan(query, analysis, context)
        plan = self.planner.optimize(plan)

        # 3. 검색 실행
        step_results = await self._execute_plan(plan, filters)

        # 4. 결과 융합
        fused_results = self._fuse_results(step_results, plan)

        # 5. 점수 필터링 및 정렬
        filtered_results = [
            r for r in fused_results
            if r.score >= self.config.min_score
        ]
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        filtered_results = filtered_results[:top_k]

        # 소스 분포 계산
        source_breakdown = {}
        for result in filtered_results:
            source = result.source
            source_breakdown[source] = source_breakdown.get(source, 0) + 1

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return HybridSearchResult(
            query=query,
            results=filtered_results,
            total_count=len(filtered_results),
            execution_time_ms=execution_time,
            plan=plan,
            analysis=analysis,
            source_breakdown=source_breakdown
        )

    async def _execute_plan(
        self,
        plan: QueryPlan,
        filters: Dict[str, Any]
    ) -> Dict[str, List[SearchResultItem]]:
        """실행 계획 수행"""
        results: Dict[str, List[SearchResultItem]] = {}

        if plan.execution_order == ExecutionOrder.PARALLEL:
            # 병렬 실행
            tasks = [
                self._execute_step(step, filters)
                for step in plan.steps
            ]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(plan.steps, step_results):
                if isinstance(result, Exception):
                    logger.error(f"Step {step.step_id} failed: {result}")
                    results[step.step_id] = []
                else:
                    results[step.step_id] = result

        else:
            # 순차 실행
            for step in plan.steps:
                try:
                    result = await self._execute_step(step, filters)
                    results[step.step_id] = result
                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {e}")
                    results[step.step_id] = []

        return results

    async def _execute_step(
        self,
        step: SearchStep,
        filters: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """단일 검색 단계 실행"""
        merged_filters = {**filters, **step.filters}

        if step.source == SearchSource.VECTOR:
            return await self._vector_search(step.query, step.top_k, merged_filters)
        elif step.source == SearchSource.GRAPH:
            return await self._graph_search(step.query, step.top_k, merged_filters)
        elif step.source == SearchSource.KEYWORD:
            return await self._keyword_search(step.query, step.top_k, merged_filters)
        elif step.source == SearchSource.METADATA:
            return await self._metadata_search(step.query, step.top_k, merged_filters)
        else:
            return []

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """벡터 검색"""
        if not self.vector_store or not self.embedder:
            return []

        try:
            # 쿼리 임베딩 생성
            embedding_result = await self.embedder.embed(query)
            query_embedding = embedding_result.embedding if hasattr(embedding_result, 'embedding') else embedding_result

            # 벡터 검색 수행
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters if filters else None
            )

            # 결과 변환
            return [
                SearchResultItem(
                    id=r.id,
                    content=r.text,
                    score=r.score,
                    source="vector",
                    metadata=r.metadata
                )
                for r in search_results
            ]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _graph_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """그래프 검색"""
        if not self.graph_store:
            return []

        try:
            # 엔티티 ID 추출
            entity_ids = filters.get("entities", [])
            if not entity_ids:
                # 이름으로 엔티티 검색
                entities = await self.graph_store.find_entities(
                    name_pattern=query,
                    limit=5
                )
                entity_ids = [e.id for e in entities]

            if not entity_ids:
                return []

            # 서브그래프 검색
            results = []
            for entity_id in entity_ids[:3]:  # 최대 3개 엔티티
                subgraph = await self.graph_store.get_subgraph(
                    entity_id=entity_id,
                    depth=2,
                    max_nodes=top_k
                )

                for entity in subgraph.entities:
                    results.append(SearchResultItem(
                        id=entity.id,
                        content=f"{entity.name}: {entity.properties}",
                        score=0.8,  # 기본 점수
                        source="graph",
                        metadata={
                            "entity_type": entity.entity_type.value,
                            "properties": entity.properties
                        }
                    ))

            return results[:top_k]

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """키워드 검색"""
        # 벡터 스토어의 메타데이터 필터링 활용
        if not self.vector_store:
            return []

        try:
            # 키워드 필터 추가
            keyword_filters = {
                **filters,
                "content_contains": query
            }

            # 임베딩 없이 메타데이터 기반 검색
            # 실제 구현에서는 전문 검색 엔진 활용
            if hasattr(self.vector_store, 'keyword_search'):
                results = await self.vector_store.keyword_search(
                    query=query,
                    top_k=top_k,
                    filters=keyword_filters
                )
                return [
                    SearchResultItem(
                        id=r.id,
                        content=r.text,
                        score=r.score,
                        source="keyword",
                        metadata=r.metadata
                    )
                    for r in results
                ]

            return []

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    async def _metadata_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """메타데이터 검색"""
        # 시간, 타입 등 메타데이터 기반 필터링
        if not self.vector_store:
            return []

        try:
            # 메타데이터 필터 구성
            metadata_filters = {}

            # 시간 필터
            if "temporal" in filters:
                metadata_filters["date_range"] = filters["temporal"]

            # 타입 필터
            if "document_type" in filters:
                metadata_filters["document_type"] = filters["document_type"]

            if not metadata_filters:
                return []

            # 메타데이터 기반 검색
            if hasattr(self.vector_store, 'filter_by_metadata'):
                results = await self.vector_store.filter_by_metadata(
                    filters=metadata_filters,
                    limit=top_k
                )
                return [
                    SearchResultItem(
                        id=r.id,
                        content=r.text,
                        score=0.5,  # 메타데이터 매칭 기본 점수
                        source="metadata",
                        metadata=r.metadata
                    )
                    for r in results
                ]

            return []

        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    def _fuse_results(
        self,
        step_results: Dict[str, List[SearchResultItem]],
        plan: QueryPlan
    ) -> List[SearchResultItem]:
        """결과 융합"""
        # 가중치 맵 구성
        weights = {}
        for step in plan.steps:
            weights[step.step_id] = step.weight

        # 모든 결과 수집
        all_results = []
        for step_id, results in step_results.items():
            weight = weights.get(step_id, 1.0)
            for result in results:
                result.metadata["step_id"] = step_id
                result.metadata["step_weight"] = weight
                all_results.append(result)

        # 융합 전략에 따른 처리
        strategy = FusionStrategy(plan.fusion_strategy) if isinstance(
            plan.fusion_strategy, str
        ) else self.config.fusion_strategy

        return self.fusion.fuse(all_results, strategy, weights)


class ContextAwareSearcher:
    """컨텍스트 인식 검색기"""

    def __init__(self, hybrid_searcher: HybridSearcher):
        self.searcher = hybrid_searcher
        self.conversation_context: List[Dict[str, Any]] = []

    async def search_with_context(
        self,
        query: str,
        top_k: int = 10,
        use_history: bool = True
    ) -> HybridSearchResult:
        """컨텍스트를 활용한 검색"""
        # 대화 이력 기반 쿼리 확장
        enhanced_query = query
        context = {}

        if use_history and self.conversation_context:
            # 최근 엔티티 추출
            recent_entities = []
            for ctx in self.conversation_context[-3:]:
                if "entities" in ctx:
                    recent_entities.extend(ctx["entities"])

            context["recent_entities"] = list(set(recent_entities))

            # 최근 토픽 추출
            recent_topics = []
            for ctx in self.conversation_context[-3:]:
                if "topics" in ctx:
                    recent_topics.extend(ctx["topics"])

            context["recent_topics"] = list(set(recent_topics))

        # 검색 수행
        result = await self.searcher.search(
            query=enhanced_query,
            top_k=top_k,
            context=context
        )

        # 컨텍스트 업데이트
        self._update_context(query, result)

        return result

    def _update_context(
        self,
        query: str,
        result: HybridSearchResult
    ) -> None:
        """대화 컨텍스트 업데이트"""
        context_entry = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
        }

        if result.analysis:
            context_entry["entities"] = result.analysis.entities
            context_entry["intent"] = result.analysis.intent

        # 결과에서 토픽 추출
        topics = []
        for item in result.results[:3]:
            if "topics" in item.metadata:
                topics.extend(item.metadata["topics"])
        context_entry["topics"] = topics

        self.conversation_context.append(context_entry)

        # 최대 10개 유지
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]

    def clear_context(self) -> None:
        """컨텍스트 초기화"""
        self.conversation_context.clear()
