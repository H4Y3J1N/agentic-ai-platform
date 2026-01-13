"""
Query Planner

쿼리 실행 계획 수립
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from .rewriter import QueryAnalysis

logger = logging.getLogger(__name__)


class SearchSource(Enum):
    """검색 소스"""
    VECTOR = "vector"           # 벡터 검색
    GRAPH = "graph"             # 그래프 검색
    KEYWORD = "keyword"         # 키워드 검색
    METADATA = "metadata"       # 메타데이터 검색


class ExecutionOrder(Enum):
    """실행 순서"""
    PARALLEL = "parallel"       # 병렬 실행
    SEQUENTIAL = "sequential"   # 순차 실행
    CONDITIONAL = "conditional" # 조건부 실행


@dataclass
class SearchStep:
    """검색 단계"""
    source: SearchSource
    query: str
    weight: float = 1.0
    top_k: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    depends_on: Optional[str] = None  # 의존하는 이전 단계 ID
    step_id: str = ""


@dataclass
class QueryPlan:
    """쿼리 실행 계획"""
    original_query: str
    steps: List[SearchStep]
    execution_order: ExecutionOrder
    fusion_strategy: str = "rrf"  # reciprocal rank fusion
    estimated_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerConfig:
    """플래너 설정"""
    enable_vector: bool = True
    enable_graph: bool = True
    enable_keyword: bool = True
    enable_metadata: bool = True
    default_top_k: int = 10
    max_parallel_steps: int = 3
    vector_weight: float = 0.6
    graph_weight: float = 0.3
    keyword_weight: float = 0.1


class QueryPlanner:
    """쿼리 실행 계획 수립기"""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def plan(
        self,
        query: str,
        analysis: Optional[QueryAnalysis] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        쿼리 실행 계획 수립

        Args:
            query: 원본 쿼리
            analysis: 쿼리 분석 결과
            context: 추가 컨텍스트

        Returns:
            실행 계획
        """
        context = context or {}
        steps = []
        step_counter = 0

        def next_step_id():
            nonlocal step_counter
            step_counter += 1
            return f"step_{step_counter}"

        # 분석 결과에 따른 계획 수립
        if analysis:
            steps = self._plan_from_analysis(query, analysis, next_step_id)
        else:
            steps = self._plan_default(query, next_step_id)

        # 실행 순서 결정
        execution_order = self._determine_execution_order(steps, analysis)

        # 융합 전략 결정
        fusion_strategy = self._determine_fusion_strategy(analysis)

        # 예상 지연 시간 계산
        estimated_latency = self._estimate_latency(steps, execution_order)

        return QueryPlan(
            original_query=query,
            steps=steps,
            execution_order=execution_order,
            fusion_strategy=fusion_strategy,
            estimated_latency_ms=estimated_latency,
            metadata={
                "analysis": analysis.__dict__ if analysis else None,
                "context": context
            }
        )

    def _plan_from_analysis(
        self,
        query: str,
        analysis: QueryAnalysis,
        next_step_id
    ) -> List[SearchStep]:
        """분석 기반 계획 수립"""
        steps = []

        # 1. 벡터 검색 (의미적 유사성)
        if self.config.enable_vector:
            vector_step = self._create_vector_step(query, analysis, next_step_id())
            steps.append(vector_step)

        # 2. 그래프 검색 (엔티티 관계)
        if self.config.enable_graph and analysis.entities:
            graph_step = self._create_graph_step(query, analysis, next_step_id())
            steps.append(graph_step)

        # 3. 키워드 검색 (정확한 매칭)
        if self.config.enable_keyword and analysis.keywords:
            keyword_step = self._create_keyword_step(query, analysis, next_step_id())
            steps.append(keyword_step)

        # 4. 메타데이터 검색 (필터링)
        if self.config.enable_metadata and analysis.temporal_references:
            metadata_step = self._create_metadata_step(query, analysis, next_step_id())
            steps.append(metadata_step)

        return steps

    def _plan_default(self, query: str, next_step_id) -> List[SearchStep]:
        """기본 계획 수립"""
        steps = []

        # 벡터 검색 (주요)
        if self.config.enable_vector:
            steps.append(SearchStep(
                source=SearchSource.VECTOR,
                query=query,
                weight=self.config.vector_weight,
                top_k=self.config.default_top_k,
                step_id=next_step_id()
            ))

        # 키워드 검색 (보조)
        if self.config.enable_keyword:
            steps.append(SearchStep(
                source=SearchSource.KEYWORD,
                query=query,
                weight=self.config.keyword_weight,
                top_k=self.config.default_top_k,
                step_id=next_step_id()
            ))

        return steps

    def _create_vector_step(
        self,
        query: str,
        analysis: QueryAnalysis,
        step_id: str
    ) -> SearchStep:
        """벡터 검색 단계 생성"""
        # 복잡도에 따른 top_k 조정
        top_k = self.config.default_top_k
        if analysis.complexity == "complex":
            top_k = int(top_k * 1.5)
        elif analysis.complexity == "simple":
            top_k = int(top_k * 0.8)

        # 가중치 조정
        weight = self.config.vector_weight
        if analysis.intent in ["definition_inquiry", "general_inquiry"]:
            weight *= 1.2  # 의미 검색 강화

        return SearchStep(
            source=SearchSource.VECTOR,
            query=query,
            weight=min(weight, 1.0),
            top_k=max(top_k, 5),
            filters={},
            step_id=step_id
        )

    def _create_graph_step(
        self,
        query: str,
        analysis: QueryAnalysis,
        step_id: str
    ) -> SearchStep:
        """그래프 검색 단계 생성"""
        # 엔티티 기반 필터
        filters = {}
        if analysis.entities:
            filters["entities"] = analysis.entities

        # 가중치 조정
        weight = self.config.graph_weight
        if analysis.intent in ["person_inquiry", "location_inquiry"]:
            weight *= 1.5  # 관계 검색 강화

        return SearchStep(
            source=SearchSource.GRAPH,
            query=query,
            weight=min(weight, 1.0),
            top_k=self.config.default_top_k,
            filters=filters,
            step_id=step_id
        )

    def _create_keyword_step(
        self,
        query: str,
        analysis: QueryAnalysis,
        step_id: str
    ) -> SearchStep:
        """키워드 검색 단계 생성"""
        # 키워드 필터
        filters = {"keywords": analysis.keywords}

        # 부정 표현이 있으면 제외 키워드 추가
        if analysis.negations:
            filters["exclude"] = analysis.negations

        return SearchStep(
            source=SearchSource.KEYWORD,
            query=" ".join(analysis.keywords),
            weight=self.config.keyword_weight,
            top_k=self.config.default_top_k,
            filters=filters,
            step_id=step_id
        )

    def _create_metadata_step(
        self,
        query: str,
        analysis: QueryAnalysis,
        step_id: str
    ) -> SearchStep:
        """메타데이터 검색 단계 생성"""
        filters = {}

        # 시간 필터
        if analysis.temporal_references:
            filters["temporal"] = analysis.temporal_references

        return SearchStep(
            source=SearchSource.METADATA,
            query=query,
            weight=0.2,
            top_k=self.config.default_top_k,
            filters=filters,
            step_id=step_id
        )

    def _determine_execution_order(
        self,
        steps: List[SearchStep],
        analysis: Optional[QueryAnalysis]
    ) -> ExecutionOrder:
        """실행 순서 결정"""
        # 의존성이 있는 단계가 있으면 순차
        has_dependency = any(s.depends_on for s in steps)
        if has_dependency:
            return ExecutionOrder.SEQUENTIAL

        # 단계가 적으면 병렬
        if len(steps) <= self.config.max_parallel_steps:
            return ExecutionOrder.PARALLEL

        # 복잡한 쿼리는 조건부
        if analysis and analysis.complexity == "complex":
            return ExecutionOrder.CONDITIONAL

        return ExecutionOrder.PARALLEL

    def _determine_fusion_strategy(
        self,
        analysis: Optional[QueryAnalysis]
    ) -> str:
        """융합 전략 결정"""
        if not analysis:
            return "rrf"  # Reciprocal Rank Fusion

        # 의도에 따른 전략
        if analysis.intent in ["listing", "comparison"]:
            return "union"  # 합집합
        elif analysis.intent == "definition_inquiry":
            return "weighted"  # 가중 합산
        elif analysis.complexity == "complex":
            return "cascade"  # 단계적 필터링

        return "rrf"

    def _estimate_latency(
        self,
        steps: List[SearchStep],
        execution_order: ExecutionOrder
    ) -> float:
        """예상 지연 시간 계산"""
        # 소스별 예상 지연 시간 (ms)
        latency_estimates = {
            SearchSource.VECTOR: 50,
            SearchSource.GRAPH: 30,
            SearchSource.KEYWORD: 20,
            SearchSource.METADATA: 10,
        }

        step_latencies = [
            latency_estimates.get(step.source, 50)
            for step in steps
        ]

        if execution_order == ExecutionOrder.PARALLEL:
            return max(step_latencies) if step_latencies else 0
        elif execution_order == ExecutionOrder.SEQUENTIAL:
            return sum(step_latencies)
        else:  # CONDITIONAL
            return sum(step_latencies) * 0.7  # 일부만 실행 가정

    def optimize(self, plan: QueryPlan) -> QueryPlan:
        """계획 최적화"""
        optimized_steps = []

        for step in plan.steps:
            # 중복 제거
            is_duplicate = any(
                s.source == step.source and s.query == step.query
                for s in optimized_steps
            )
            if is_duplicate:
                continue

            # 낮은 가중치 단계 제거
            if step.weight < 0.05:
                continue

            optimized_steps.append(step)

        # 가중치 정규화
        total_weight = sum(s.weight for s in optimized_steps)
        if total_weight > 0:
            for step in optimized_steps:
                step.weight /= total_weight

        return QueryPlan(
            original_query=plan.original_query,
            steps=optimized_steps,
            execution_order=plan.execution_order,
            fusion_strategy=plan.fusion_strategy,
            estimated_latency_ms=self._estimate_latency(
                optimized_steps, plan.execution_order
            ),
            metadata=plan.metadata
        )
