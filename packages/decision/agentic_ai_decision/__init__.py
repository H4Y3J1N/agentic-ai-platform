"""
Agentic AI Decision Package

의사결정 지원 확장 모듈 - Decision Types, Decision Scoring, Lifecycle Management

사용 시점:
- 의사결정 영향도를 계산해야 하는 도메인
- 문서의 의사결정 관련성을 스코어링할 때
- 데이터 생명주기 관리가 필요한 경우

단순 RAG/지식그래프만 필요하면 agentic-ai-core + agentic-ai-knowledge를 사용하세요.
"""

# Schema
from .schema import (
    # Decision Types
    DecisionFrequency,
    ImpactLevel,
    InfluenceType,
    DecisionType,
    DecisionMapping,
    DecisionContext,
    DEFAULT_DECISION_TYPES,
)

# Scoring
from .scoring import (
    DecisionScorer,
    DecisionScore,
    DocumentDecisionProfile,
    DecisionScorerConfig,
    BatchDecisionScorer,
)

# Lifecycle
from .lifecycle import (
    LifecycleManager,
    BatchLifecycleManager,
    LifecycleState,
    ResolutionLevel,
    StorageTier,
    LifecyclePolicy,
    LifecycleTransition,
    DocumentLifecycle,
    AccessPattern,
    LifecycleScheduler,
    ScheduledTask,
    SchedulerConfig,
)


__all__ = [
    # Decision Schema
    "DecisionFrequency",
    "ImpactLevel",
    "InfluenceType",
    "DecisionType",
    "DecisionMapping",
    "DecisionContext",
    "DEFAULT_DECISION_TYPES",
    # Scoring
    "DecisionScorer",
    "DecisionScore",
    "DocumentDecisionProfile",
    "DecisionScorerConfig",
    "BatchDecisionScorer",
    # Lifecycle
    "LifecycleManager",
    "BatchLifecycleManager",
    "LifecycleState",
    "ResolutionLevel",
    "StorageTier",
    "LifecyclePolicy",
    "LifecycleTransition",
    "DocumentLifecycle",
    "AccessPattern",
    "LifecycleScheduler",
    "ScheduledTask",
    "SchedulerConfig",
]
