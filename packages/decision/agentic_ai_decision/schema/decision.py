"""
Decision Schema

의사결정 타입 및 영향도 매핑 스키마 정의

사용 시점:
- 의사결정 유형을 정의할 때 (휴가 신청, 비용 승인 등)
- 데이터가 의사결정에 미치는 영향도를 매핑할 때
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import Field

from agentic_ai_core.schema import (
    IdentifiableTimestamped,
    Scorable,
    SchemaBase,
    DocumentType,
)

# Optional dependency on knowledge package
try:
    from agentic_ai_knowledge import EntityType
except ImportError:
    # Fallback - define locally if knowledge package not installed
    from enum import Enum as EntityType
    class EntityType(str, Enum):
        PERSON = "person"
        TEAM = "team"
        PROJECT = "project"
        POLICY = "policy"


class DecisionFrequency(str, Enum):
    """의사결정 빈도"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    RARE = "rare"


class ImpactLevel(str, Enum):
    """의사결정 영향 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InfluenceType(str, Enum):
    """데이터가 의사결정에 미치는 영향 유형"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONTEXTUAL = "contextual"


class DecisionType(IdentifiableTimestamped):
    """의사결정 유형 정의"""

    name: str
    description: str = ""
    category: Optional[str] = None

    frequency: DecisionFrequency = DecisionFrequency.WEEKLY
    impact_level: ImpactLevel = ImpactLevel.MEDIUM

    relevant_entity_types: List[str] = Field(default_factory=list)
    relevant_document_types: List[DocumentType] = Field(default_factory=list)

    keywords: List[str] = Field(default_factory=list)
    learned_weights: Dict[str, float] = Field(default_factory=dict)

    total_queries: int = 0
    successful_queries: int = 0

    def add_keyword(self, keyword: str) -> None:
        """키워드 추가"""
        normalized = keyword.lower().strip()
        if normalized and normalized not in self.keywords:
            self.keywords.append(normalized)

    def matches_query(self, query: str) -> bool:
        """쿼리가 이 의사결정 타입과 관련있는지"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.keywords)

    @property
    def success_rate(self) -> float:
        """쿼리 성공률"""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries


class DecisionMapping(IdentifiableTimestamped, Scorable):
    """문서/엔티티 → 의사결정 매핑"""

    decision_type_id: str
    target_type: str = "document"
    target_id: str

    influence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    influence_type: InfluenceType = InfluenceType.INDIRECT

    query_hit_count: int = 0
    feedback_positive: int = 0
    feedback_negative: int = 0
    last_hit_at: Optional[datetime] = None

    reasoning: str = ""
    auto_generated: bool = True

    @property
    def feedback_score(self) -> float:
        """피드백 기반 점수"""
        total = self.feedback_positive + self.feedback_negative
        if total == 0:
            return 0.5
        return self.feedback_positive / total

    def record_hit(self) -> None:
        """검색 히트 기록"""
        self.query_hit_count += 1
        self.last_hit_at = datetime.now()

    def record_feedback(self, positive: bool) -> None:
        """피드백 기록"""
        if positive:
            self.feedback_positive += 1
        else:
            self.feedback_negative += 1

    @classmethod
    def create_for_document(
        cls,
        document_id: str,
        decision_type_id: str,
        influence_score: float,
        influence_type: InfluenceType = InfluenceType.INDIRECT,
        **kwargs
    ) -> "DecisionMapping":
        """문서용 매핑 생성"""
        return cls(
            target_type="document",
            target_id=document_id,
            decision_type_id=decision_type_id,
            influence_score=influence_score,
            influence_type=influence_type,
            **kwargs
        )


class DecisionContext(SchemaBase):
    """의사결정 컨텍스트"""

    decision_type: Optional[DecisionType] = None
    user_role: Optional[str] = None
    department: Optional[str] = None

    recency_boost: float = 1.0
    authority_boost: float = 1.0

    def get_boost_factor(self, mapping: DecisionMapping) -> float:
        """매핑에 대한 부스트 계수 계산"""
        base_boost = 1.0

        if mapping.influence_type == InfluenceType.DIRECT:
            base_boost *= 1.5
        elif mapping.influence_type == InfluenceType.CONTEXTUAL:
            base_boost *= 0.8

        base_boost *= (0.5 + mapping.feedback_score)
        return base_boost


# 기본 의사결정 타입
DEFAULT_DECISION_TYPES = [
    DecisionType(
        id="leave_request",
        name="휴가 신청",
        description="휴가 신청 및 승인 관련 의사결정",
        category="hr",
        frequency=DecisionFrequency.DAILY,
        impact_level=ImpactLevel.LOW,
        relevant_entity_types=["person", "policy", "department"],
        relevant_document_types=[DocumentType.POLICY, DocumentType.FAQ],
        keywords=["휴가", "연차", "반차", "휴일", "leave", "vacation", "PTO"]
    ),
    DecisionType(
        id="onboarding",
        name="신규 입사자 온보딩",
        description="신규 입사자 온보딩 관련 의사결정",
        category="hr",
        frequency=DecisionFrequency.WEEKLY,
        impact_level=ImpactLevel.MEDIUM,
        relevant_entity_types=["person", "department", "process"],
        relevant_document_types=[DocumentType.POLICY, DocumentType.TECHNICAL_DOC, DocumentType.WIKI],
        keywords=["온보딩", "입사", "신규", "가입", "onboarding", "new hire"]
    ),
    DecisionType(
        id="expense_approval",
        name="비용 승인",
        description="경비 및 비용 승인 관련 의사결정",
        category="finance",
        frequency=DecisionFrequency.WEEKLY,
        impact_level=ImpactLevel.MEDIUM,
        relevant_entity_types=["person", "policy", "project"],
        relevant_document_types=[DocumentType.POLICY, DocumentType.REPORT],
        keywords=["비용", "경비", "승인", "expense", "budget", "approval"]
    ),
    DecisionType(
        id="project_planning",
        name="프로젝트 계획",
        description="프로젝트 계획 및 리소스 배분 의사결정",
        category="project",
        frequency=DecisionFrequency.MONTHLY,
        impact_level=ImpactLevel.HIGH,
        relevant_entity_types=["project", "person", "team"],
        relevant_document_types=[DocumentType.TECHNICAL_DOC, DocumentType.MEETING_NOTE, DocumentType.REPORT],
        keywords=["프로젝트", "계획", "일정", "리소스", "project", "planning", "timeline"]
    ),
]
