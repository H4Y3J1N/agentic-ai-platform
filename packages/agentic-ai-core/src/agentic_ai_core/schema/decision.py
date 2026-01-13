"""
Decision Schema

의사결정 타입 및 영향도 매핑 스키마 정의
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import Field

from .base import IdentifiableTimestamped, Scorable, SchemaBase
from .document import DocumentType
from .entity import EntityType


class DecisionFrequency(str, Enum):
    """의사결정 빈도"""
    DAILY = "daily"              # 매일 발생
    WEEKLY = "weekly"            # 매주
    MONTHLY = "monthly"          # 매월
    QUARTERLY = "quarterly"      # 분기별
    YEARLY = "yearly"            # 연간
    RARE = "rare"                # 드물게


class ImpactLevel(str, Enum):
    """의사결정 영향 수준"""
    LOW = "low"                  # 낮음 (개인 수준)
    MEDIUM = "medium"            # 중간 (팀 수준)
    HIGH = "high"                # 높음 (부서 수준)
    CRITICAL = "critical"        # 중요 (조직 수준)


class InfluenceType(str, Enum):
    """데이터가 의사결정에 미치는 영향 유형"""
    DIRECT = "direct"            # 직접적 영향 (정책 → 의사결정)
    INDIRECT = "indirect"        # 간접적 영향 (참조 문서)
    CONTEXTUAL = "contextual"    # 맥락 제공


class DecisionType(IdentifiableTimestamped):
    """의사결정 유형 정의"""

    # 기본 정보
    name: str
    description: str = ""
    category: Optional[str] = None

    # 특성
    frequency: DecisionFrequency = DecisionFrequency.WEEKLY
    impact_level: ImpactLevel = ImpactLevel.MEDIUM

    # 관련 데이터 타입
    relevant_entity_types: List[EntityType] = Field(default_factory=list)
    relevant_document_types: List[DocumentType] = Field(default_factory=list)

    # 키워드 (검색/매칭용)
    keywords: List[str] = Field(default_factory=list)

    # 학습된 가중치 (data_point_id → weight)
    learned_weights: Dict[str, float] = Field(default_factory=dict)

    # 통계
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

    def update_weight(self, data_id: str, weight: float) -> None:
        """데이터 포인트 가중치 업데이트"""
        self.learned_weights[data_id] = max(0.0, min(1.0, weight))

    def get_weight(self, data_id: str) -> float:
        """데이터 포인트 가중치 조회"""
        return self.learned_weights.get(data_id, 0.0)

    def record_query(self, successful: bool = True) -> None:
        """쿼리 통계 기록"""
        self.total_queries += 1
        if successful:
            self.successful_queries += 1

    @property
    def success_rate(self) -> float:
        """쿼리 성공률"""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries


class DecisionMapping(IdentifiableTimestamped, Scorable):
    """문서/엔티티 → 의사결정 매핑"""

    # 매핑 대상
    decision_type_id: str
    target_type: str = "document"  # "document" | "entity" | "chunk"
    target_id: str

    # 영향도
    influence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    influence_type: InfluenceType = InfluenceType.INDIRECT

    # 학습 메트릭
    query_hit_count: int = 0        # 해당 의사결정 쿼리에서 검색된 횟수
    feedback_positive: int = 0      # 긍정 피드백
    feedback_negative: int = 0      # 부정 피드백
    last_hit_at: Optional[datetime] = None

    # 계산 근거
    reasoning: str = ""
    auto_generated: bool = True     # 자동 생성 여부

    @property
    def feedback_score(self) -> float:
        """피드백 기반 점수 (0~1)"""
        total = self.feedback_positive + self.feedback_negative
        if total == 0:
            return 0.5  # 기본값
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
        # 영향도 점수 업데이트
        self._update_influence_from_feedback()

    def _update_influence_from_feedback(self) -> None:
        """피드백 기반 영향도 업데이트"""
        # 현재 점수와 피드백 점수의 가중 평균
        feedback_weight = 0.3
        self.influence_score = (
            self.influence_score * (1 - feedback_weight) +
            self.feedback_score * feedback_weight
        )

    @classmethod
    def create_for_document(
        cls,
        document_id: str,
        decision_type_id: str,
        influence_score: float,
        influence_type: InfluenceType = InfluenceType.INDIRECT,
        reasoning: str = "",
        **kwargs
    ) -> "DecisionMapping":
        """문서용 매핑 생성"""
        return cls(
            target_type="document",
            target_id=document_id,
            decision_type_id=decision_type_id,
            influence_score=influence_score,
            influence_type=influence_type,
            reasoning=reasoning,
            **kwargs
        )

    @classmethod
    def create_for_entity(
        cls,
        entity_id: str,
        decision_type_id: str,
        influence_score: float,
        influence_type: InfluenceType = InfluenceType.INDIRECT,
        reasoning: str = "",
        **kwargs
    ) -> "DecisionMapping":
        """엔티티용 매핑 생성"""
        return cls(
            target_type="entity",
            target_id=entity_id,
            decision_type_id=decision_type_id,
            influence_score=influence_score,
            influence_type=influence_type,
            reasoning=reasoning,
            **kwargs
        )


class DecisionContext(SchemaBase):
    """의사결정 컨텍스트 (검색 시 사용)"""

    decision_type: Optional[DecisionType] = None
    user_role: Optional[str] = None
    department: Optional[str] = None

    # 부스트 설정
    recency_boost: float = 1.0      # 최신성 가중치
    authority_boost: float = 1.0    # 권위 가중치 (출처 신뢰도)

    def get_boost_factor(self, mapping: DecisionMapping) -> float:
        """매핑에 대한 부스트 계수 계산"""
        base_boost = 1.0

        # 직접 영향은 더 높은 부스트
        if mapping.influence_type == InfluenceType.DIRECT:
            base_boost *= 1.5
        elif mapping.influence_type == InfluenceType.CONTEXTUAL:
            base_boost *= 0.8

        # 피드백 기반 조정
        base_boost *= (0.5 + mapping.feedback_score)

        return base_boost


# 기본 의사결정 타입 정의 (도메인별 확장 가능)
DEFAULT_DECISION_TYPES = [
    DecisionType(
        id="leave_request",
        name="휴가 신청",
        description="휴가 신청 및 승인 관련 의사결정",
        category="hr",
        frequency=DecisionFrequency.DAILY,
        impact_level=ImpactLevel.LOW,
        relevant_entity_types=[EntityType.PERSON, EntityType.POLICY, EntityType.DEPARTMENT],
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
        relevant_entity_types=[EntityType.PERSON, EntityType.DEPARTMENT, EntityType.PROCESS],
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
        relevant_entity_types=[EntityType.PERSON, EntityType.POLICY, EntityType.PROJECT],
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
        relevant_entity_types=[EntityType.PROJECT, EntityType.PERSON, EntityType.TEAM],
        relevant_document_types=[DocumentType.TECHNICAL_DOC, DocumentType.MEETING_NOTE, DocumentType.REPORT],
        keywords=["프로젝트", "계획", "일정", "리소스", "project", "planning", "timeline"]
    ),
]
