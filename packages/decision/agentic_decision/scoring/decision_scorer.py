"""
Decision Scorer

의사결정 영향도 계산
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from ..schema import (
    Document,
    Entity,
    Relationship,
    DecisionType,
    DecisionMapping,
    DocumentType,
    EntityType,
)

logger = logging.getLogger(__name__)


@dataclass
class DecisionScore:
    """의사결정 영향 점수"""
    decision_type: str
    score: float  # 0.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    factors: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class DocumentDecisionProfile:
    """문서의 의사결정 프로파일"""
    document_id: str
    scores: List[DecisionScore]
    primary_decision: Optional[str] = None
    total_influence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionScorerConfig:
    """스코어러 설정"""
    # 가중치
    content_weight: float = 0.4
    entity_weight: float = 0.25
    metadata_weight: float = 0.2
    freshness_weight: float = 0.15

    # 임계값
    min_score_threshold: float = 0.1
    high_influence_threshold: float = 0.7

    # 시간 감쇠
    freshness_decay_days: int = 90

    # 기본 의사결정 타입
    default_decision_types: Optional[List[DecisionType]] = None


class DecisionScorer:
    """의사결정 영향도 스코어러"""

    def __init__(self, config: Optional[DecisionScorerConfig] = None):
        self.config = config or DecisionScorerConfig()
        self.decision_types = (
            self.config.default_decision_types or
            self._get_default_decision_types()
        )

        # 문서 타입 → 의사결정 매핑
        self._doc_type_mappings = self._build_doc_type_mappings()

        # 엔티티 타입 → 의사결정 매핑
        self._entity_type_mappings = self._build_entity_type_mappings()

        # 키워드 → 의사결정 매핑
        self._keyword_mappings = self._build_keyword_mappings()

    def score_document(
        self,
        content: str,
        doc_type: Optional[DocumentType] = None,
        entities: Optional[List[Entity]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None
    ) -> DocumentDecisionProfile:
        """
        문서의 의사결정 영향도 계산

        Args:
            content: 문서 내용
            doc_type: 문서 타입
            entities: 추출된 엔티티
            metadata: 메타데이터
            created_at: 생성 시간

        Returns:
            문서 의사결정 프로파일
        """
        entities = entities or []
        metadata = metadata or {}

        scores = []

        for decision_type in self.decision_types:
            score = self._calculate_decision_score(
                decision_type=decision_type,
                content=content,
                doc_type=doc_type,
                entities=entities,
                metadata=metadata,
                created_at=created_at
            )

            if score.score >= self.config.min_score_threshold:
                scores.append(score)

        # 점수 기준 정렬
        scores.sort(key=lambda x: x.score, reverse=True)

        # 주요 의사결정 결정
        primary = scores[0].decision_type if scores else None

        # 총 영향도
        total_influence = sum(s.score * s.confidence for s in scores)

        return DocumentDecisionProfile(
            document_id=metadata.get("document_id", ""),
            scores=scores,
            primary_decision=primary,
            total_influence=min(total_influence, 1.0),
            metadata={
                "scored_at": datetime.now().isoformat(),
                "decision_count": len(scores),
                "doc_type": doc_type.value if doc_type else None
            }
        )

    def _calculate_decision_score(
        self,
        decision_type: DecisionType,
        content: str,
        doc_type: Optional[DocumentType],
        entities: List[Entity],
        metadata: Dict[str, Any],
        created_at: Optional[datetime]
    ) -> DecisionScore:
        """개별 의사결정 점수 계산"""
        factors = {}

        # 1. 콘텐츠 기반 점수
        content_score = self._score_content(content, decision_type)
        factors["content"] = content_score

        # 2. 문서 타입 기반 점수
        doc_type_score = self._score_doc_type(doc_type, decision_type)
        factors["doc_type"] = doc_type_score

        # 3. 엔티티 기반 점수
        entity_score = self._score_entities(entities, decision_type)
        factors["entities"] = entity_score

        # 4. 메타데이터 기반 점수
        metadata_score = self._score_metadata(metadata, decision_type)
        factors["metadata"] = metadata_score

        # 5. 신선도 점수
        freshness_score = self._score_freshness(created_at)
        factors["freshness"] = freshness_score

        # 가중 합산
        weighted_score = (
            self.config.content_weight * content_score +
            self.config.entity_weight * entity_score +
            self.config.metadata_weight * (doc_type_score + metadata_score) / 2 +
            self.config.freshness_weight * freshness_score
        )

        # 신뢰도 계산
        confidence = self._calculate_confidence(factors)

        # 설명 생성
        explanation = self._generate_explanation(decision_type, factors)

        return DecisionScore(
            decision_type=decision_type.name,
            score=min(max(weighted_score, 0.0), 1.0),
            confidence=confidence,
            factors=factors,
            explanation=explanation
        )

    def _score_content(
        self,
        content: str,
        decision_type: DecisionType
    ) -> float:
        """콘텐츠 기반 점수"""
        if not content:
            return 0.0

        content_lower = content.lower()
        keywords = self._keyword_mappings.get(decision_type.name, [])

        if not keywords:
            return 0.0

        # 키워드 매칭
        matches = sum(1 for kw in keywords if kw.lower() in content_lower)

        # 정규화 (최대 점수 1.0)
        return min(matches / max(len(keywords) * 0.3, 1), 1.0)

    def _score_doc_type(
        self,
        doc_type: Optional[DocumentType],
        decision_type: DecisionType
    ) -> float:
        """문서 타입 기반 점수"""
        if not doc_type:
            return 0.3  # 기본값

        relevant_types = self._doc_type_mappings.get(decision_type.name, [])

        if doc_type in relevant_types:
            return 1.0

        return 0.2

    def _score_entities(
        self,
        entities: List[Entity],
        decision_type: DecisionType
    ) -> float:
        """엔티티 기반 점수"""
        if not entities:
            return 0.0

        relevant_types = self._entity_type_mappings.get(decision_type.name, [])

        if not relevant_types:
            return 0.3

        # 관련 엔티티 수
        relevant_count = sum(
            1 for e in entities
            if e.entity_type in relevant_types
        )

        # 정규화
        return min(relevant_count / max(len(entities) * 0.5, 1), 1.0)

    def _score_metadata(
        self,
        metadata: Dict[str, Any],
        decision_type: DecisionType
    ) -> float:
        """메타데이터 기반 점수"""
        score = 0.0

        # 태그/카테고리 체크
        tags = metadata.get("tags", [])
        categories = metadata.get("categories", [])

        decision_keywords = self._keyword_mappings.get(decision_type.name, [])

        for tag in tags + categories:
            if any(kw.lower() in tag.lower() for kw in decision_keywords):
                score += 0.3

        # 우선순위 체크
        priority = metadata.get("priority", "")
        if priority in ["high", "urgent", "긴급", "중요"]:
            score += 0.2

        return min(score, 1.0)

    def _score_freshness(
        self,
        created_at: Optional[datetime]
    ) -> float:
        """신선도 점수 (시간 감쇠)"""
        if not created_at:
            return 0.5  # 기본값

        age_days = (datetime.now() - created_at).days

        # 지수 감쇠
        decay_rate = age_days / self.config.freshness_decay_days
        freshness = max(0.0, 1.0 - decay_rate)

        return freshness

    def _calculate_confidence(self, factors: Dict[str, float]) -> float:
        """신뢰도 계산"""
        # 팩터 분산 기반 신뢰도
        values = list(factors.values())
        if not values:
            return 0.5

        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)

        # 분산이 낮을수록 신뢰도 높음
        confidence = max(0.3, 1.0 - variance)

        # 0이 아닌 팩터 수에 따른 보정
        non_zero = sum(1 for v in values if v > 0)
        coverage = non_zero / len(values)

        return confidence * (0.5 + 0.5 * coverage)

    def _generate_explanation(
        self,
        decision_type: DecisionType,
        factors: Dict[str, float]
    ) -> str:
        """점수 설명 생성"""
        top_factors = sorted(
            factors.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        factor_strs = [
            f"{name}({score:.2f})"
            for name, score in top_factors
            if score > 0
        ]

        return f"{decision_type.name}: {', '.join(factor_strs)}"

    def create_decision_mapping(
        self,
        profile: DocumentDecisionProfile,
        document_id: str
    ) -> List[DecisionMapping]:
        """의사결정 매핑 생성"""
        mappings = []

        for score in profile.scores:
            if score.score >= self.config.min_score_threshold:
                mapping = DecisionMapping(
                    decision_type=score.decision_type,
                    document_id=document_id,
                    influence_score=score.score,
                    confidence=score.confidence,
                    factors=score.factors
                )
                mappings.append(mapping)

        return mappings

    # ==================
    # 매핑 빌더
    # ==================

    def _get_default_decision_types(self) -> List[DecisionType]:
        """기본 의사결정 타입"""
        from ..schema.decision import DEFAULT_DECISION_TYPES
        return DEFAULT_DECISION_TYPES

    def _build_doc_type_mappings(self) -> Dict[str, List[DocumentType]]:
        """문서 타입 → 의사결정 매핑"""
        return {
            "leave_request": [DocumentType.POLICY, DocumentType.FAQ],
            "expense_approval": [DocumentType.POLICY, DocumentType.FAQ],
            "onboarding": [
                DocumentType.POLICY,
                DocumentType.WIKI,
                DocumentType.FAQ,
                DocumentType.TECHNICAL_DOC
            ],
            "project_planning": [
                DocumentType.MEETING_NOTE,
                DocumentType.TECHNICAL_DOC,
                DocumentType.WIKI
            ],
            "hr_inquiry": [DocumentType.POLICY, DocumentType.FAQ],
            "technical_support": [
                DocumentType.TECHNICAL_DOC,
                DocumentType.FAQ,
                DocumentType.WIKI
            ],
        }

    def _build_entity_type_mappings(self) -> Dict[str, List[EntityType]]:
        """엔티티 타입 → 의사결정 매핑"""
        return {
            "leave_request": [EntityType.PERSON, EntityType.POLICY],
            "expense_approval": [EntityType.PERSON, EntityType.PROJECT, EntityType.POLICY],
            "onboarding": [
                EntityType.PERSON,
                EntityType.TEAM,
                EntityType.TOOL,
                EntityType.PROCESS
            ],
            "project_planning": [
                EntityType.PROJECT,
                EntityType.PERSON,
                EntityType.TEAM,
                EntityType.TASK
            ],
            "hr_inquiry": [EntityType.PERSON, EntityType.POLICY, EntityType.TEAM],
            "technical_support": [EntityType.TOOL, EntityType.PROCESS, EntityType.CONCEPT],
        }

    def _build_keyword_mappings(self) -> Dict[str, List[str]]:
        """키워드 → 의사결정 매핑"""
        return {
            "leave_request": [
                "휴가", "연차", "병가", "육아휴직", "leave", "vacation",
                "PTO", "time off", "absent", "휴일", "신청"
            ],
            "expense_approval": [
                "비용", "경비", "청구", "expense", "reimbursement",
                "영수증", "receipt", "payment", "승인", "결재"
            ],
            "onboarding": [
                "온보딩", "신입", "입사", "onboarding", "new hire",
                "orientation", "시작", "가이드", "setup", "환경설정"
            ],
            "project_planning": [
                "프로젝트", "계획", "일정", "마일스톤", "project",
                "planning", "roadmap", "sprint", "deadline", "목표"
            ],
            "hr_inquiry": [
                "인사", "HR", "급여", "복지", "benefits", "salary",
                "평가", "승진", "promotion", "계약"
            ],
            "technical_support": [
                "기술", "버그", "오류", "error", "technical",
                "support", "도움", "help", "문제", "troubleshoot"
            ],
        }


class BatchDecisionScorer:
    """배치 의사결정 스코어러"""

    def __init__(self, scorer: Optional[DecisionScorer] = None):
        self.scorer = scorer or DecisionScorer()

    def score_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[DocumentDecisionProfile]:
        """
        문서 배치 스코어링

        Args:
            documents: 문서 리스트 (content, doc_type, entities, metadata 포함)

        Returns:
            프로파일 리스트
        """
        profiles = []

        for doc in documents:
            profile = self.scorer.score_document(
                content=doc.get("content", ""),
                doc_type=doc.get("doc_type"),
                entities=doc.get("entities", []),
                metadata=doc.get("metadata", {}),
                created_at=doc.get("created_at")
            )
            profiles.append(profile)

        return profiles

    def get_high_influence_documents(
        self,
        profiles: List[DocumentDecisionProfile],
        decision_type: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[DocumentDecisionProfile]:
        """고영향 문서 필터링"""
        min_score = min_score or self.scorer.config.high_influence_threshold

        filtered = []
        for profile in profiles:
            for score in profile.scores:
                if decision_type and score.decision_type != decision_type:
                    continue
                if score.score >= min_score:
                    filtered.append(profile)
                    break

        return filtered
