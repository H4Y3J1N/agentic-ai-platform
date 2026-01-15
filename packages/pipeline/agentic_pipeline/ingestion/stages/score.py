"""
Score Stage

의사결정 관련성 스코어링 스테이지
"""

from typing import Optional, List, Dict, Any
import logging

from .base import Stage
from ..context import PipelineContext
from ...schema import (
    DecisionType,
    DecisionMapping,
    InfluenceType,
    DocumentType,
    EntityType,
    DEFAULT_DECISION_TYPES,
)

logger = logging.getLogger(__name__)


class ScoreStage(Stage):
    """의사결정 관련성 스코어링 스테이지"""

    def __init__(
        self,
        decision_types: Optional[List[DecisionType]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__("ScoreStage")
        self.decision_types = decision_types or DEFAULT_DECISION_TYPES
        self.weights = weights or DEFAULT_WEIGHTS

    async def process(self, context: PipelineContext) -> PipelineContext:
        """스코어링 실행"""
        content = context.raw_content or ""
        if not content:
            return context

        mappings = []
        max_score = 0.0

        for decision_type in self.decision_types:
            score = self._calculate_relevance(context, decision_type)

            if score > 0.1:  # 최소 임계값
                influence_type = self._determine_influence_type(score)

                mapping = DecisionMapping.create_for_document(
                    document_id=context.source_item.id,
                    decision_type_id=decision_type.id,
                    influence_score=score,
                    influence_type=influence_type,
                    reasoning=self._generate_reasoning(context, decision_type, score)
                )
                mappings.append(mapping)

                if score > max_score:
                    max_score = score

        context.decision_mappings = mappings
        context.overall_relevance = max_score

        logger.debug(
            f"Scored document: {len(mappings)} decision mappings, "
            f"max_score={max_score:.2f}"
        )

        return context

    def _calculate_relevance(
        self,
        context: PipelineContext,
        decision_type: DecisionType
    ) -> float:
        """
        의사결정 관련성 점수 계산

        Score = Σ(weight_i × factor_i) / Σ(weight_i)
        """
        factors = {}

        # 1. 문서 타입 매칭
        doc_type_match = 0.0
        if context.document_type and decision_type.relevant_document_types:
            if context.document_type in decision_type.relevant_document_types:
                doc_type_match = 1.0
        factors["document_type_match"] = doc_type_match

        # 2. 엔티티 타입 오버랩
        entity_overlap = self._calculate_entity_overlap(context, decision_type)
        factors["entity_overlap"] = entity_overlap

        # 3. 키워드 매칭
        keyword_match = self._calculate_keyword_match(context, decision_type)
        factors["keyword_match"] = keyword_match

        # 4. 토픽 매칭
        topic_match = self._calculate_topic_match(context, decision_type)
        factors["topic_match"] = topic_match

        # 5. 최신성 (현재는 1.0으로 고정)
        factors["freshness"] = 1.0

        # 가중 평균 계산
        weighted_sum = sum(
            self.weights.get(k, 0.1) * v
            for k, v in factors.items()
        )
        total_weight = sum(
            self.weights.get(k, 0.1)
            for k in factors
        )

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_entity_overlap(
        self,
        context: PipelineContext,
        decision_type: DecisionType
    ) -> float:
        """엔티티 타입 오버랩 계산"""
        if not context.entities or not decision_type.relevant_entity_types:
            return 0.0

        doc_entity_types = {e.entity_type for e in context.entities}
        relevant_types = set(decision_type.relevant_entity_types)

        intersection = len(doc_entity_types & relevant_types)
        union = len(doc_entity_types | relevant_types)

        return intersection / union if union > 0 else 0.0

    def _calculate_keyword_match(
        self,
        context: PipelineContext,
        decision_type: DecisionType
    ) -> float:
        """키워드 매칭 점수"""
        if not decision_type.keywords:
            return 0.0

        content_lower = (context.raw_content or "").lower()

        matched = sum(
            1 for kw in decision_type.keywords
            if kw.lower() in content_lower
        )

        return min(matched / len(decision_type.keywords), 1.0)

    def _calculate_topic_match(
        self,
        context: PipelineContext,
        decision_type: DecisionType
    ) -> float:
        """토픽 매칭 점수"""
        doc_topics = context.inferred_metadata.get("topics", [])
        if not doc_topics:
            return 0.0

        doc_topics_lower = {t.lower() for t in doc_topics}
        decision_keywords = {kw.lower() for kw in decision_type.keywords}

        matched = len(doc_topics_lower & decision_keywords)
        return min(matched / max(len(doc_topics), 1), 1.0)

    def _determine_influence_type(self, score: float) -> InfluenceType:
        """영향 유형 결정"""
        if score >= 0.7:
            return InfluenceType.DIRECT
        elif score >= 0.4:
            return InfluenceType.INDIRECT
        else:
            return InfluenceType.CONTEXTUAL

    def _generate_reasoning(
        self,
        context: PipelineContext,
        decision_type: DecisionType,
        score: float
    ) -> str:
        """스코어링 근거 생성"""
        reasons = []

        if context.document_type in decision_type.relevant_document_types:
            reasons.append(f"문서 타입({context.document_type.value})이 관련됨")

        entity_types = [e.entity_type.value for e in context.entities[:3]]
        if entity_types:
            reasons.append(f"관련 엔티티: {', '.join(entity_types)}")

        matched_keywords = [
            kw for kw in decision_type.keywords
            if kw.lower() in (context.raw_content or "").lower()
        ][:3]
        if matched_keywords:
            reasons.append(f"키워드 매칭: {', '.join(matched_keywords)}")

        return "; ".join(reasons) if reasons else "일반 관련성"


# 기본 가중치
DEFAULT_WEIGHTS: Dict[str, float] = {
    "document_type_match": 0.3,
    "entity_overlap": 0.2,
    "keyword_match": 0.25,
    "topic_match": 0.15,
    "freshness": 0.1,
}
