"""
ROI Calculator

벡터화 ROI(Return on Investment) 계산
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math
import logging

from ..schema import DocumentType

logger = logging.getLogger(__name__)


class VectorizationGranularity(Enum):
    """벡터화 세분화 수준"""
    NONE = "none"           # 벡터화 안 함
    METADATA = "metadata"   # 메타데이터만
    SUMMARY = "summary"     # 요약본만
    CHUNKS = "chunks"       # 청크 단위
    FULL = "full"          # 전체 문서


class StorageTier(Enum):
    """저장소 계층"""
    HOT = "hot"            # 자주 접근 (SSD/메모리)
    WARM = "warm"          # 가끔 접근 (SSD)
    COLD = "cold"          # 드물게 접근 (HDD/아카이브)


@dataclass
class ROIScore:
    """ROI 점수"""
    total_roi: float
    expected_value: float
    storage_cost: float
    query_probability: float
    granularity: VectorizationGranularity
    storage_tier: StorageTier
    factors: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class ROICalculatorConfig:
    """ROI 계산기 설정"""
    # 가치 계수
    base_query_value: float = 10.0  # 기본 쿼리 가치
    relevance_multiplier: float = 2.0  # 관련성 승수

    # 비용 계수 (청크당)
    embedding_cost_per_chunk: float = 0.001  # 임베딩 비용
    storage_cost_per_chunk: float = 0.0001  # 저장 비용 (월별)
    retrieval_cost_per_query: float = 0.0001  # 검색 비용

    # 확률 계수
    base_query_probability: float = 0.5
    type_probability_boost: Dict[str, float] = field(default_factory=dict)

    # 임계값
    vectorize_threshold: float = 1.0  # ROI > 1이면 벡터화
    high_value_threshold: float = 5.0  # ROI > 5면 고가치

    # 시간 감쇠
    value_decay_rate: float = 0.1  # 월별 가치 감소율

    def __post_init__(self):
        if not self.type_probability_boost:
            self.type_probability_boost = {
                "policy": 0.9,
                "faq": 0.95,
                "technical_doc": 0.7,
                "wiki": 0.6,
                "meeting_note": 0.4,
                "announcement": 0.3,
            }


class ROICalculator:
    """벡터화 ROI 계산기"""

    def __init__(self, config: Optional[ROICalculatorConfig] = None):
        self.config = config or ROICalculatorConfig()

    def calculate(
        self,
        content_length: int,
        doc_type: Optional[DocumentType] = None,
        relevance_score: float = 0.5,
        decision_influence: float = 0.5,
        created_at: Optional[datetime] = None,
        historical_queries: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ROIScore:
        """
        벡터화 ROI 계산

        ROI = (Expected_Value × Query_Probability) / Total_Cost

        Args:
            content_length: 콘텐츠 길이 (문자 수)
            doc_type: 문서 타입
            relevance_score: 관련성 점수 (0~1)
            decision_influence: 의사결정 영향도 (0~1)
            created_at: 생성 시간
            historical_queries: 과거 쿼리 횟수
            metadata: 추가 메타데이터

        Returns:
            ROI 점수
        """
        metadata = metadata or {}
        factors = {}

        # 1. 예상 가치 계산
        expected_value = self._calculate_expected_value(
            relevance_score=relevance_score,
            decision_influence=decision_influence,
            historical_queries=historical_queries,
            created_at=created_at
        )
        factors["expected_value"] = expected_value

        # 2. 쿼리 확률 계산
        query_probability = self._calculate_query_probability(
            doc_type=doc_type,
            relevance_score=relevance_score,
            historical_queries=historical_queries,
            metadata=metadata
        )
        factors["query_probability"] = query_probability

        # 3. 비용 계산
        storage_cost = self._calculate_storage_cost(
            content_length=content_length,
            created_at=created_at
        )
        factors["storage_cost"] = storage_cost

        # 4. ROI 계산
        if storage_cost > 0:
            total_roi = (expected_value * query_probability) / storage_cost
        else:
            total_roi = float('inf') if expected_value > 0 else 0

        # 5. 세분화 수준 결정
        granularity = self._determine_granularity(total_roi, content_length)

        # 6. 저장소 계층 결정
        storage_tier = self._determine_storage_tier(total_roi, query_probability)

        # 7. 권장사항 생성
        recommendation = self._generate_recommendation(
            total_roi=total_roi,
            granularity=granularity,
            storage_tier=storage_tier
        )

        return ROIScore(
            total_roi=total_roi,
            expected_value=expected_value,
            storage_cost=storage_cost,
            query_probability=query_probability,
            granularity=granularity,
            storage_tier=storage_tier,
            factors=factors,
            recommendation=recommendation
        )

    def _calculate_expected_value(
        self,
        relevance_score: float,
        decision_influence: float,
        historical_queries: int,
        created_at: Optional[datetime]
    ) -> float:
        """예상 가치 계산"""
        # 기본 가치
        base_value = self.config.base_query_value

        # 관련성 보정
        relevance_factor = 1 + (relevance_score * self.config.relevance_multiplier)

        # 의사결정 영향도 보정
        decision_factor = 1 + decision_influence

        # 과거 쿼리 기반 보정
        history_factor = 1 + math.log10(historical_queries + 1) * 0.2

        # 시간 감쇠
        time_factor = 1.0
        if created_at:
            months_old = (datetime.now() - created_at).days / 30
            time_factor = math.exp(-self.config.value_decay_rate * months_old)

        expected_value = (
            base_value *
            relevance_factor *
            decision_factor *
            history_factor *
            time_factor
        )

        return expected_value

    def _calculate_query_probability(
        self,
        doc_type: Optional[DocumentType],
        relevance_score: float,
        historical_queries: int,
        metadata: Dict[str, Any]
    ) -> float:
        """쿼리 확률 계산"""
        # 기본 확률
        probability = self.config.base_query_probability

        # 문서 타입별 확률
        if doc_type:
            type_boost = self.config.type_probability_boost.get(
                doc_type.value, 0.5
            )
            probability = (probability + type_boost) / 2

        # 관련성 기반 보정
        probability *= (0.5 + relevance_score * 0.5)

        # 과거 쿼리 기반 보정
        if historical_queries > 0:
            probability = min(probability * (1 + math.log10(historical_queries + 1) * 0.1), 1.0)

        # 메타데이터 기반 보정
        if metadata.get("featured", False):
            probability *= 1.2
        if metadata.get("pinned", False):
            probability *= 1.3

        return min(max(probability, 0.01), 1.0)

    def _calculate_storage_cost(
        self,
        content_length: int,
        created_at: Optional[datetime]
    ) -> float:
        """저장 비용 계산"""
        # 예상 청크 수
        avg_chunk_size = 500  # 평균 청크 크기 (문자)
        estimated_chunks = max(content_length / avg_chunk_size, 1)

        # 임베딩 비용 (일회성)
        embedding_cost = estimated_chunks * self.config.embedding_cost_per_chunk

        # 저장 비용 (월별, 12개월 가정)
        months = 12
        if created_at:
            months = max((datetime.now() - created_at).days / 30, 1)
            months = min(months, 24)  # 최대 2년

        storage_cost = estimated_chunks * self.config.storage_cost_per_chunk * months

        # 검색 비용 (예상 쿼리 수 기반)
        estimated_queries = months * 10  # 월 10회 가정
        retrieval_cost = estimated_queries * self.config.retrieval_cost_per_query

        total_cost = embedding_cost + storage_cost + retrieval_cost

        return max(total_cost, 0.001)  # 최소값 보장

    def _determine_granularity(
        self,
        roi: float,
        content_length: int
    ) -> VectorizationGranularity:
        """세분화 수준 결정"""
        if roi < 0.5:
            return VectorizationGranularity.NONE
        elif roi < 1.0:
            return VectorizationGranularity.METADATA
        elif roi < 2.0:
            return VectorizationGranularity.SUMMARY
        elif roi < self.config.high_value_threshold:
            return VectorizationGranularity.CHUNKS
        else:
            return VectorizationGranularity.FULL

    def _determine_storage_tier(
        self,
        roi: float,
        query_probability: float
    ) -> StorageTier:
        """저장소 계층 결정"""
        if roi >= self.config.high_value_threshold or query_probability >= 0.8:
            return StorageTier.HOT
        elif roi >= self.config.vectorize_threshold or query_probability >= 0.4:
            return StorageTier.WARM
        else:
            return StorageTier.COLD

    def _generate_recommendation(
        self,
        total_roi: float,
        granularity: VectorizationGranularity,
        storage_tier: StorageTier
    ) -> str:
        """권장사항 생성"""
        if granularity == VectorizationGranularity.NONE:
            return "벡터화 불필요 - 메타데이터만 저장 권장"
        elif granularity == VectorizationGranularity.METADATA:
            return "메타데이터만 벡터화 - 필터링용으로 활용"
        elif granularity == VectorizationGranularity.SUMMARY:
            return "요약본 벡터화 - 전체 문서 대신 요약 사용"
        elif granularity == VectorizationGranularity.CHUNKS:
            return f"청크 단위 벡터화 - {storage_tier.value} 계층 저장"
        else:
            return f"전체 문서 벡터화 (고가치) - {storage_tier.value} 계층 저장, ROI={total_roi:.2f}"

    def should_vectorize(self, roi_score: ROIScore) -> bool:
        """벡터화 여부 결정"""
        return roi_score.granularity not in [
            VectorizationGranularity.NONE,
            VectorizationGranularity.METADATA
        ]

    def calculate_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[ROIScore]:
        """배치 ROI 계산"""
        scores = []

        for doc in documents:
            score = self.calculate(
                content_length=len(doc.get("content", "")),
                doc_type=doc.get("doc_type"),
                relevance_score=doc.get("relevance_score", 0.5),
                decision_influence=doc.get("decision_influence", 0.5),
                created_at=doc.get("created_at"),
                historical_queries=doc.get("historical_queries", 0),
                metadata=doc.get("metadata", {})
            )
            scores.append(score)

        return scores


class AdaptiveROICalculator(ROICalculator):
    """적응형 ROI 계산기 (학습 기반)"""

    def __init__(self, config: Optional[ROICalculatorConfig] = None):
        super().__init__(config)
        self._query_history: Dict[str, int] = {}
        self._feedback_history: List[Dict[str, Any]] = []

    def record_query(self, document_id: str) -> None:
        """쿼리 기록"""
        self._query_history[document_id] = (
            self._query_history.get(document_id, 0) + 1
        )

    def record_feedback(
        self,
        document_id: str,
        was_useful: bool,
        roi_score: float
    ) -> None:
        """피드백 기록"""
        self._feedback_history.append({
            "document_id": document_id,
            "was_useful": was_useful,
            "roi_score": roi_score,
            "timestamp": datetime.now().isoformat()
        })

        # 최대 1000개 유지
        if len(self._feedback_history) > 1000:
            self._feedback_history = self._feedback_history[-1000:]

    def get_document_query_count(self, document_id: str) -> int:
        """문서별 쿼리 횟수 조회"""
        return self._query_history.get(document_id, 0)

    def calculate_with_history(
        self,
        document_id: str,
        content_length: int,
        doc_type: Optional[DocumentType] = None,
        relevance_score: float = 0.5,
        decision_influence: float = 0.5,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ROIScore:
        """과거 이력을 반영한 ROI 계산"""
        historical_queries = self.get_document_query_count(document_id)

        return self.calculate(
            content_length=content_length,
            doc_type=doc_type,
            relevance_score=relevance_score,
            decision_influence=decision_influence,
            created_at=created_at,
            historical_queries=historical_queries,
            metadata=metadata
        )

    def get_roi_accuracy(self) -> float:
        """ROI 예측 정확도 계산"""
        if not self._feedback_history:
            return 0.5

        # 고ROI 문서가 유용했는지 확인
        high_roi_useful = 0
        high_roi_total = 0

        for feedback in self._feedback_history:
            if feedback["roi_score"] >= self.config.high_value_threshold:
                high_roi_total += 1
                if feedback["was_useful"]:
                    high_roi_useful += 1

        if high_roi_total == 0:
            return 0.5

        return high_roi_useful / high_roi_total
