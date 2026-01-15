"""
Evaluation Base

평가 메트릭 기본 인터페이스 및 공통 타입
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class MetricType(str, Enum):
    """평가 메트릭 타입"""
    # RAG 메트릭
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RELEVANCE = "context_relevance"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"

    # LLM 메트릭
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    TOXICITY = "toxicity"
    HALLUCINATION = "hallucination"
    BIAS = "bias"

    # 종합 메트릭
    OVERALL_QUALITY = "overall_quality"
    CUSTOM = "custom"


@dataclass
class EvaluationResult:
    """평가 결과"""

    metric_type: MetricType
    score: float  # 0.0 ~ 1.0
    passed: bool  # 임계값 통과 여부

    # 상세 정보
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    # 입력 데이터 참조
    query: Optional[str] = None
    response: Optional[str] = None
    context: Optional[List[str]] = None
    ground_truth: Optional[str] = None

    # 메타데이터
    evaluated_at: datetime = field(default_factory=datetime.now)
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "metric_type": self.metric_type.value,
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details,
            "evaluated_at": self.evaluated_at.isoformat(),
            "evaluation_time_ms": self.evaluation_time_ms,
        }


@dataclass
class EvaluationConfig:
    """평가 설정"""

    # 활성화할 메트릭
    metrics: List[MetricType] = field(default_factory=lambda: [
        MetricType.FAITHFULNESS,
        MetricType.ANSWER_RELEVANCE,
    ])

    # 임계값
    thresholds: Dict[MetricType, float] = field(default_factory=lambda: {
        MetricType.FAITHFULNESS: 0.7,
        MetricType.ANSWER_RELEVANCE: 0.7,
        MetricType.CONTEXT_RELEVANCE: 0.6,
        MetricType.CONTEXT_RECALL: 0.6,
        MetricType.COHERENCE: 0.7,
        MetricType.FLUENCY: 0.7,
        MetricType.TOXICITY: 0.1,  # 낮을수록 좋음
        MetricType.HALLUCINATION: 0.2,  # 낮을수록 좋음
    })

    # LLM 평가자 설정
    judge_model: str = "gpt-4o-mini"  # 평가에 사용할 모델
    judge_temperature: float = 0.0

    # 병렬 처리
    max_concurrent: int = 5

    # 캐싱
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600


class EvaluationMetric(ABC):
    """
    평가 메트릭 추상 클래스

    모든 평가 메트릭은 이 클래스를 상속받아 구현합니다.
    """

    def __init__(
        self,
        metric_type: MetricType,
        threshold: float = 0.7,
        reverse_score: bool = False,
    ):
        """
        Args:
            metric_type: 메트릭 타입
            threshold: 통과 임계값
            reverse_score: True이면 낮은 점수가 좋음 (toxicity 등)
        """
        self.metric_type = metric_type
        self.threshold = threshold
        self.reverse_score = reverse_score

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        평가 수행

        Args:
            query: 사용자 질문
            response: 시스템 응답
            context: 검색된 컨텍스트 (RAG용)
            ground_truth: 정답 (있는 경우)
            **kwargs: 추가 파라미터

        Returns:
            EvaluationResult: 평가 결과
        """
        pass

    def _create_result(
        self,
        score: float,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        response: Optional[str] = None,
        context: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        evaluation_time_ms: float = 0.0,
    ) -> EvaluationResult:
        """결과 객체 생성 헬퍼"""
        # reverse_score면 임계값 비교 반전
        if self.reverse_score:
            passed = score <= self.threshold
        else:
            passed = score >= self.threshold

        return EvaluationResult(
            metric_type=self.metric_type,
            score=score,
            passed=passed,
            reason=reason,
            details=details or {},
            query=query,
            response=response,
            context=context,
            ground_truth=ground_truth,
            evaluation_time_ms=evaluation_time_ms,
        )

    @property
    def name(self) -> str:
        """메트릭 이름"""
        return self.metric_type.value
