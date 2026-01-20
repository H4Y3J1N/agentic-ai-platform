"""
Reference-Free Evaluator

Ground Truth 없이 RAG 시스템을 평가하는 간편 래퍼.
Faithfulness, Answer Relevance, Context Relevance 메트릭을 통합.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime

from .base import MetricType, EvaluationResult, EvaluationConfig
from .rag_evaluator import (
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextRelevanceMetric,
)
from .llm_judge import LLMJudge, LLMJudgeConfig, create_llm_judge

logger = logging.getLogger(__name__)


@dataclass
class ReferenceFreeConfig:
    """Reference-Free Evaluator 설정"""
    # LLM 설정
    llm_model: str = "gemini/gemini-1.5-flash"
    llm_temperature: float = 0.0
    llm_timeout: float = 30.0

    # 활성화할 메트릭
    enable_faithfulness: bool = True
    enable_answer_relevance: bool = True
    enable_context_relevance: bool = True

    # 임계값
    faithfulness_threshold: float = 0.7
    answer_relevance_threshold: float = 0.7
    context_relevance_threshold: float = 0.6

    # 병렬 처리
    max_concurrent: int = 5


@dataclass
class EvaluationSample:
    """평가 샘플"""
    query: str
    response: str
    contexts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceFreeResult:
    """Reference-Free 평가 결과"""
    # 개별 점수
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_relevance: Optional[float] = None

    # 통과 여부
    faithfulness_passed: Optional[bool] = None
    answer_relevance_passed: Optional[bool] = None
    context_relevance_passed: Optional[bool] = None

    # 종합 점수
    overall_score: float = 0.0
    all_passed: bool = False

    # 상세 정보
    details: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.now)
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "faithfulness_passed": self.faithfulness_passed,
            "answer_relevance_passed": self.answer_relevance_passed,
            "context_relevance_passed": self.context_relevance_passed,
            "overall_score": self.overall_score,
            "all_passed": self.all_passed,
            "details": self.details,
            "evaluated_at": self.evaluated_at.isoformat(),
            "evaluation_time_ms": self.evaluation_time_ms,
        }


@dataclass
class BatchEvaluationResult:
    """배치 평가 결과"""
    results: List[ReferenceFreeResult]
    samples: List[EvaluationSample]

    # 집계 통계
    mean_faithfulness: float = 0.0
    mean_answer_relevance: float = 0.0
    mean_context_relevance: float = 0.0
    mean_overall: float = 0.0

    pass_rate_faithfulness: float = 0.0
    pass_rate_answer_relevance: float = 0.0
    pass_rate_context_relevance: float = 0.0
    pass_rate_all: float = 0.0

    total_samples: int = 0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "summary": {
                "total_samples": self.total_samples,
                "mean_faithfulness": self.mean_faithfulness,
                "mean_answer_relevance": self.mean_answer_relevance,
                "mean_context_relevance": self.mean_context_relevance,
                "mean_overall": self.mean_overall,
                "pass_rate_faithfulness": self.pass_rate_faithfulness,
                "pass_rate_answer_relevance": self.pass_rate_answer_relevance,
                "pass_rate_context_relevance": self.pass_rate_context_relevance,
                "pass_rate_all": self.pass_rate_all,
                "total_time_ms": self.total_time_ms,
            },
            "results": [r.to_dict() for r in self.results],
        }


class ReferenceFreeEvaluator:
    """
    Reference-Free RAG Evaluator

    Ground Truth 없이 RAG 시스템을 평가합니다.
    LLM-as-Judge 방식으로 3가지 핵심 메트릭을 측정:
    - Faithfulness: 응답이 컨텍스트에 충실한가
    - Answer Relevance: 응답이 질문에 관련 있는가
    - Context Relevance: 검색된 컨텍스트가 질문에 관련 있는가

    Example:
        >>> evaluator = ReferenceFreeEvaluator()
        >>> result = await evaluator.evaluate(
        ...     query="프로젝트 진행상황은?",
        ...     response="현재 검토 단계입니다.",
        ...     contexts=["프로젝트 A는 검토 단계", "담당자: 홍길동"]
        ... )
        >>> print(result.overall_score)
    """

    def __init__(self, config: Optional[ReferenceFreeConfig] = None):
        self.config = config or ReferenceFreeConfig()

        # LLM Judge 초기화
        self._llm_judge = create_llm_judge(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            timeout=self.config.llm_timeout
        )

        # 메트릭 초기화
        self._metrics: Dict[str, Any] = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """메트릭 초기화"""
        if self.config.enable_faithfulness:
            self._metrics["faithfulness"] = FaithfulnessMetric(
                threshold=self.config.faithfulness_threshold,
                llm_judge=self._llm_judge
            )

        if self.config.enable_answer_relevance:
            self._metrics["answer_relevance"] = AnswerRelevanceMetric(
                threshold=self.config.answer_relevance_threshold,
                llm_judge=self._llm_judge
            )

        if self.config.enable_context_relevance:
            self._metrics["context_relevance"] = ContextRelevanceMetric(
                threshold=self.config.context_relevance_threshold,
                llm_judge=self._llm_judge
            )

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        **kwargs
    ) -> ReferenceFreeResult:
        """
        단일 샘플 평가

        Args:
            query: 사용자 질문
            response: 시스템 응답
            contexts: 검색된 컨텍스트 목록

        Returns:
            ReferenceFreeResult: 평가 결과
        """
        import time
        start_time = time.time()

        result = ReferenceFreeResult()
        scores = []

        # 각 메트릭 평가 (병렬)
        tasks = {}
        for name, metric in self._metrics.items():
            tasks[name] = metric.evaluate(
                query=query,
                response=response,
                context=contexts
            )

        metric_results = {}
        for name, task in tasks.items():
            try:
                metric_results[name] = await task
            except Exception as e:
                logger.warning(f"Metric {name} failed: {e}")
                metric_results[name] = None

        # 결과 매핑
        if "faithfulness" in metric_results and metric_results["faithfulness"]:
            mr = metric_results["faithfulness"]
            result.faithfulness = mr.score
            result.faithfulness_passed = mr.passed
            result.details["faithfulness"] = mr.details
            scores.append(mr.score)

        if "answer_relevance" in metric_results and metric_results["answer_relevance"]:
            mr = metric_results["answer_relevance"]
            result.answer_relevance = mr.score
            result.answer_relevance_passed = mr.passed
            result.details["answer_relevance"] = mr.details
            scores.append(mr.score)

        if "context_relevance" in metric_results and metric_results["context_relevance"]:
            mr = metric_results["context_relevance"]
            result.context_relevance = mr.score
            result.context_relevance_passed = mr.passed
            result.details["context_relevance"] = mr.details
            scores.append(mr.score)

        # 종합 점수 (평균)
        if scores:
            result.overall_score = sum(scores) / len(scores)

        # 전체 통과 여부
        passed_list = [
            result.faithfulness_passed,
            result.answer_relevance_passed,
            result.context_relevance_passed
        ]
        result.all_passed = all(p for p in passed_list if p is not None)

        result.evaluation_time_ms = (time.time() - start_time) * 1000

        return result

    async def evaluate_batch(
        self,
        samples: List[EvaluationSample]
    ) -> BatchEvaluationResult:
        """
        배치 평가

        Args:
            samples: 평가할 샘플 목록

        Returns:
            BatchEvaluationResult: 집계된 평가 결과
        """
        import time
        start_time = time.time()

        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def evaluate_with_limit(sample: EvaluationSample) -> ReferenceFreeResult:
            async with semaphore:
                return await self.evaluate(
                    query=sample.query,
                    response=sample.response,
                    contexts=sample.contexts
                )

        tasks = [evaluate_with_limit(s) for s in samples]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 에러 처리
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Sample evaluation failed: {r}")
                valid_results.append(ReferenceFreeResult())
            else:
                valid_results.append(r)

        # 집계
        batch_result = BatchEvaluationResult(
            results=valid_results,
            samples=samples,
            total_samples=len(samples)
        )

        # 평균 계산
        faithfulness_scores = [r.faithfulness for r in valid_results if r.faithfulness is not None]
        answer_rel_scores = [r.answer_relevance for r in valid_results if r.answer_relevance is not None]
        context_rel_scores = [r.context_relevance for r in valid_results if r.context_relevance is not None]
        overall_scores = [r.overall_score for r in valid_results if r.overall_score > 0]

        if faithfulness_scores:
            batch_result.mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        if answer_rel_scores:
            batch_result.mean_answer_relevance = sum(answer_rel_scores) / len(answer_rel_scores)
        if context_rel_scores:
            batch_result.mean_context_relevance = sum(context_rel_scores) / len(context_rel_scores)
        if overall_scores:
            batch_result.mean_overall = sum(overall_scores) / len(overall_scores)

        # 통과율 계산
        faith_passed = [r.faithfulness_passed for r in valid_results if r.faithfulness_passed is not None]
        ans_passed = [r.answer_relevance_passed for r in valid_results if r.answer_relevance_passed is not None]
        ctx_passed = [r.context_relevance_passed for r in valid_results if r.context_relevance_passed is not None]
        all_passed = [r.all_passed for r in valid_results]

        if faith_passed:
            batch_result.pass_rate_faithfulness = sum(faith_passed) / len(faith_passed)
        if ans_passed:
            batch_result.pass_rate_answer_relevance = sum(ans_passed) / len(ans_passed)
        if ctx_passed:
            batch_result.pass_rate_context_relevance = sum(ctx_passed) / len(ctx_passed)
        if all_passed:
            batch_result.pass_rate_all = sum(all_passed) / len(all_passed)

        batch_result.total_time_ms = (time.time() - start_time) * 1000

        return batch_result

    def print_summary(self, result: BatchEvaluationResult):
        """배치 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("RAG Evaluation Summary (Reference-Free)")
        print("=" * 60)
        print(f"Total Samples: {result.total_samples}")
        print(f"Total Time: {result.total_time_ms:.0f}ms")
        print("-" * 60)
        print(f"{'Metric':<25} {'Mean Score':<15} {'Pass Rate':<15}")
        print("-" * 60)

        if self.config.enable_faithfulness:
            print(f"{'Faithfulness':<25} {result.mean_faithfulness:.3f}{'':<10} {result.pass_rate_faithfulness:.1%}")

        if self.config.enable_answer_relevance:
            print(f"{'Answer Relevance':<25} {result.mean_answer_relevance:.3f}{'':<10} {result.pass_rate_answer_relevance:.1%}")

        if self.config.enable_context_relevance:
            print(f"{'Context Relevance':<25} {result.mean_context_relevance:.3f}{'':<10} {result.pass_rate_context_relevance:.1%}")

        print("-" * 60)
        print(f"{'Overall':<25} {result.mean_overall:.3f}{'':<10} {result.pass_rate_all:.1%}")
        print("=" * 60 + "\n")


def create_reference_free_evaluator(
    model: str = "gemini/gemini-1.5-flash",
    **kwargs
) -> ReferenceFreeEvaluator:
    """
    Reference-Free Evaluator 팩토리 함수

    Args:
        model: LLM 모델
        **kwargs: 추가 설정

    Returns:
        ReferenceFreeEvaluator 인스턴스

    Example:
        >>> evaluator = create_reference_free_evaluator()
        >>> result = await evaluator.evaluate(
        ...     query="질문",
        ...     response="응답",
        ...     contexts=["컨텍스트1", "컨텍스트2"]
        ... )
    """
    config = ReferenceFreeConfig(
        llm_model=model,
        **{k: v for k, v in kwargs.items() if k in ReferenceFreeConfig.__dataclass_fields__}
    )
    return ReferenceFreeEvaluator(config)
