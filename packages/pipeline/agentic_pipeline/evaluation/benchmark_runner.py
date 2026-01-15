"""
Benchmark Runner

벤치마크 데이터셋을 사용한 평가 실행기
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
from pathlib import Path

from .base import EvaluationConfig, EvaluationResult, MetricType
from .rag_evaluator import RAGEvaluator
from .llm_evaluator import LLMEvaluator


@dataclass
class BenchmarkSample:
    """벤치마크 샘플"""
    id: str
    query: str
    ground_truth: Optional[str] = None
    context: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """벤치마크 데이터셋"""
    name: str
    samples: List[BenchmarkSample]
    description: Optional[str] = None
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkDataset":
        """JSON 파일에서 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = [
            BenchmarkSample(
                id=s.get("id", str(i)),
                query=s["query"],
                ground_truth=s.get("ground_truth"),
                context=s.get("context"),
                metadata=s.get("metadata", {}),
            )
            for i, s in enumerate(data.get("samples", []))
        ]

        return cls(
            name=data.get("name", "unknown"),
            samples=samples,
            description=data.get("description"),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, path: str):
        """JSON 파일로 저장"""
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
            "samples": [
                {
                    "id": s.id,
                    "query": s.query,
                    "ground_truth": s.ground_truth,
                    "context": s.context,
                    "metadata": s.metadata,
                }
                for s in self.samples
            ]
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    dataset_name: str
    run_id: str
    started_at: datetime
    completed_at: datetime

    # 집계 결과
    aggregate_scores: Dict[str, Dict[str, float]]  # metric -> {mean, min, max, pass_rate}
    overall_pass_rate: float

    # 개별 결과
    sample_results: List[Dict[str, Any]]

    # 메타데이터
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "aggregate_scores": self.aggregate_scores,
            "overall_pass_rate": self.overall_pass_rate,
            "sample_count": len(self.sample_results),
            "config": self.config,
            "metadata": self.metadata,
        }

    def save(self, path: str):
        """결과 저장"""
        data = self.to_dict()
        data["sample_results"] = self.sample_results

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


class BenchmarkRunner:
    """
    벤치마크 실행기

    데이터셋에 대해 RAG/LLM 시스템을 평가합니다.
    """

    def __init__(
        self,
        response_generator: Callable[[str, Optional[List[str]]], str],
        retriever: Optional[Callable[[str], List[str]]] = None,
        config: Optional[EvaluationConfig] = None,
        llm_judge: Optional[Callable] = None,
    ):
        """
        Args:
            response_generator: 응답 생성 함수 (query, context) -> response
            retriever: 검색 함수 (query) -> context (없으면 데이터셋 context 사용)
            config: 평가 설정
            llm_judge: LLM 평가 함수
        """
        self.response_generator = response_generator
        self.retriever = retriever
        self.config = config or EvaluationConfig()
        self.llm_judge = llm_judge

        # 평가기 초기화
        self.rag_evaluator = RAGEvaluator(config=self.config, llm_judge=llm_judge)
        self.llm_evaluator = LLMEvaluator(config=self.config, llm_judge=llm_judge)

    async def run(
        self,
        dataset: BenchmarkDataset,
        run_id: Optional[str] = None,
        include_rag_metrics: bool = True,
        include_llm_metrics: bool = True,
    ) -> BenchmarkResult:
        """
        벤치마크 실행

        Args:
            dataset: 벤치마크 데이터셋
            run_id: 실행 ID (없으면 자동 생성)
            include_rag_metrics: RAG 메트릭 포함 여부
            include_llm_metrics: LLM 메트릭 포함 여부

        Returns:
            BenchmarkResult: 벤치마크 결과
        """
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        started_at = datetime.now()
        sample_results = []

        # 샘플별 처리
        for sample in dataset.samples:
            result = await self._process_sample(
                sample,
                include_rag_metrics,
                include_llm_metrics,
            )
            sample_results.append(result)

        completed_at = datetime.now()

        # 결과 집계
        aggregate_scores = self._aggregate_results(sample_results)
        overall_pass_rate = self._calculate_overall_pass_rate(sample_results)

        return BenchmarkResult(
            dataset_name=dataset.name,
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            aggregate_scores=aggregate_scores,
            overall_pass_rate=overall_pass_rate,
            sample_results=sample_results,
            config=self.config.model_dump() if hasattr(self.config, 'model_dump') else {},
            metadata={
                "dataset_version": dataset.version,
                "sample_count": len(dataset.samples),
            },
        )

    async def _process_sample(
        self,
        sample: BenchmarkSample,
        include_rag_metrics: bool,
        include_llm_metrics: bool,
    ) -> Dict[str, Any]:
        """개별 샘플 처리"""
        # 컨텍스트 검색 (retriever가 있으면 사용, 없으면 데이터셋 context)
        if self.retriever:
            context = await self._async_call(self.retriever, sample.query)
        else:
            context = sample.context

        # 응답 생성
        response = await self._async_call(
            self.response_generator,
            sample.query,
            context,
        )

        result = {
            "sample_id": sample.id,
            "query": sample.query,
            "response": response,
            "context": context,
            "ground_truth": sample.ground_truth,
            "metrics": {},
        }

        # RAG 평가
        if include_rag_metrics and context:
            rag_results = await self.rag_evaluator.evaluate(
                query=sample.query,
                response=response,
                context=context,
                ground_truth=sample.ground_truth,
            )
            for metric_type, eval_result in rag_results.items():
                result["metrics"][metric_type.value] = eval_result.to_dict()

        # LLM 평가
        if include_llm_metrics:
            llm_results = await self.llm_evaluator.evaluate(
                query=sample.query,
                response=response,
                context=context,
                ground_truth=sample.ground_truth,
            )
            for metric_type, eval_result in llm_results.items():
                result["metrics"][metric_type.value] = eval_result.to_dict()

        return result

    async def _async_call(self, func: Callable, *args) -> Any:
        """동기/비동기 함수 호출"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args)

    def _aggregate_results(
        self,
        sample_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """결과 집계"""
        aggregates = {}

        # 모든 메트릭 타입 수집
        all_metrics = set()
        for result in sample_results:
            all_metrics.update(result.get("metrics", {}).keys())

        for metric_name in all_metrics:
            scores = []
            passed_count = 0

            for result in sample_results:
                metric_data = result.get("metrics", {}).get(metric_name)
                if metric_data:
                    scores.append(metric_data.get("score", 0.0))
                    if metric_data.get("passed"):
                        passed_count += 1

            if scores:
                aggregates[metric_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std": self._std(scores),
                    "pass_rate": passed_count / len(scores),
                    "count": len(scores),
                }

        return aggregates

    def _calculate_overall_pass_rate(
        self,
        sample_results: List[Dict[str, Any]]
    ) -> float:
        """전체 통과율 계산"""
        if not sample_results:
            return 0.0

        sample_pass_count = 0

        for result in sample_results:
            metrics = result.get("metrics", {})
            if metrics:
                # 모든 메트릭이 통과해야 샘플 통과
                all_passed = all(
                    m.get("passed", False)
                    for m in metrics.values()
                )
                if all_passed:
                    sample_pass_count += 1

        return sample_pass_count / len(sample_results)

    def _std(self, values: List[float]) -> float:
        """표준편차 계산"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    async def run_comparison(
        self,
        dataset: BenchmarkDataset,
        systems: Dict[str, Callable[[str, Optional[List[str]]], str]],
    ) -> Dict[str, BenchmarkResult]:
        """
        여러 시스템 비교 벤치마크

        Args:
            dataset: 벤치마크 데이터셋
            systems: 시스템 이름 -> 응답 생성 함수

        Returns:
            시스템별 벤치마크 결과
        """
        results = {}

        for name, generator in systems.items():
            self.response_generator = generator
            result = await self.run(
                dataset,
                run_id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}",
            )
            results[name] = result

        return results
