"""
RAG Evaluator

RAG 시스템 품질 평가 메트릭
- Faithfulness: 응답이 컨텍스트에 충실한지
- Answer Relevance: 응답이 질문에 관련있는지
- Context Relevance: 검색된 컨텍스트가 질문에 관련있는지
- Context Recall: 정답에 필요한 정보가 컨텍스트에 있는지
"""

from typing import Optional, Dict, Any, List, Callable
import time
import asyncio
import json
import re

from .base import EvaluationMetric, EvaluationResult, MetricType, EvaluationConfig


class FaithfulnessMetric(EvaluationMetric):
    """
    Faithfulness (충실도)

    응답이 주어진 컨텍스트의 정보만을 기반으로 하는지 평가합니다.
    환각(hallucination)을 감지하는 데 중요합니다.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        llm_judge: Optional[Callable] = None,
    ):
        super().__init__(MetricType.FAITHFULNESS, threshold)
        self.llm_judge = llm_judge

    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        start_time = time.time()

        if not context:
            return self._create_result(
                score=0.0,
                reason="No context provided",
                query=query,
                response=response,
            )

        context_text = "\n\n".join(context)

        if self.llm_judge:
            # LLM 기반 평가
            prompt = f"""You are evaluating the faithfulness of an AI response.

Question: {query}

Context:
{context_text}

Response: {response}

Evaluate whether the response is faithful to the given context.
A faithful response only contains information that can be derived from the context.

Return your evaluation in the following JSON format:
{{
    "score": <float between 0 and 1>,
    "faithful_claims": [<list of claims in response that are supported by context>],
    "unfaithful_claims": [<list of claims in response not supported by context>],
    "reason": "<brief explanation>"
}}
"""
            result = await self.llm_judge(prompt)
            try:
                parsed = self._parse_json_response(result)
                score = parsed.get("score", 0.5)
                reason = parsed.get("reason", "")
                details = {
                    "faithful_claims": parsed.get("faithful_claims", []),
                    "unfaithful_claims": parsed.get("unfaithful_claims", []),
                }
            except Exception:
                score = 0.5
                reason = "Failed to parse LLM response"
                details = {}
        else:
            # 간단한 키워드 기반 평가 (LLM 없이)
            score, details = self._keyword_based_evaluation(response, context_text)
            reason = "Keyword-based evaluation (no LLM judge)"

        elapsed_ms = (time.time() - start_time) * 1000

        return self._create_result(
            score=score,
            reason=reason,
            details=details,
            query=query,
            response=response,
            context=context,
            evaluation_time_ms=elapsed_ms,
        )

    def _keyword_based_evaluation(
        self,
        response: str,
        context: str
    ) -> tuple[float, Dict[str, Any]]:
        """간단한 키워드 기반 평가"""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        # 응답 단어 중 컨텍스트에 있는 비율
        if not response_words:
            return 0.0, {}

        overlap = response_words & context_words
        coverage = len(overlap) / len(response_words)

        return min(coverage * 1.5, 1.0), {"word_overlap_ratio": coverage}

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 추출"""
        # JSON 블록 찾기
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)


class AnswerRelevanceMetric(EvaluationMetric):
    """
    Answer Relevance (응답 관련성)

    응답이 질문에 얼마나 관련있고 적절한지 평가합니다.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        llm_judge: Optional[Callable] = None,
    ):
        super().__init__(MetricType.ANSWER_RELEVANCE, threshold)
        self.llm_judge = llm_judge

    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        start_time = time.time()

        if self.llm_judge:
            prompt = f"""You are evaluating the relevance of an AI response to a question.

Question: {query}

Response: {response}

Evaluate how well the response addresses the question.
Consider:
1. Does it directly answer what was asked?
2. Is it complete or does it miss key aspects?
3. Does it stay on topic?

Return your evaluation in JSON format:
{{
    "score": <float between 0 and 1>,
    "addresses_question": <true/false>,
    "completeness": "<low/medium/high>",
    "reason": "<brief explanation>"
}}
"""
            result = await self.llm_judge(prompt)
            try:
                parsed = self._parse_json_response(result)
                score = parsed.get("score", 0.5)
                reason = parsed.get("reason", "")
                details = {
                    "addresses_question": parsed.get("addresses_question"),
                    "completeness": parsed.get("completeness"),
                }
            except Exception:
                score = 0.5
                reason = "Failed to parse LLM response"
                details = {}
        else:
            # 간단한 휴리스틱
            score = self._heuristic_evaluation(query, response)
            reason = "Heuristic evaluation"
            details = {}

        elapsed_ms = (time.time() - start_time) * 1000

        return self._create_result(
            score=score,
            reason=reason,
            details=details,
            query=query,
            response=response,
            context=context,
            evaluation_time_ms=elapsed_ms,
        )

    def _heuristic_evaluation(self, query: str, response: str) -> float:
        """간단한 휴리스틱 평가"""
        # 응답이 너무 짧거나 길면 감점
        if len(response) < 10:
            return 0.2
        if len(response) < 50:
            return 0.5

        # 질문 키워드가 응답에 포함되어 있는지
        query_words = set(query.lower().split())
        response_lower = response.lower()

        keyword_hits = sum(1 for w in query_words if w in response_lower)
        keyword_ratio = keyword_hits / max(len(query_words), 1)

        return min(0.5 + keyword_ratio * 0.5, 1.0)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)


class ContextRelevanceMetric(EvaluationMetric):
    """
    Context Relevance (컨텍스트 관련성)

    검색된 컨텍스트가 질문에 얼마나 관련있는지 평가합니다.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        llm_judge: Optional[Callable] = None,
    ):
        super().__init__(MetricType.CONTEXT_RELEVANCE, threshold)
        self.llm_judge = llm_judge

    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        start_time = time.time()

        if not context:
            return self._create_result(
                score=0.0,
                reason="No context provided",
                query=query,
                response=response,
            )

        if self.llm_judge:
            # 각 컨텍스트 청크의 관련성 평가
            relevance_scores = []
            for i, ctx in enumerate(context):
                prompt = f"""Rate the relevance of this context to the question.

Question: {query}

Context: {ctx}

Return JSON: {{"score": <0-1>, "reason": "<brief>"}}
"""
                try:
                    result = await self.llm_judge(prompt)
                    parsed = self._parse_json_response(result)
                    relevance_scores.append({
                        "index": i,
                        "score": parsed.get("score", 0.5),
                        "reason": parsed.get("reason", ""),
                    })
                except Exception:
                    relevance_scores.append({"index": i, "score": 0.5})

            avg_score = sum(r["score"] for r in relevance_scores) / len(relevance_scores)
            details = {"per_context_scores": relevance_scores}
            reason = f"Average relevance across {len(context)} contexts"
        else:
            # 키워드 기반
            query_words = set(query.lower().split())
            scores = []
            for ctx in context:
                ctx_words = set(ctx.lower().split())
                overlap = len(query_words & ctx_words) / max(len(query_words), 1)
                scores.append(overlap)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            reason = "Keyword-based evaluation"
            details = {"per_context_scores": scores}

        elapsed_ms = (time.time() - start_time) * 1000

        return self._create_result(
            score=avg_score,
            reason=reason,
            details=details,
            query=query,
            response=response,
            context=context,
            evaluation_time_ms=elapsed_ms,
        )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)


class ContextRecallMetric(EvaluationMetric):
    """
    Context Recall (컨텍스트 재현율)

    정답에 필요한 정보가 검색된 컨텍스트에 얼마나 포함되어 있는지 평가합니다.
    Ground truth가 필요합니다.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        llm_judge: Optional[Callable] = None,
    ):
        super().__init__(MetricType.CONTEXT_RECALL, threshold)
        self.llm_judge = llm_judge

    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        start_time = time.time()

        if not ground_truth:
            return self._create_result(
                score=0.5,
                reason="No ground truth provided, cannot compute recall",
                query=query,
                response=response,
                context=context,
            )

        if not context:
            return self._create_result(
                score=0.0,
                reason="No context provided",
                query=query,
                response=response,
                ground_truth=ground_truth,
            )

        context_text = "\n".join(context)

        if self.llm_judge:
            prompt = f"""Evaluate what fraction of the ground truth can be attributed to the context.

Question: {query}

Ground Truth: {ground_truth}

Retrieved Context:
{context_text}

Break down the ground truth into key facts/claims.
For each, determine if it can be found in the context.

Return JSON:
{{
    "score": <fraction of ground truth claims found in context, 0-1>,
    "claims_in_ground_truth": [<list of key claims>],
    "claims_found_in_context": [<claims that appear in context>],
    "claims_missing": [<claims not in context>],
    "reason": "<explanation>"
}}
"""
            try:
                result = await self.llm_judge(prompt)
                parsed = self._parse_json_response(result)
                score = parsed.get("score", 0.5)
                reason = parsed.get("reason", "")
                details = {
                    "claims_in_ground_truth": parsed.get("claims_in_ground_truth", []),
                    "claims_found": parsed.get("claims_found_in_context", []),
                    "claims_missing": parsed.get("claims_missing", []),
                }
            except Exception:
                score = 0.5
                reason = "Failed to parse LLM response"
                details = {}
        else:
            # 키워드 기반
            gt_words = set(ground_truth.lower().split())
            ctx_words = set(context_text.lower().split())
            recall = len(gt_words & ctx_words) / max(len(gt_words), 1)
            score = recall
            reason = "Keyword-based recall"
            details = {"word_recall": recall}

        elapsed_ms = (time.time() - start_time) * 1000

        return self._create_result(
            score=score,
            reason=reason,
            details=details,
            query=query,
            response=response,
            context=context,
            ground_truth=ground_truth,
            evaluation_time_ms=elapsed_ms,
        )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)


class RAGEvaluator:
    """
    RAG 평가기

    여러 메트릭을 조합하여 RAG 시스템을 종합 평가합니다.
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        llm_judge: Optional[Callable] = None,
    ):
        self.config = config or EvaluationConfig()
        self.llm_judge = llm_judge

        # 메트릭 초기화
        self.metrics: Dict[MetricType, EvaluationMetric] = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """설정에 따라 메트릭 초기화"""
        metric_classes = {
            MetricType.FAITHFULNESS: FaithfulnessMetric,
            MetricType.ANSWER_RELEVANCE: AnswerRelevanceMetric,
            MetricType.CONTEXT_RELEVANCE: ContextRelevanceMetric,
            MetricType.CONTEXT_RECALL: ContextRecallMetric,
        }

        for metric_type in self.config.metrics:
            if metric_type in metric_classes:
                threshold = self.config.thresholds.get(metric_type, 0.7)
                self.metrics[metric_type] = metric_classes[metric_type](
                    threshold=threshold,
                    llm_judge=self.llm_judge,
                )

    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, EvaluationResult]:
        """
        모든 활성 메트릭으로 평가

        Returns:
            메트릭별 평가 결과
        """
        tasks = []
        metric_names = []

        for name, metric in self.metrics.items():
            tasks.append(
                metric.evaluate(query, response, context, ground_truth, **kwargs)
            )
            metric_names.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            name: result if not isinstance(result, Exception)
            else self._error_result(name, str(result))
            for name, result in zip(metric_names, results)
        }

    def _error_result(self, metric_type: MetricType, error: str) -> EvaluationResult:
        """에러 결과 생성"""
        return EvaluationResult(
            metric_type=metric_type,
            score=0.0,
            passed=False,
            reason=f"Evaluation error: {error}",
        )

    async def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, EvaluationResult]]:
        """
        배치 평가

        Args:
            samples: [{"query": ..., "response": ..., "context": ..., "ground_truth": ...}, ...]

        Returns:
            샘플별 평가 결과 리스트
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def evaluate_with_limit(sample):
            async with semaphore:
                return await self.evaluate(**sample)

        tasks = [evaluate_with_limit(s) for s in samples]
        return await asyncio.gather(*tasks)

    def get_aggregate_scores(
        self,
        results: List[Dict[str, EvaluationResult]]
    ) -> Dict[str, Dict[str, float]]:
        """
        결과 집계

        Returns:
            메트릭별 평균, 최소, 최대, 통과율
        """
        aggregates = {}

        for metric_type in self.metrics:
            scores = [
                r[metric_type].score
                for r in results
                if metric_type in r
            ]
            passed = [
                r[metric_type].passed
                for r in results
                if metric_type in r
            ]

            if scores:
                aggregates[metric_type.value] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "pass_rate": sum(passed) / len(passed),
                    "count": len(scores),
                }

        return aggregates
