"""
Evaluation Package

LLM/RAG 시스템 품질 평가 모듈
- RAG 평가: Faithfulness, Relevance, Context Recall
- LLM 평가: Coherence, Fluency, Toxicity
- 벤치마크 실행기
"""

from .base import (
    EvaluationMetric,
    EvaluationResult,
    EvaluationConfig,
    MetricType,
)
from .rag_evaluator import (
    RAGEvaluator,
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextRelevanceMetric,
    ContextRecallMetric,
)
from .llm_evaluator import (
    LLMEvaluator,
    CoherenceMetric,
    FluencyMetric,
    ToxicityMetric,
    HallucinationMetric,
)
from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkDataset,
)

__all__ = [
    # Base
    "EvaluationMetric",
    "EvaluationResult",
    "EvaluationConfig",
    "MetricType",
    # RAG Evaluator
    "RAGEvaluator",
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "ContextRelevanceMetric",
    "ContextRecallMetric",
    # LLM Evaluator
    "LLMEvaluator",
    "CoherenceMetric",
    "FluencyMetric",
    "ToxicityMetric",
    "HallucinationMetric",
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkDataset",
]
