"""
RAG Strategies Package

고급 RAG 파이프라인 전략 모음

Strategies:
- SingleShotRAG: 기본 RAG (검색 → 답변)
- CorrectiveRAG: 품질 평가 + 자동 재검색
- SelfRAG: 자기 검증 + 답변 수정

Usage:
    from agentic_core.rag.strategies import create_rag_strategy, RAGStrategyConfig

    # 팩토리로 생성
    strategy = create_rag_strategy("corrective")

    # 직접 생성
    from agentic_core.rag.strategies import CorrectiveRAG
    strategy = CorrectiveRAG(config)

    # 실행
    result = await strategy.execute(query, search_fn)
"""

from .base import (
    RAGStrategy,
    RAGStrategyType,
    RAGStrategyConfig,
    RAGResult,
    Document,
    ReasoningStep,
    RetrievalQuality,
    SearchFn,
    LLMFn,
)

from .single_shot import SingleShotRAG
from .corrective import CorrectiveRAG
from .self_rag import SelfRAG


# Strategy registry
_STRATEGY_REGISTRY = {
    RAGStrategyType.SINGLE_SHOT: SingleShotRAG,
    RAGStrategyType.CORRECTIVE: CorrectiveRAG,
    RAGStrategyType.SELF_RAG: SelfRAG,
    # RAGStrategyType.AGENTIC: AgenticRAG,  # TODO: 구현 예정
}

# String aliases
_STRATEGY_ALIASES = {
    "single_shot": RAGStrategyType.SINGLE_SHOT,
    "single": RAGStrategyType.SINGLE_SHOT,
    "basic": RAGStrategyType.SINGLE_SHOT,
    "corrective": RAGStrategyType.CORRECTIVE,
    "crag": RAGStrategyType.CORRECTIVE,
    "self_rag": RAGStrategyType.SELF_RAG,
    "self": RAGStrategyType.SELF_RAG,
    "critique": RAGStrategyType.SELF_RAG,
}


def create_rag_strategy(
    strategy_type: str | RAGStrategyType = "single_shot",
    config: RAGStrategyConfig | dict | None = None,
    **kwargs
) -> RAGStrategy:
    """
    RAG Strategy 팩토리 함수

    Args:
        strategy_type: 전략 타입 ("single_shot", "corrective", "self_rag") 또는 RAGStrategyType
        config: 설정 (RAGStrategyConfig 또는 dict)
        **kwargs: 추가 설정 (config보다 우선)

    Returns:
        RAGStrategy 인스턴스

    Examples:
        # 기본 전략
        strategy = create_rag_strategy()

        # Corrective RAG
        strategy = create_rag_strategy("corrective", max_retries=3)

        # Self-RAG with config
        config = RAGStrategyConfig(
            llm_model="gemini/gemini-1.5-flash",
            critique_threshold=0.8
        )
        strategy = create_rag_strategy("self_rag", config)
    """
    # 문자열이면 타입으로 변환
    if isinstance(strategy_type, str):
        strategy_type_lower = strategy_type.lower()
        if strategy_type_lower in _STRATEGY_ALIASES:
            strategy_type = _STRATEGY_ALIASES[strategy_type_lower]
        else:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available: {list(_STRATEGY_ALIASES.keys())}"
            )

    # 전략 클래스 조회
    if strategy_type not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Strategy not implemented: {strategy_type}. "
            f"Available: {[s.value for s in _STRATEGY_REGISTRY.keys()]}"
        )

    strategy_class = _STRATEGY_REGISTRY[strategy_type]

    # Config 처리
    if config is None:
        config = RAGStrategyConfig(**kwargs)
    elif isinstance(config, dict):
        merged = {**config, **kwargs}
        config = RAGStrategyConfig(**{
            k: v for k, v in merged.items()
            if k in RAGStrategyConfig.__dataclass_fields__
        })
    elif kwargs:
        # 기존 config에 kwargs 병합
        config_dict = {
            k: getattr(config, k)
            for k in RAGStrategyConfig.__dataclass_fields__
        }
        config_dict.update(kwargs)
        config = RAGStrategyConfig(**config_dict)

    return strategy_class(config)


def get_available_strategies() -> list[str]:
    """사용 가능한 전략 목록 반환"""
    return [s.value for s in _STRATEGY_REGISTRY.keys()]


__all__ = [
    # Base
    "RAGStrategy",
    "RAGStrategyType",
    "RAGStrategyConfig",
    "RAGResult",
    "Document",
    "ReasoningStep",
    "RetrievalQuality",
    "SearchFn",
    "LLMFn",
    # Strategies
    "SingleShotRAG",
    "CorrectiveRAG",
    "SelfRAG",
    # Factory
    "create_rag_strategy",
    "get_available_strategies",
]
