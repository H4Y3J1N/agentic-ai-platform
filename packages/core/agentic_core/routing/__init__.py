"""
Routing Package - Intent Detection and Request Routing

LLM 기반 Intent 분류 및 라우팅 모듈

Usage:
    from agentic_core.routing import IntentRouter, Intent, IntentType

    router = IntentRouter(config)
    intent = await router.route("일본 프로젝트 진행상황 알려줘")

    if intent.type == IntentType.RAG_QA:
        result = await knowledge_agent.execute(query)
    elif intent.type == IntentType.TOOL:
        tool = registry.get(intent.tool_name)
        result = await tool.execute(**intent.params)
"""

from .intent import (
    Intent,
    IntentType,
    IntentConfidence,
    SubIntent,
)

from .router import (
    IntentRouter,
    IntentRouterConfig,
    QuickMatchRule,
)

__all__ = [
    # Intent types
    "Intent",
    "IntentType",
    "IntentConfidence",
    "SubIntent",
    # Router
    "IntentRouter",
    "IntentRouterConfig",
    "QuickMatchRule",
]
