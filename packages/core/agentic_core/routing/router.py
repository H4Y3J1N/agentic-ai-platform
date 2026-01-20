"""
Intent Router - LLM 기반 Intent 분류기

2단계 라우팅:
1. Quick Match: 명확한 패턴은 정규식으로 빠르게 분류
2. LLM Classification: 애매한 경우 LLM으로 정밀 분류
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple

from .intent import Intent, IntentType, IntentConfidence, SubIntent

logger = logging.getLogger(__name__)


@dataclass
class QuickMatchRule:
    """빠른 매칭을 위한 규칙"""
    name: str
    patterns: List[str]                 # 정규식 패턴 목록
    intent_type: IntentType
    tool_name: Optional[str] = None     # 매칭 시 사용할 Tool
    confidence: float = 0.95            # 규칙 매칭 시 신뢰도
    extract_params: Optional[Callable[[str, re.Match], Dict]] = None  # 파라미터 추출 함수


@dataclass
class IntentRouterConfig:
    """Router 설정"""
    # LLM 설정
    llm_model: str = "gemini/gemini-1.5-flash"
    llm_temperature: float = 0.1        # 낮은 temperature로 일관된 분류
    llm_timeout: float = 10.0

    # 라우팅 설정
    use_llm_fallback: bool = True       # 규칙 미매칭 시 LLM 사용
    quick_match_only: bool = False      # True면 LLM 사용 안함
    confidence_threshold: float = 0.5   # 이 이상이면 분류 확정

    # 대화 감지
    detect_conversation: bool = True    # 일반 대화 감지 여부

    # 커스텀 규칙
    custom_rules: List[QuickMatchRule] = field(default_factory=list)


class IntentRouter:
    """
    LLM 기반 Intent Router

    2단계 라우팅으로 속도와 정확도 균형:
    1. Quick Match (< 1ms): 명확한 패턴
    2. LLM Classification (~500ms): 복잡한 케이스

    Example:
        router = IntentRouter(config)

        # 단순 라우팅
        intent = await router.route("태스크 만들어줘")
        # Intent(type=TOOL_CREATE, tool_name="task_creation", confidence=0.95)

        # 복합 인텐트
        intent = await router.route("프로젝트 검색하고 태스크 추가해줘")
        # Intent(type=MULTI_INTENT, sub_intents=[...])
    """

    # LLM 분류 프롬프트
    CLASSIFICATION_PROMPT = """You are an intent classifier for an internal operations assistant.

Classify the user's request into ONE of these categories:

**Tool Intents** (actions that modify data):
- TOOL_CREATE: Create something (task, document, etc.)
- TOOL_SEARCH: Explicit search request with keywords like "검색", "찾아", "search"
- TOOL_UPDATE: Modify existing data
- TOOL_DELETE: Remove data

**RAG Intents** (questions requiring knowledge lookup):
- RAG_QA: Questions about facts, status, information ("~이 뭐야?", "~알려줘", "진행상황")
- RAG_SUMMARY: Summarization requests ("요약해줘", "정리해줘")
- RAG_COMPARE: Comparison requests ("비교해줘", "차이점")

**Conversation Intents** (no action needed):
- CONVERSATION: Greetings, chitchat ("안녕", "고마워", "뭐해?")
- CLARIFICATION: Asking for clarification ("뭐라고?", "다시 설명해줘")
- FEEDBACK: Giving feedback ("좋아", "별로야")

**Special**:
- MULTI_INTENT: Multiple distinct requests in one message
- UNKNOWN: Cannot classify

User message: "{query}"

Respond with JSON only:
{{
    "intent": "<INTENT_TYPE>",
    "confidence": <0.0-1.0>,
    "tool_name": "<tool_name or null>",
    "processed_query": "<cleaned query for execution>",
    "reasoning": "<brief explanation>",
    "sub_intents": [
        {{"intent": "<type>", "query": "<part>", "tool_name": "<tool>", "order": <n>}}
    ]
}}

Tool names: "notion_search", "slack_search", "task_creation"

Examples:
- "태스크 만들어" → TOOL_CREATE, tool_name="task_creation"
- "프로젝트 진행상황 알려줘" → RAG_QA (question about status)
- "노션에서 검색해줘" → TOOL_SEARCH, tool_name="notion_search"
- "안녕하세요" → CONVERSATION
- "검색하고 태스크도 만들어줘" → MULTI_INTENT

JSON:"""

    def __init__(self, config: Optional[IntentRouterConfig] = None):
        self.config = config or IntentRouterConfig()
        self._llm_gateway = None
        self._quick_rules = self._build_default_rules()

        # Add custom rules
        self._quick_rules.extend(self.config.custom_rules)

    def _build_default_rules(self) -> List[QuickMatchRule]:
        """기본 Quick Match 규칙 생성"""
        return [
            # Task Creation (Korean + English)
            QuickMatchRule(
                name="task_create",
                patterns=[
                    r"create\s+(a\s+)?task",
                    r"add\s+(a\s+)?task",
                    r"make\s+(a\s+)?task",
                    r"new\s+task",
                    r"태스크\s*(를|을)?\s*(만들|생성|추가|등록)",
                    r"task\s*(를|을)?\s*(만들|생성|추가|등록)",
                    r"할\s*일\s*(을|를)?\s*(만들|생성|추가|등록)",
                ],
                intent_type=IntentType.TOOL_CREATE,
                tool_name="task_creation",
                confidence=0.95,
            ),

            # Slack Search
            QuickMatchRule(
                name="slack_search",
                patterns=[
                    r"slack\s*(에서|에|을|를)?\s*(검색|찾|search)",
                    r"(in|from|on)\s+slack",
                    r"슬랙\s*(에서|에)?\s*(검색|찾)",
                ],
                intent_type=IntentType.TOOL_SEARCH,
                tool_name="slack_search",
                confidence=0.95,
            ),

            # Notion Search
            QuickMatchRule(
                name="notion_search",
                patterns=[
                    r"notion\s*(에서|에|을|를)?\s*(검색|찾|search)",
                    r"(in|from|on)\s+notion",
                    r"노션\s*(에서|에)?\s*(검색|찾)",
                ],
                intent_type=IntentType.TOOL_SEARCH,
                tool_name="notion_search",
                confidence=0.95,
            ),

            # Conversation - Greetings
            QuickMatchRule(
                name="greeting",
                patterns=[
                    r"^(안녕|하이|헬로우?|hi|hello|hey)\s*[!.?]?$",
                    r"^(좋은\s*(아침|저녁|오후))",
                    r"^good\s+(morning|afternoon|evening)",
                ],
                intent_type=IntentType.CONVERSATION,
                confidence=0.99,
            ),

            # Conversation - Thanks
            QuickMatchRule(
                name="thanks",
                patterns=[
                    r"^(고마워|감사|thanks?|thank\s*you)",
                    r"^(잘\s*했어|좋아|굿|great|good\s*job)",
                ],
                intent_type=IntentType.FEEDBACK,
                confidence=0.99,
            ),

            # Clarification
            QuickMatchRule(
                name="clarification",
                patterns=[
                    r"^(뭐라고|뭐야|뭔|what)\s*\??$",
                    r"다시\s*(말해|설명|알려)",
                    r"무슨\s*말이?야",
                    r"이해가?\s*안\s*(돼|가)",
                ],
                intent_type=IntentType.CLARIFICATION,
                confidence=0.95,
            ),

            # Summary Request
            QuickMatchRule(
                name="summary",
                patterns=[
                    r"요약\s*(해|좀)",
                    r"정리\s*(해|좀)",
                    r"summarize",
                    r"summary",
                ],
                intent_type=IntentType.RAG_SUMMARY,
                confidence=0.90,
            ),

            # Comparison Request
            QuickMatchRule(
                name="compare",
                patterns=[
                    r"비교\s*(해|좀)",
                    r"차이점|차이가",
                    r"compare",
                    r"difference",
                ],
                intent_type=IntentType.RAG_COMPARE,
                confidence=0.90,
            ),
        ]

    async def route(self, query: str) -> Intent:
        """
        쿼리를 분류하고 Intent 반환

        Args:
            query: 사용자 입력

        Returns:
            Intent: 분류된 인텐트
        """
        query = query.strip()

        if not query:
            return Intent(
                type=IntentType.UNKNOWN,
                confidence=0.0,
                original_query=query,
            )

        # 1단계: Quick Match (빠른 규칙 기반)
        quick_result = self._quick_match(query)
        if quick_result:
            logger.debug(f"Quick match: {quick_result.type.value} for '{query[:30]}...'")
            return quick_result

        # 2단계: LLM 분류 (복잡한 케이스)
        if self.config.quick_match_only:
            # Quick match only 모드면 기본값 반환
            return Intent(
                type=IntentType.RAG_QA,  # 기본: RAG Q&A
                confidence=0.5,
                original_query=query,
                processed_query=query,
            )

        if self.config.use_llm_fallback:
            llm_result = await self._llm_classify(query)
            if llm_result:
                logger.debug(f"LLM classified: {llm_result.type.value} for '{query[:30]}...'")
                return llm_result

        # Fallback: RAG Q&A
        return Intent(
            type=IntentType.RAG_QA,
            confidence=0.5,
            original_query=query,
            processed_query=query,
            reasoning="No pattern matched, defaulting to RAG Q&A",
        )

    def _quick_match(self, query: str) -> Optional[Intent]:
        """
        빠른 규칙 기반 매칭

        명확한 패턴은 LLM 호출 없이 바로 분류
        """
        query_lower = query.lower()

        for rule in self._quick_rules:
            for pattern in rule.patterns:
                match = re.search(pattern, query_lower)
                if match:
                    # 파라미터 추출
                    params = {}
                    if rule.extract_params:
                        params = rule.extract_params(query, match)

                    return Intent(
                        type=rule.intent_type,
                        confidence=rule.confidence,
                        original_query=query,
                        processed_query=query,
                        tool_name=rule.tool_name,
                        params=params,
                        reasoning=f"Matched rule: {rule.name}",
                        metadata={"match_rule": rule.name},
                    )

        return None

    async def _llm_classify(self, query: str) -> Optional[Intent]:
        """
        LLM 기반 분류

        복잡하거나 애매한 쿼리를 LLM으로 분류
        """
        await self._ensure_llm_initialized()

        prompt = self.CLASSIFICATION_PROMPT.format(query=query)

        try:
            from agentic_core.llm import LLMResponse

            response: LLMResponse = await self._llm_gateway.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )

            result = self._parse_llm_response(response.content, query)
            return result

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None

    def _parse_llm_response(self, response: str, original_query: str) -> Optional[Intent]:
        """LLM 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            # Intent 타입 매핑
            intent_str = data.get("intent", "UNKNOWN").upper()
            intent_type = self._map_intent_type(intent_str)

            # Sub-intents 파싱
            sub_intents = []
            for i, sub in enumerate(data.get("sub_intents", [])):
                sub_intents.append(SubIntent(
                    type=self._map_intent_type(sub.get("intent", "UNKNOWN")),
                    query=sub.get("query", ""),
                    tool_name=sub.get("tool_name"),
                    order=sub.get("order", i),
                ))

            # Multi-intent 감지
            if len(sub_intents) > 1:
                intent_type = IntentType.MULTI_INTENT

            return Intent(
                type=intent_type,
                confidence=float(data.get("confidence", 0.5)),
                original_query=original_query,
                processed_query=data.get("processed_query", original_query),
                tool_name=data.get("tool_name"),
                sub_intents=sub_intents,
                reasoning=data.get("reasoning", ""),
                metadata={"llm_response": data},
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None

    def _map_intent_type(self, intent_str: str) -> IntentType:
        """문자열을 IntentType으로 매핑"""
        mapping = {
            "TOOL_CREATE": IntentType.TOOL_CREATE,
            "TOOL_SEARCH": IntentType.TOOL_SEARCH,
            "TOOL_UPDATE": IntentType.TOOL_UPDATE,
            "TOOL_DELETE": IntentType.TOOL_DELETE,
            "RAG_QA": IntentType.RAG_QA,
            "RAG_SUMMARY": IntentType.RAG_SUMMARY,
            "RAG_COMPARE": IntentType.RAG_COMPARE,
            "CONVERSATION": IntentType.CONVERSATION,
            "CLARIFICATION": IntentType.CLARIFICATION,
            "FEEDBACK": IntentType.FEEDBACK,
            "MULTI_INTENT": IntentType.MULTI_INTENT,
        }
        return mapping.get(intent_str.upper(), IntentType.UNKNOWN)

    async def _ensure_llm_initialized(self):
        """LLM Gateway 초기화"""
        if self._llm_gateway is None:
            from agentic_core.llm import LLMGateway, GatewayConfig

            config = GatewayConfig(
                default_model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                timeout=self.config.llm_timeout,
            )
            self._llm_gateway = LLMGateway(config)

    def add_rule(self, rule: QuickMatchRule):
        """커스텀 규칙 추가"""
        self._quick_rules.append(rule)

    def remove_rule(self, name: str):
        """규칙 제거"""
        self._quick_rules = [r for r in self._quick_rules if r.name != name]

    @property
    def rules(self) -> List[QuickMatchRule]:
        """등록된 규칙 목록"""
        return self._quick_rules.copy()
