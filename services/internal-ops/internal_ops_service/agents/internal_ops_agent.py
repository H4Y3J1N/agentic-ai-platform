"""
Internal Operations Agent - Unified orchestrator for Notion + Slack + Task Management

Tool Registry 기반으로 리팩토링:
- Tool 인스턴스를 Registry에서 관리
- 설정 파일로 Tool 활성화/비활성화 가능
- 새 Tool 추가 시 Agent 코드 수정 불필요
"""

from typing import List, Dict, Any, Optional, AsyncIterator
import os
import re
import logging
import sys
from pathlib import Path

# Add core package to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))

from agentic_core.observability import Tracer, get_tracer
from agentic_core.llm import LLMGateway, GatewayConfig, LLMResponse
from agentic_core.tools import ToolRegistry, ToolConfig, ToolCapability
from agentic_core.routing import IntentRouter, IntentRouterConfig, Intent, IntentType

logger = logging.getLogger(__name__)

# Default model - Gemini
DEFAULT_MODEL = "gemini/gemini-1.5-flash"


class InternalOpsAgent:
    """
    Unified agent that orchestrates:
    - Notion search (via NotionSearchTool)
    - Slack search (via SlackSearchTool)
    - Task creation (via TaskCreationTool)
    - Knowledge Q&A (via KnowledgeAgent for complex questions)
    """

    NAME = "internal_ops_agent"
    DESCRIPTION = "Unified assistant for knowledge search, Slack search, and task management"
    CAPABILITIES = [
        "notion_search",
        "slack_search",
        "unified_search",
        "task_creation",
        "knowledge_qa"
    ]

    def __init__(
        self,
        config: dict = None,
        tool_registry: Optional[ToolRegistry] = None,
        intent_router: Optional[IntentRouter] = None
    ):
        self.config = config or {}

        # Tool Registry - 외부 주입 또는 자체 생성
        self._tool_registry = tool_registry or ToolRegistry()
        self._tools_registered = False

        # Intent Router - LLM 기반 인텐트 분류
        router_config = IntentRouterConfig(
            llm_model=self.config.get("model", DEFAULT_MODEL),
            use_llm_fallback=self.config.get("use_llm_intent", True),
            quick_match_only=self.config.get("quick_match_only", False),
        )
        self._intent_router = intent_router or IntentRouter(router_config)

        # Lazy-loaded agents
        self._knowledge_agent = None
        self._llm_gateway: Optional[LLMGateway] = None

        # Initialize Langfuse tracer
        self._tracer: Tracer = get_tracer()

        # LLM model configuration
        self._model = self.config.get("model", DEFAULT_MODEL)

        # Tool configurations from config
        self._tool_configs = self.config.get("tools", {})

        self.system_prompt = """You are an Internal Operations Assistant.
You help team members:
1. Find information across Notion documentation and Slack messages
2. Create tasks linked to projects
3. Answer questions with citations from internal knowledge bases

Guidelines:
- When searching, combine results from both Notion and Slack when relevant
- Always cite your sources with [N1], [N2] for Notion and [S1], [S2] for Slack
- When creating tasks, confirm the details before proceeding
- If information is uncertain, say so clearly
- Never make up information that isn't in the retrieved context"""

    async def _ensure_initialized(self):
        """Lazy initialization of tools and clients"""
        # Register tools to registry
        if not self._tools_registered:
            await self._register_tools()
            self._tools_registered = True

        if self._knowledge_agent is None:
            from .knowledge_agent import KnowledgeAgent
            self._knowledge_agent = KnowledgeAgent(self.config)

        if self._llm_gateway is None:
            gateway_config = GatewayConfig(
                default_model=self._model,
                temperature=0.3,
                max_retries=2,
                timeout=60.0
            )
            self._llm_gateway = LLMGateway(gateway_config)
            logger.info(f"LLM Gateway initialized with model: {self._model}")

    async def _register_tools(self):
        """Register tools to the registry"""
        from ..tools import NotionSearchTool, SlackSearchTool, TaskCreationTool

        # Apply configurations from config
        tool_configs = {}
        for tool_name, tool_cfg in self._tool_configs.items():
            tool_configs[tool_name] = ToolConfig(**tool_cfg) if isinstance(tool_cfg, dict) else tool_cfg

        self._tool_registry.configure(self._tool_configs)

        # Register tools with legacy config support
        legacy_config = {
            "persist_dir": self.config.get("persist_dir"),
            "collection_name": self.config.get("collection_name"),
            "embedding_model": self.config.get("embedding_model"),
        }

        # Notion Search Tool
        if self._tool_configs.get("notion_search", {}).get("enabled", True):
            notion_tool = NotionSearchTool(
                config=tool_configs.get("notion_search"),
                legacy_config={**legacy_config, "collection_name": "internal_ops_notion"}
            )
            self._tool_registry.register(notion_tool, name="notion_search")

        # Slack Search Tool
        if self._tool_configs.get("slack_search", {}).get("enabled", True):
            slack_tool = SlackSearchTool(
                config=tool_configs.get("slack_search"),
                legacy_config={**legacy_config, "collection_name": "internal_ops_slack"}
            )
            self._tool_registry.register(slack_tool, name="slack_search")

        # Task Creation Tool
        if self._tool_configs.get("task_creation", {}).get("enabled", True):
            task_tool = TaskCreationTool(
                config=tool_configs.get("task_creation"),
                legacy_config=self.config
            )
            self._tool_registry.register(task_tool, name="task_creation")

        logger.info(f"Registered tools: {self._tool_registry.available_tools}")

    @property
    def tool_registry(self) -> ToolRegistry:
        """Get the tool registry"""
        return self._tool_registry

    def get_tool(self, name: str):
        """Get a tool by name"""
        return self._tool_registry.get_optional(name)

    async def _detect_intent(self, task: str) -> Intent:
        """
        Detect user intent using IntentRouter.

        Uses 2-stage routing:
        1. Quick match for obvious patterns (regex)
        2. LLM classification for complex cases

        Returns:
            Intent object with type, confidence, tool_name, etc.
        """
        return await self._intent_router.route(task)

    def _intent_to_legacy(self, intent: Intent) -> str:
        """
        Intent를 레거시 문자열 포맷으로 변환 (하위 호환성)
        """
        if intent.type == IntentType.TOOL_CREATE:
            return "create_task"
        elif intent.type == IntentType.TOOL_SEARCH:
            if intent.tool_name == "slack_search":
                return "search_slack"
            elif intent.tool_name == "notion_search":
                return "search_notion"
            return "unified_search"
        elif intent.type in (IntentType.RAG_QA, IntentType.RAG_SUMMARY, IntentType.RAG_COMPARE):
            return "ask"
        elif intent.type in (IntentType.CONVERSATION, IntentType.FEEDBACK, IntentType.CLARIFICATION):
            return "conversation"
        elif intent.type == IntentType.MULTI_INTENT:
            return "multi"
        return "ask"

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute task based on detected intent.

        Uses IntentRouter for intelligent routing:
        - TOOL intents → Tool execution
        - RAG intents → KnowledgeAgent
        - Conversation intents → Simple response
        - Multi-intent → Sequential execution

        Args:
            task: Natural language request
            context: Optional context (user_id, domain, etc.)

        Returns:
            Response string
        """
        await self._ensure_initialized()

        # Intent detection using LLM-based router
        intent = await self._detect_intent(task)
        legacy_intent = self._intent_to_legacy(intent)

        logger.info(
            f"Detected intent: {intent.type.value} "
            f"(confidence: {intent.confidence:.2f}, tool: {intent.tool_name}) "
            f"for task: {task[:50]}..."
        )

        # Start Langfuse trace
        user_id = context.get("user_id") if context else None
        session_id = context.get("session_id") if context else None

        async with self._tracer.trace(
            name="internal_ops_agent",
            user_id=user_id,
            session_id=session_id,
            metadata={
                "intent_type": intent.type.value,
                "intent_confidence": intent.confidence,
                "intent_tool": intent.tool_name,
                "task_preview": task[:100],
                "reasoning": intent.reasoning,
            },
            tags=["internal_ops", intent.type.value]
        ) as trace_ctx:
            try:
                # Handle based on intent type
                if intent.type == IntentType.TOOL_CREATE:
                    trace_ctx.event("intent_detected", {"type": "tool_create"})
                    result = await self._handle_task_creation(task, context)

                elif intent.type == IntentType.TOOL_SEARCH:
                    trace_ctx.event("intent_detected", {"type": "tool_search", "tool": intent.tool_name})
                    if intent.tool_name == "slack_search":
                        result = await self._handle_slack_search(task, context)
                    elif intent.tool_name == "notion_search":
                        result = await self._handle_notion_search(task, context)
                    else:
                        result = await self._handle_unified_search(task, context)

                elif intent.type in (IntentType.RAG_QA, IntentType.RAG_SUMMARY, IntentType.RAG_COMPARE):
                    trace_ctx.event("intent_detected", {"type": "rag", "subtype": intent.type.value})
                    result = await self._handle_knowledge_qa(task, context)

                elif intent.type in (IntentType.CONVERSATION, IntentType.FEEDBACK):
                    trace_ctx.event("intent_detected", {"type": "conversation"})
                    result = await self._handle_conversation(task, intent)

                elif intent.type == IntentType.CLARIFICATION:
                    trace_ctx.event("intent_detected", {"type": "clarification"})
                    result = await self._handle_clarification(task, context)

                elif intent.type == IntentType.MULTI_INTENT:
                    trace_ctx.event("intent_detected", {"type": "multi_intent", "count": len(intent.sub_intents)})
                    result = await self._handle_multi_intent(task, intent, context)

                else:  # UNKNOWN or fallback
                    trace_ctx.event("intent_detected", {"type": "fallback"})
                    result = await self._handle_knowledge_qa(task, context)

                trace_ctx.log_output(result)
                return result

            except Exception as e:
                trace_ctx.log_error(str(e))
                raise

    async def execute_stream(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Streaming execution.

        Args:
            task: Natural language request
            context: Optional context

        Yields:
            Response chunks
        """
        await self._ensure_initialized()

        intent = self._detect_intent(task)
        logger.info(f"Detected intent (stream): {intent}")

        if intent == "create_task":
            result = await self._handle_task_creation(task, context)
            yield result

        elif intent == "search_slack":
            async for chunk in self._handle_slack_search_stream(task, context):
                yield chunk

        elif intent == "search_notion":
            # Delegate to KnowledgeAgent streaming
            async for chunk in self._knowledge_agent.execute_stream(task, context):
                yield chunk

        elif intent == "unified_search":
            async for chunk in self._handle_unified_search_stream(task, context):
                yield chunk

        else:  # "ask"
            async for chunk in self._knowledge_agent.execute_stream(task, context):
                yield chunk

    async def _handle_task_creation(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Handle task creation requests"""
        # Extract task title and project from natural language
        # Look for quoted strings for title
        title_match = re.search(r'["\']([^"\']+)["\']', task)

        # Look for project reference
        project_patterns = [
            r'(for|in|to)\s+project\s+["\']?([^"\']+)["\']?',
            r'프로젝트\s+["\']?([^"\']+)["\']?\s*(에|의|에서)?',
            r'["\']?([^"\']+)["\']?\s+프로젝트\s*(에|의|에서)?',
        ]

        project_name = None
        for pattern in project_patterns:
            match = re.search(pattern, task, re.I)
            if match:
                project_name = match.group(2) if match.lastindex >= 2 else match.group(1)
                break

        if not title_match:
            return """태스크를 생성하려면 제목을 따옴표로 감싸주세요.

예시:
- "계약서 검토" 태스크를 프로젝트 A에 추가해줘
- Create task "Review contract" for project "Japan"

프로젝트 목록을 보려면 "프로젝트 목록 보여줘"라고 말씀해주세요."""

        title = title_match.group(1)

        try:
            task_tool = self._tool_registry.get("task_creation")
            result = await task_tool.create_task(
                title=title,
                project_name=project_name
            )

            response = f"**태스크가 생성되었습니다**\n\n"
            response += f"- 제목: {result['title']}\n"
            response += f"- URL: {result['url']}\n"

            if result.get('project_title'):
                response += f"- 연결된 프로젝트: {result['project_title']}\n"
            elif project_name:
                response += f"- 프로젝트 '{project_name}'을(를) 찾지 못해 연결하지 않았습니다.\n"

            return response

        except Exception as e:
            logger.error(f"Task creation failed: {e}")
            return f"태스크 생성에 실패했습니다: {e}"

    async def _handle_slack_search(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Handle Slack-specific search"""
        # Remove Slack-specific keywords from query
        query = re.sub(
            r'(slack\s*(에서|에|을|를)?|슬랙\s*(에서|에)?)\s*(검색|찾아?|search)',
            '',
            task,
            flags=re.I
        ).strip()

        if not query:
            query = task

        try:
            slack_tool = self._tool_registry.get("slack_search")
            results = await slack_tool.search(query, top_k=5)

            if not results:
                return f"Slack에서 '{query}'에 대한 결과를 찾지 못했습니다."

            formatted = self._format_slack_results(results)
            return await self._generate_search_response(query, formatted, source="Slack")

        except Exception as e:
            logger.error(f"Slack search failed: {e}")
            return f"Slack 검색 중 오류가 발생했습니다: {e}"

    async def _handle_slack_search_stream(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> AsyncIterator[str]:
        """Streaming Slack search"""
        query = re.sub(
            r'(slack\s*(에서|에|을|를)?|슬랙\s*(에서|에)?)\s*(검색|찾아?|search)',
            '',
            task,
            flags=re.I
        ).strip() or task

        yield "Slack에서 검색 중...\n\n"

        try:
            slack_tool = self._tool_registry.get("slack_search")
            results = await slack_tool.search(query, top_k=5)

            if not results:
                yield f"'{query}'에 대한 결과를 찾지 못했습니다."
                return

            formatted = self._format_slack_results(results)

            async for chunk in self._stream_search_response(query, formatted, source="Slack"):
                yield chunk

        except Exception as e:
            yield f"검색 중 오류 발생: {e}"

    async def _handle_notion_search(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Handle Notion-specific search - delegate to KnowledgeAgent"""
        return await self._knowledge_agent.execute(task, context)

    async def _handle_unified_search(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Handle unified search across Notion and Slack"""
        # Clean query
        query = re.sub(r'(search|find|look for|검색|찾아)', '', task, flags=re.I).strip()
        if not query:
            query = task

        # Search both sources in parallel
        notion_results = []
        slack_results = []

        notion_tool = self._tool_registry.get_optional("notion_search")
        slack_tool = self._tool_registry.get_optional("slack_search")

        if notion_tool:
            try:
                notion_results = await notion_tool.search(query, top_k=3)
            except Exception as e:
                logger.warning(f"Notion search failed: {e}")

        if slack_tool:
            try:
                slack_results = await slack_tool.search(query, top_k=3)
            except Exception as e:
                logger.warning(f"Slack search failed: {e}")

        if not notion_results and not slack_results:
            return f"'{query}'에 대한 결과를 Notion과 Slack에서 찾지 못했습니다."

        # Format combined results
        combined = self._format_combined_results(notion_results, slack_results)
        return await self._generate_search_response(query, combined, source="Notion & Slack")

    async def _handle_unified_search_stream(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> AsyncIterator[str]:
        """Streaming unified search"""
        query = re.sub(r'(search|find|look for|검색|찾아)', '', task, flags=re.I).strip() or task

        yield "Notion과 Slack에서 검색 중...\n\n"

        notion_results = []
        slack_results = []

        notion_tool = self._tool_registry.get_optional("notion_search")
        slack_tool = self._tool_registry.get_optional("slack_search")

        if notion_tool:
            try:
                notion_results = await notion_tool.search(query, top_k=3)
            except Exception as e:
                logger.warning(f"Notion search failed: {e}")

        if slack_tool:
            try:
                slack_results = await slack_tool.search(query, top_k=3)
            except Exception as e:
                logger.warning(f"Slack search failed: {e}")

        if not notion_results and not slack_results:
            yield f"'{query}'에 대한 결과를 찾지 못했습니다."
            return

        combined = self._format_combined_results(notion_results, slack_results)

        async for chunk in self._stream_search_response(query, combined, source="Notion & Slack"):
            yield chunk

    async def _handle_knowledge_qa(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Delegate to KnowledgeAgent for Q&A"""
        return await self._knowledge_agent.execute(task, context)

    async def _handle_conversation(
        self,
        task: str,
        intent: Intent
    ) -> str:
        """
        Handle conversational intents (greetings, thanks, etc.)

        No RAG or tool calls needed - simple response generation.
        """
        task_lower = task.lower().strip()

        # Greetings
        if any(g in task_lower for g in ["안녕", "하이", "hello", "hi", "hey"]):
            return "안녕하세요! 무엇을 도와드릴까요? 📋\n\n" \
                   "다음과 같은 요청을 처리할 수 있습니다:\n" \
                   "- 프로젝트/업무 진행상황 질문\n" \
                   "- Notion/Slack 검색\n" \
                   "- 태스크 생성"

        # Thanks/Feedback
        if any(t in task_lower for t in ["고마워", "감사", "thanks", "thank"]):
            return "천만에요! 다른 도움이 필요하시면 말씀해주세요. 😊"

        if any(t in task_lower for t in ["잘했어", "좋아", "굿", "great", "good"]):
            return "감사합니다! 더 필요한 게 있으시면 말씀해주세요."

        # Default conversational response
        return "네, 무엇을 도와드릴까요?"

    async def _handle_clarification(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Handle clarification requests.

        User asked for clarification about previous response.
        """
        # TODO: Implement conversation history to provide context
        return "죄송합니다, 어떤 부분을 더 자세히 설명해 드릴까요?\n\n" \
               "구체적으로 질문해 주시면 더 정확한 답변을 드릴 수 있습니다."

    async def _handle_multi_intent(
        self,
        task: str,
        intent: Intent,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Handle multi-intent requests.

        Executes sub-intents in order and combines results.

        Example: "프로젝트 검색하고 태스크도 만들어줘"
        → 1) Search for project
        → 2) Create task
        """
        if not intent.sub_intents:
            # Fallback to knowledge QA
            return await self._handle_knowledge_qa(task, context)

        results = []
        execution_order = intent.get_execution_order()

        for i, sub_intent in enumerate(execution_order, 1):
            results.append(f"### 요청 {i}: {sub_intent.type.value}\n")

            try:
                if sub_intent.type == IntentType.TOOL_CREATE:
                    result = await self._handle_task_creation(sub_intent.original_query, context)
                elif sub_intent.type == IntentType.TOOL_SEARCH:
                    if sub_intent.tool_name == "slack_search":
                        result = await self._handle_slack_search(sub_intent.original_query, context)
                    else:
                        result = await self._handle_unified_search(sub_intent.original_query, context)
                else:
                    result = await self._handle_knowledge_qa(sub_intent.original_query, context)

                results.append(result)

            except Exception as e:
                results.append(f"오류 발생: {e}")

            results.append("\n---\n")

        return "\n".join(results)

    def _format_slack_results(self, results: List[Dict]) -> str:
        """Format Slack search results with citations"""
        sections = []

        for i, msg in enumerate(results, 1):
            channel = msg.get('channel_name', 'unknown')
            user = msg.get('user_name', 'Unknown')
            content = msg.get('content', '')[:500]
            url = msg.get('url', '')
            score = msg.get('relevance_score', 0)

            section = f"[S{i}] #{channel} - {user} (relevance: {score:.2f})"
            if url:
                section += f"\n    URL: {url}"
            section += f"\n\n{content}"

            sections.append(section)

        return "\n\n---\n\n".join(sections)

    def _format_combined_results(
        self,
        notion_results: List[Dict],
        slack_results: List[Dict]
    ) -> str:
        """Format combined search results"""
        sections = []

        if notion_results:
            sections.append("## Notion Documents\n")
            for i, doc in enumerate(notion_results, 1):
                title = doc.get('title', 'Untitled')
                url = doc.get('url', '')
                content = doc.get('content', '')[:400]
                score = doc.get('relevance_score', 0)

                section = f"[N{i}] {title} (relevance: {score:.2f})"
                if url:
                    section += f"\n    URL: {url}"
                section += f"\n\n{content}..."

                sections.append(section)

        if slack_results:
            sections.append("\n## Slack Messages\n")
            for i, msg in enumerate(slack_results, 1):
                channel = msg.get('channel_name', 'unknown')
                user = msg.get('user_name', 'Unknown')
                content = msg.get('content', '')[:300]
                url = msg.get('url', '')
                score = msg.get('relevance_score', 0)

                section = f"[S{i}] #{channel} - {user} (relevance: {score:.2f})"
                if url:
                    section += f"\n    URL: {url}"
                section += f"\n\n{content}..."

                sections.append(section)

        return "\n".join(sections)

    async def _generate_search_response(
        self,
        question: str,
        context: str,
        source: str = "internal"
    ) -> str:
        """Generate response from search results using LLM Gateway"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Question: {question}

Retrieved Context (from {source}):
{context}

Based on the retrieved context, provide a helpful answer. Include citations like [N1], [S1].
"""}
        ]

        # Use LLM Gateway (supports Gemini, OpenAI, etc.)
        response: LLMResponse = await self._llm_gateway.chat(
            messages=messages,
            temperature=0.3
        )

        result = response.content

        # Log LLM call to Langfuse
        usage = {
            "input": response.prompt_tokens,
            "output": response.completion_tokens,
            "total": response.total_tokens
        } if response.total_tokens else None

        self._tracer.log_llm_call(
            name=f"search_response_{source}",
            model=response.model,
            messages=messages,
            response=result,
            usage=usage,
            model_parameters={"temperature": 0.3}
        )

        return result

    async def _stream_search_response(
        self,
        question: str,
        context: str,
        source: str = "internal"
    ) -> AsyncIterator[str]:
        """Stream search response using LLM Gateway"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Question: {question}

Retrieved Context (from {source}):
{context}

Based on the retrieved context, provide a helpful answer. Include citations like [N1], [S1].
"""}
        ]

        # Use LLM Gateway streaming (supports Gemini, OpenAI, etc.)
        async for chunk in self._llm_gateway.chat_stream(
            messages=messages,
            temperature=0.3
        ):
            yield chunk
