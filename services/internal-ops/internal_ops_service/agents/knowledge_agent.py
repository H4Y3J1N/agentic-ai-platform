"""
Knowledge Agent - Answers questions using RAG over internal knowledge bases
"""

from typing import List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """Agent for answering questions using RAG over internal knowledge bases"""

    NAME = "knowledge_agent"
    DESCRIPTION = "Answers questions about internal documentation using semantic search"
    CAPABILITIES = [
        "notion_search",
        "knowledge_qa",
        "document_retrieval"
    ]

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._search_tool = None
        self._openai_client = None

        self.system_prompt = """You are an Internal Operations Knowledge Assistant.
Your role is to help team members find information from our internal documentation,
primarily stored in Notion.

When answering questions:
1. Always cite your sources with Notion page titles and URLs
2. If information is uncertain or incomplete, say so clearly
3. Suggest related pages the user might want to explore
4. For technical questions, include relevant code snippets if available

If the retrieved context doesn't contain enough information to answer the question,
clearly state that and suggest how the user might find the answer.

Never make up information that isn't in the retrieved context."""

    async def _ensure_initialized(self):
        """Lazy initialization"""
        if self._search_tool is None:
            from ..tools import NotionSearchTool
            self._search_tool = NotionSearchTool(self.config)

        if self._openai_client is None:
            try:
                import openai
            except ImportError:
                raise ImportError("openai is required")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")

            self._openai_client = openai.AsyncOpenAI(api_key=api_key)

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute knowledge query.

        Args:
            task: The question or task to answer
            context: Optional context (user_id, domain, etc.)

        Returns:
            AI-generated answer with citations
        """
        await self._ensure_initialized()

        context = context or {}

        # 1. RAG retrieval
        try:
            relevant_docs = await self._search_tool.search(
                query=task,
                top_k=5
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._error_response(str(e))

        # 2. Check if we have relevant context
        if not relevant_docs:
            return self._no_results_response(task)

        # 3. Build context with citations
        context_with_citations = self._format_docs_with_citations(relevant_docs)

        # 4. Generate answer using LLM
        try:
            response = await self._generate_answer(task, context_with_citations)
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._error_response(str(e))

    async def execute_stream(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Execute knowledge query with streaming response.

        Args:
            task: The question or task to answer
            context: Optional context

        Yields:
            Chunks of the AI-generated answer
        """
        await self._ensure_initialized()

        # 1. RAG retrieval
        try:
            relevant_docs = await self._search_tool.search(
                query=task,
                top_k=5
            )
        except Exception as e:
            yield f"Error searching: {e}"
            return

        if not relevant_docs:
            yield self._no_results_response(task)
            return

        # 2. Build context
        context_with_citations = self._format_docs_with_citations(relevant_docs)

        # 3. Stream LLM response
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Question: {task}

Retrieved Context:
{context_with_citations}

Please provide a helpful answer based on the retrieved context. Include citations.
"""}
        ]

        try:
            model = self.config.get("model", "gpt-4o-mini")

            stream = await self._openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"\n\nError generating response: {e}"

    async def _generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """Generate answer using LLM"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Question: {question}

Retrieved Context:
{context}

Please provide a helpful answer based on the retrieved context. Include citations.
"""}
        ]

        model = self.config.get("model", "gpt-4o-mini")

        response = await self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )

        return response.choices[0].message.content

    def _format_docs_with_citations(self, docs: List[Dict]) -> str:
        """Format documents with citation information"""
        formatted = []

        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Untitled")
            url = doc.get("url", "")
            content = doc.get("content", "")
            score = doc.get("relevance_score", 0)

            citation = f"[{i}] {title} (relevance: {score:.2f})"
            if url:
                citation += f"\n    URL: {url}"

            formatted.append(f"{citation}\n\n{content}\n")

        return "\n---\n".join(formatted)

    def _no_results_response(self, query: str) -> str:
        """Response when no relevant documents found"""
        return f"""I couldn't find relevant information in our Notion workspace for: "{query}"

This could mean:
1. The information hasn't been documented yet
2. It might be under a different topic or title
3. The content might not be indexed yet

Suggestions:
- Try rephrasing your question with different keywords
- Check with the relevant team directly
- Consider creating documentation if this is a common question
"""

    def _error_response(self, error: str) -> str:
        """Response when an error occurs"""
        return f"""I encountered an error while searching for information.

Error: {error}

Please try again or contact support if the issue persists.
"""
