"""
Single-Shot RAG Strategy

기본 RAG 파이프라인: 검색 → 답변 생성
가장 단순하고 빠른 전략
"""

import time
from typing import Optional, AsyncIterator

from .base import (
    RAGStrategy,
    RAGStrategyType,
    RAGStrategyConfig,
    RAGResult,
    Document,
    SearchFn,
    RetrievalQuality,
)


class SingleShotRAG(RAGStrategy):
    """
    Single-Shot RAG Strategy

    가장 기본적인 RAG 파이프라인:
    1. 검색 실행
    2. 컨텍스트 구성
    3. LLM으로 답변 생성

    장점: 빠름, 단순함
    단점: 검색 품질이 낮으면 답변 품질도 낮음
    """

    strategy_type = RAGStrategyType.SINGLE_SHOT

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite sources using [1], [2], etc.
4. Be concise and accurate
5. Never make up information not in the context"""

    ANSWER_PROMPT = """Based on the following context, answer the user's question.

Context:
{context}

Question: {question}

Provide a helpful, accurate answer based on the context. Include citations like [1], [2]."""

    async def execute(
        self,
        query: str,
        search_fn: SearchFn,
        **kwargs
    ) -> RAGResult:
        """
        Single-shot RAG 실행

        Args:
            query: 사용자 질문
            search_fn: 검색 함수

        Returns:
            RAGResult: 실행 결과
        """
        start_time = time.time()

        result = RAGResult(
            answer="",
            strategy_type=self.strategy_type
        )

        # Step 1: 검색
        result.add_step("search", f"Searching for: {query[:50]}...")
        self._log(f"Searching: {query}")

        try:
            docs = await search_fn(query, self.config.top_k)
        except Exception as e:
            result.answer = f"Search failed: {e}"
            result.add_step("error", f"Search error: {e}")
            return result

        # 문서 변환 (필요시)
        documents = self._normalize_documents(docs)
        result.sources = documents

        # Step 2: 검색 결과 평가
        if not documents:
            result.answer = self._no_results_response(query)
            result.retrieval_quality = RetrievalQuality.POOR
            result.add_step("no_results", "No documents found")
            return result

        # 품질 평가 (단순)
        avg_score = sum(d.score for d in documents) / len(documents)
        if avg_score >= 0.7:
            result.retrieval_quality = RetrievalQuality.EXCELLENT
        elif avg_score >= 0.5:
            result.retrieval_quality = RetrievalQuality.GOOD
        elif avg_score >= 0.3:
            result.retrieval_quality = RetrievalQuality.AMBIGUOUS
        else:
            result.retrieval_quality = RetrievalQuality.POOR

        result.add_step(
            "retrieval_complete",
            f"Retrieved {len(documents)} documents (avg score: {avg_score:.3f})"
        )

        # Step 3: 컨텍스트 구성
        context = self._format_docs_for_context(documents)

        # Step 4: LLM으로 답변 생성
        result.add_step("generation", "Generating answer with LLM")
        self._log("Generating answer...")

        prompt = self.ANSWER_PROMPT.format(
            context=context,
            question=query
        )

        try:
            answer = await self._call_llm(prompt, self.SYSTEM_PROMPT)
            result.answer = answer
            result.total_llm_calls = 1
        except Exception as e:
            result.answer = f"Generation failed: {e}"
            result.add_step("error", f"LLM error: {e}")
            return result

        result.add_step("complete", "Answer generated successfully")

        # 실행 시간
        result.execution_time_ms = (time.time() - start_time) * 1000

        self._log(f"Completed in {result.execution_time_ms:.0f}ms")

        return result

    async def execute_stream(
        self,
        query: str,
        search_fn: SearchFn,
        **kwargs
    ) -> AsyncIterator[str]:
        """스트리밍 실행"""
        await self._ensure_llm()

        # Step 1: 검색
        try:
            docs = await search_fn(query, self.config.top_k)
        except Exception as e:
            yield f"Search failed: {e}"
            return

        documents = self._normalize_documents(docs)

        if not documents:
            yield self._no_results_response(query)
            return

        # Step 2: 컨텍스트 구성
        context = self._format_docs_for_context(documents)

        # Step 3: 스트리밍 답변 생성
        prompt = self.ANSWER_PROMPT.format(
            context=context,
            question=query
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            from litellm import acompletion

            response = await acompletion(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"\n\nGeneration error: {e}"

    def _normalize_documents(self, docs) -> list[Document]:
        """다양한 포맷의 문서를 Document로 변환"""
        documents = []

        for doc in docs:
            if isinstance(doc, Document):
                documents.append(doc)
            elif hasattr(doc, 'content'):
                # EnhancedSearchResult 등
                documents.append(Document(
                    id=getattr(doc, 'id', str(len(documents))),
                    content=doc.content,
                    score=getattr(doc, 'score', 0.0),
                    metadata=getattr(doc, 'metadata', {}),
                    rerank_score=getattr(doc, 'rerank_score', None)
                ))
            elif isinstance(doc, dict):
                documents.append(Document(
                    id=doc.get('id', str(len(documents))),
                    content=doc.get('content', ''),
                    score=doc.get('score', doc.get('relevance_score', 0.0)),
                    metadata=doc.get('metadata', {}),
                    rerank_score=doc.get('rerank_score')
                ))

        return documents

    def _no_results_response(self, query: str) -> str:
        """검색 결과 없을 때 응답"""
        return f"""I couldn't find relevant information for: "{query}"

This could mean:
- The information hasn't been documented yet
- Try different keywords or phrasing
- The topic might be under a different category

Please try rephrasing your question or ask about a related topic."""
