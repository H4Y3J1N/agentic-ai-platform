"""
Self-RAG Strategy

답변 생성 후 자기 검증을 수행하고, 문제가 있으면 수정하는 전략.

Paper: "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023)

Flow:
1. 검색 실행
2. 답변 생성
3. 자기 검증 (Faithfulness, Completeness, Relevance)
4. 검증 실패 시 → 추가 검색 또는 답변 수정
5. 최종 답변 반환
"""

import time
import json
import re
from typing import Optional, List

from .base import (
    RAGStrategy,
    RAGStrategyType,
    RAGStrategyConfig,
    RAGResult,
    Document,
    SearchFn,
    RetrievalQuality,
)


class SelfRAG(RAGStrategy):
    """
    Self-RAG Strategy

    답변을 생성한 후 자체적으로 검증하고, 필요시 수정합니다.

    검증 항목:
    1. Faithfulness: 답변이 컨텍스트에 근거하는가
    2. Completeness: 질문에 완전히 답했는가
    3. Relevance: 답변이 질문과 관련 있는가

    장점: 환각 감소, 답변 품질 향상
    단점: LLM 호출 증가 (검증용), 레이턴시 증가
    """

    strategy_type = RAGStrategyType.SELF_RAG

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

    # 자기 검증 프롬프트
    CRITIQUE_PROMPT = """Evaluate the following AI-generated answer for quality issues.

Question: {question}

Context (source documents):
{context}

Generated Answer:
{answer}

Evaluate the answer on these criteria:

1. **Faithfulness**: Does the answer only contain information from the context? Are there any hallucinations or made-up facts?
2. **Completeness**: Does the answer fully address the question? Is anything important missing?
3. **Relevance**: Is the answer on-topic and directly addresses what was asked?

Respond with a JSON object:
{{
    "faithfulness": {{
        "score": <0.0-1.0>,
        "issues": ["<list of faithfulness issues if any>"]
    }},
    "completeness": {{
        "score": <0.0-1.0>,
        "missing": ["<list of missing information if any>"]
    }},
    "relevance": {{
        "score": <0.0-1.0>,
        "issues": ["<list of relevance issues if any>"]
    }},
    "overall_score": <0.0-1.0>,
    "needs_revision": <true/false>,
    "revision_suggestions": ["<suggestions for improvement>"]
}}

Only respond with the JSON object."""

    # 답변 수정 프롬프트
    REVISION_PROMPT = """Revise the following answer based on the critique.

Question: {question}

Context:
{context}

Original Answer:
{answer}

Critique Issues:
{issues}

Revision Suggestions:
{suggestions}

Generate an improved answer that:
1. Fixes the identified issues
2. Only uses information from the context
3. Includes proper citations [1], [2]
4. Fully addresses the question

Revised Answer:"""

    # 추가 검색 프롬프트
    ADDITIONAL_SEARCH_PROMPT = """Based on the missing information identified, generate a search query to find the missing information.

Original Question: {question}

Missing Information:
{missing}

Generate a focused search query to find this missing information.
Respond with just the search query, nothing else."""

    async def execute(
        self,
        query: str,
        search_fn: SearchFn,
        **kwargs
    ) -> RAGResult:
        """
        Self-RAG 실행

        Args:
            query: 사용자 질문
            search_fn: 검색 함수

        Returns:
            RAGResult: 실행 결과
        """
        start_time = time.time()

        result = RAGResult(
            answer="",
            strategy_type=self.strategy_type,
            total_searches=0,
            total_llm_calls=0
        )

        # Step 1: 검색
        result.add_step("search", f"Searching for: {query[:50]}...")
        self._log(f"Searching: {query}")

        try:
            docs = await search_fn(query, self.config.top_k)
            result.total_searches += 1
        except Exception as e:
            result.answer = f"Search failed: {e}"
            result.add_step("error", f"Search error: {e}")
            return result

        documents = self._normalize_documents(docs)

        if not documents:
            result.answer = self._no_results_response(query)
            result.retrieval_quality = RetrievalQuality.POOR
            result.add_step("no_results", "No documents found")
            return result

        result.sources = documents
        context = self._format_docs_for_context(documents)

        # Step 2: 초기 답변 생성
        result.add_step("generation", "Generating initial answer...")
        self._log("Generating initial answer...")

        prompt = self.ANSWER_PROMPT.format(context=context, question=query)

        try:
            initial_answer = await self._call_llm(prompt, self.SYSTEM_PROMPT)
            result.total_llm_calls += 1
        except Exception as e:
            result.answer = f"Generation failed: {e}"
            result.add_step("error", f"LLM error: {e}")
            return result

        result.add_step("initial_answer", f"Initial answer generated ({len(initial_answer)} chars)")

        # Step 3: 자기 검증
        if not self.config.enable_self_critique:
            result.answer = initial_answer
            result.add_step("skip_critique", "Self-critique disabled")
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        result.add_step("critique", "Self-critiquing the answer...")
        self._log("Self-critiquing...")

        critique = await self._critique_answer(query, context, initial_answer, result)

        if critique is None:
            # 검증 실패시 원본 답변 반환
            result.answer = initial_answer
            result.add_step("critique_failed", "Critique failed, using initial answer")
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        overall_score = critique.get("overall_score", 0.5)
        result.confidence_score = overall_score

        result.add_step(
            "critique_complete",
            f"Critique score: {overall_score:.2f}, needs_revision: {critique.get('needs_revision', False)}"
        )

        # Step 4: 필요시 수정
        if critique.get("needs_revision", False) and overall_score < self.config.critique_threshold:
            # 누락된 정보가 있으면 추가 검색
            missing = critique.get("completeness", {}).get("missing", [])

            if missing and result.total_searches < self.config.max_retries + 1:
                result.add_step("additional_search", f"Searching for missing info: {missing}")
                additional_docs = await self._search_for_missing(query, missing, search_fn, result)

                if additional_docs:
                    # 기존 문서에 추가
                    documents.extend(additional_docs)
                    documents = self._deduplicate_docs(documents)
                    result.sources = documents[:self.config.top_k]
                    context = self._format_docs_for_context(result.sources)

            # 답변 수정
            result.add_step("revision", "Revising answer based on critique...")
            self._log("Revising answer...")

            issues = []
            for key in ["faithfulness", "completeness", "relevance"]:
                if key in critique:
                    issues.extend(critique[key].get("issues", []))
                    issues.extend(critique[key].get("missing", []))

            suggestions = critique.get("revision_suggestions", [])

            revised_answer = await self._revise_answer(
                query, context, initial_answer, issues, suggestions, result
            )

            if revised_answer:
                result.answer = revised_answer
                result.add_step("revision_complete", "Answer revised successfully")
            else:
                result.answer = initial_answer
                result.add_step("revision_failed", "Revision failed, using initial answer")

        else:
            result.answer = initial_answer
            result.add_step("no_revision_needed", f"Answer quality OK (score: {overall_score:.2f})")

        # 품질 등급 결정
        if overall_score >= 0.8:
            result.retrieval_quality = RetrievalQuality.EXCELLENT
        elif overall_score >= 0.6:
            result.retrieval_quality = RetrievalQuality.GOOD
        elif overall_score >= 0.4:
            result.retrieval_quality = RetrievalQuality.AMBIGUOUS
        else:
            result.retrieval_quality = RetrievalQuality.POOR

        result.add_step("complete", f"Completed with {result.total_llm_calls} LLM calls")
        result.execution_time_ms = (time.time() - start_time) * 1000

        self._log(f"Completed in {result.execution_time_ms:.0f}ms")

        return result

    async def _critique_answer(
        self,
        question: str,
        context: str,
        answer: str,
        result: RAGResult
    ) -> Optional[dict]:
        """답변 자기 검증"""
        prompt = self.CRITIQUE_PROMPT.format(
            question=question,
            context=context[:3000],  # 토큰 제한
            answer=answer
        )

        try:
            response = await self._call_llm(prompt)
            result.total_llm_calls += 1

            critique = self._parse_json_response(response)
            result.metadata["critique"] = critique
            return critique

        except Exception as e:
            self._log(f"Critique failed: {e}")
            return None

    async def _search_for_missing(
        self,
        original_query: str,
        missing: List[str],
        search_fn: SearchFn,
        result: RAGResult
    ) -> List[Document]:
        """누락된 정보 추가 검색"""
        # 누락된 정보로 검색 쿼리 생성
        prompt = self.ADDITIONAL_SEARCH_PROMPT.format(
            question=original_query,
            missing="\n".join(f"- {m}" for m in missing)
        )

        try:
            search_query = await self._call_llm(prompt)
            result.total_llm_calls += 1

            # 검색 실행
            docs = await search_fn(search_query.strip(), self.config.top_k // 2)
            result.total_searches += 1

            return self._normalize_documents(docs)

        except Exception as e:
            self._log(f"Additional search failed: {e}")
            return []

    async def _revise_answer(
        self,
        question: str,
        context: str,
        original_answer: str,
        issues: List[str],
        suggestions: List[str],
        result: RAGResult
    ) -> Optional[str]:
        """답변 수정"""
        prompt = self.REVISION_PROMPT.format(
            question=question,
            context=context[:3000],
            answer=original_answer,
            issues="\n".join(f"- {i}" for i in issues) if issues else "None identified",
            suggestions="\n".join(f"- {s}" for s in suggestions) if suggestions else "None"
        )

        try:
            revised = await self._call_llm(prompt, self.SYSTEM_PROMPT)
            result.total_llm_calls += 1
            return revised

        except Exception as e:
            self._log(f"Revision failed: {e}")
            return None

    def _normalize_documents(self, docs) -> List[Document]:
        """다양한 포맷의 문서를 Document로 변환"""
        documents = []

        for doc in docs:
            if isinstance(doc, Document):
                documents.append(doc)
            elif hasattr(doc, 'content'):
                documents.append(Document(
                    id=getattr(doc, 'id', str(len(documents))),
                    content=doc.content,
                    score=getattr(doc, 'score', 0.0),
                    metadata=getattr(doc, 'metadata', {}) or {},
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

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """중복 문서 제거"""
        seen = set()
        unique = []

        for doc in docs:
            if doc.id not in seen:
                seen.add(doc.id)
                unique.append(doc)

        unique.sort(key=lambda d: d.rerank_score or d.score, reverse=True)
        return unique

    def _parse_json_response(self, text: str) -> dict:
        """LLM 응답에서 JSON 추출"""
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _no_results_response(self, query: str) -> str:
        """검색 결과 없을 때 응답"""
        return f"""I couldn't find relevant information for: "{query}"

Please try:
- Rephrasing your question
- Using different keywords
- Asking about a more specific aspect"""
