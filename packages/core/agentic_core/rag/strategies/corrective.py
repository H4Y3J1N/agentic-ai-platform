"""
Corrective RAG (CRAG) Strategy

검색 결과 품질을 평가하고, 불충분하면 쿼리를 재작성하여 재검색하는 전략.

Paper: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)

Flow:
1. 검색 실행
2. 검색 결과 품질 평가 (LLM)
3. 품질이 낮으면 → 쿼리 재작성 → 재검색
4. 최종 답변 생성
"""

import time
import json
import re
from typing import Optional, List, AsyncIterator

from .base import (
    RAGStrategy,
    RAGStrategyType,
    RAGStrategyConfig,
    RAGResult,
    Document,
    SearchFn,
    RetrievalQuality,
)


class CorrectiveRAG(RAGStrategy):
    """
    Corrective RAG (CRAG) Strategy

    검색 결과를 평가하고, 품질이 낮으면 자동으로 재검색을 수행합니다.

    특징:
    - LLM으로 각 문서의 관련성 평가 ("relevant", "ambiguous", "irrelevant")
    - 관련 문서가 부족하면 쿼리 재작성 후 재검색
    - 최대 max_retries 횟수까지 재시도

    장점: 검색 품질 보장, 첫 검색 실패 시 자동 복구
    단점: LLM 호출 증가, 레이턴시 증가
    """

    strategy_type = RAGStrategyType.CORRECTIVE

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite sources using [1], [2], etc.
4. Be concise and accurate
5. Never make up information not in the context"""

    # 문서 관련성 평가 프롬프트
    RELEVANCE_EVAL_PROMPT = """Evaluate if the following document is relevant to answering the question.

Question: {question}

Document:
{document}

Is this document relevant to answering the question?
Respond with a JSON object:
{{
    "relevance": "relevant" | "ambiguous" | "irrelevant",
    "reason": "<brief explanation>",
    "key_info": "<key information from doc if relevant, else null>"
}}

Only respond with the JSON object, no other text."""

    # 쿼리 재작성 프롬프트
    QUERY_REWRITE_PROMPT = """The initial search for the following question didn't return enough relevant results.

Original Question: {question}

Previous search returned documents that were mostly irrelevant.

Generate 2-3 alternative search queries that might find better results.
Consider:
- Using synonyms or related terms
- Breaking down complex questions
- Focusing on key concepts

Respond with a JSON object:
{{
    "queries": ["query1", "query2", "query3"],
    "reasoning": "<why these queries might work better>"
}}

Only respond with the JSON object."""

    # 최종 답변 생성 프롬프트
    ANSWER_PROMPT = """Based on the following context, answer the user's question.

Context (filtered for relevance):
{context}

Question: {question}

Provide a helpful, accurate answer based on the context. Include citations like [1], [2].
If information is incomplete, acknowledge what's missing."""

    async def execute(
        self,
        query: str,
        search_fn: SearchFn,
        **kwargs
    ) -> RAGResult:
        """
        Corrective RAG 실행

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

        all_relevant_docs: List[Document] = []
        current_query = query
        retry_count = 0

        while retry_count <= self.config.max_retries:
            # Step 1: 검색
            result.add_step(
                "search",
                f"Search #{retry_count + 1}: {current_query[:50]}..."
            )
            self._log(f"Search #{retry_count + 1}: {current_query}")

            try:
                docs = await search_fn(current_query, self.config.top_k)
                result.total_searches += 1
            except Exception as e:
                result.answer = f"Search failed: {e}"
                result.add_step("error", f"Search error: {e}")
                break

            documents = self._normalize_documents(docs)

            if not documents:
                result.add_step("no_results", f"No results for query #{retry_count + 1}")
                retry_count += 1

                if retry_count <= self.config.max_retries:
                    # 쿼리 재작성
                    current_query = await self._rewrite_query(query, result)
                continue

            # Step 2: 문서 관련성 평가
            result.add_step("evaluate", f"Evaluating {len(documents)} documents...")
            self._log(f"Evaluating {len(documents)} documents")

            evaluated_docs = await self._evaluate_documents(query, documents, result)

            # 관련 문서 필터링
            relevant = [d for d in evaluated_docs if d.relevance_label == "relevant"]
            ambiguous = [d for d in evaluated_docs if d.relevance_label == "ambiguous"]

            result.add_step(
                "evaluation_complete",
                f"Found {len(relevant)} relevant, {len(ambiguous)} ambiguous, "
                f"{len(evaluated_docs) - len(relevant) - len(ambiguous)} irrelevant"
            )

            # 관련 문서 수집
            all_relevant_docs.extend(relevant)
            if len(ambiguous) > 0 and len(relevant) < 2:
                # 관련 문서가 부족하면 애매한 문서도 포함
                all_relevant_docs.extend(ambiguous[:2])

            # Step 3: 품질 평가
            relevance_ratio = len(relevant) / len(evaluated_docs) if evaluated_docs else 0

            if relevance_ratio >= self.config.quality_threshold:
                result.retrieval_quality = RetrievalQuality.EXCELLENT
                result.add_step("quality_check", f"Quality OK: {relevance_ratio:.1%} relevant")
                break
            elif relevance_ratio >= self.config.quality_threshold * 0.7:
                result.retrieval_quality = RetrievalQuality.GOOD
                result.add_step("quality_check", f"Quality acceptable: {relevance_ratio:.1%} relevant")
                break
            else:
                result.retrieval_quality = RetrievalQuality.AMBIGUOUS
                result.add_step(
                    "quality_check",
                    f"Quality low: {relevance_ratio:.1%} relevant, will retry"
                )

            retry_count += 1

            if retry_count <= self.config.max_retries:
                # 쿼리 재작성
                current_query = await self._rewrite_query(query, result)

        # Step 4: 최종 답변 생성
        if not all_relevant_docs:
            result.answer = self._no_results_response(query)
            result.retrieval_quality = RetrievalQuality.POOR
        else:
            # 중복 제거 및 정렬
            unique_docs = self._deduplicate_docs(all_relevant_docs)
            result.sources = unique_docs[:self.config.top_k]

            result.add_step("generation", f"Generating answer from {len(result.sources)} documents")
            self._log(f"Generating answer from {len(result.sources)} documents")

            context = self._format_docs_for_context(result.sources)
            prompt = self.ANSWER_PROMPT.format(context=context, question=query)

            try:
                answer = await self._call_llm(prompt, self.SYSTEM_PROMPT)
                result.answer = answer
                result.total_llm_calls += 1
            except Exception as e:
                result.answer = f"Generation failed: {e}"
                result.add_step("error", f"LLM error: {e}")

        result.add_step("complete", f"Completed with {result.total_searches} searches, {result.total_llm_calls} LLM calls")
        result.execution_time_ms = (time.time() - start_time) * 1000

        self._log(f"Completed in {result.execution_time_ms:.0f}ms")

        return result

    async def _evaluate_documents(
        self,
        question: str,
        documents: List[Document],
        result: RAGResult
    ) -> List[Document]:
        """LLM으로 각 문서의 관련성 평가"""
        evaluated = []

        for doc in documents:
            prompt = self.RELEVANCE_EVAL_PROMPT.format(
                question=question,
                document=doc.content[:1000]  # 토큰 제한
            )

            try:
                response = await self._call_llm(prompt)
                result.total_llm_calls += 1

                # JSON 파싱
                parsed = self._parse_json_response(response)
                relevance = parsed.get("relevance", "ambiguous")

                doc.relevance_label = relevance
                doc.metadata["eval_reason"] = parsed.get("reason", "")
                doc.metadata["key_info"] = parsed.get("key_info")

            except Exception as e:
                self._log(f"Evaluation failed for doc: {e}")
                doc.relevance_label = "ambiguous"  # 실패시 애매함으로 처리

            evaluated.append(doc)

        return evaluated

    async def _rewrite_query(self, original_query: str, result: RAGResult) -> str:
        """쿼리 재작성"""
        result.add_step("query_rewrite", "Rewriting query for better results...")
        self._log("Rewriting query...")

        prompt = self.QUERY_REWRITE_PROMPT.format(question=original_query)

        try:
            response = await self._call_llm(prompt)
            result.total_llm_calls += 1

            parsed = self._parse_json_response(response)
            queries = parsed.get("queries", [])

            if queries:
                # 첫 번째 대안 쿼리 사용
                new_query = queries[0]
                result.add_step("query_rewritten", f"New query: {new_query}")
                return new_query

        except Exception as e:
            self._log(f"Query rewrite failed: {e}")
            result.add_step("rewrite_failed", f"Rewrite failed: {e}")

        # 실패시 원본 쿼리에 약간 변형
        return f"{original_query} (details)"

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
        """중복 문서 제거 (ID 기준)"""
        seen = set()
        unique = []

        for doc in docs:
            if doc.id not in seen:
                seen.add(doc.id)
                unique.append(doc)

        # 점수 기준 정렬
        unique.sort(key=lambda d: d.rerank_score or d.score, reverse=True)
        return unique

    def _parse_json_response(self, text: str) -> dict:
        """LLM 응답에서 JSON 추출"""
        # JSON 블록 찾기
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 전체 텍스트 시도
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _no_results_response(self, query: str) -> str:
        """검색 결과 없을 때 응답"""
        return f"""I couldn't find relevant information for: "{query}"

Despite multiple search attempts with different queries, no sufficiently relevant documents were found.

Suggestions:
- The information might not be documented yet
- Try asking about a more specific aspect
- Check if the topic uses different terminology
- Consider asking the relevant team directly"""
