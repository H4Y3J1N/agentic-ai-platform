#!/usr/bin/env python
"""
RAG Evaluation Script

Internal Ops RAG 시스템의 품질을 평가합니다.
Reference-Free 방식으로 Faithfulness, Answer Relevance, Context Relevance를 측정.

Usage:
    # 기본 평가 (샘플 쿼리)
    python scripts/evaluate_rag.py

    # 커스텀 쿼리 파일로 평가
    python scripts/evaluate_rag.py --queries queries.json

    # 결과를 JSON으로 저장
    python scripts/evaluate_rag.py --output results.json

    # 상세 출력
    python scripts/evaluate_rag.py --verbose
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add paths
SERVICE_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = SERVICE_ROOT.parent.parent
sys.path.insert(0, str(SERVICE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pipeline"))

from agentic_pipeline.evaluation import (
    create_reference_free_evaluator,
    ReferenceFreeEvaluator,
    EvaluationSample,
    BatchEvaluationResult,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 기본 샘플 쿼리 (테스트용)
DEFAULT_SAMPLE_QUERIES = [
    "프로젝트 진행상황 알려줘",
    "최근 미팅 내용 요약해줘",
    "담당자가 누구야?",
    "일정이 어떻게 돼?",
    "현재 진행 중인 태스크는?",
]


class RAGEvaluationRunner:
    """RAG 평가 실행기"""

    def __init__(
        self,
        collection_name: str = "internal_ops_notion",
        model: str = "gemini/gemini-1.5-flash",
        verbose: bool = False
    ):
        self.collection_name = collection_name
        self.model = model
        self.verbose = verbose

        # Evaluator
        self._evaluator: Optional[ReferenceFreeEvaluator] = None

        # Search tool
        self._search_tool = None

        # LLM for response generation
        self._acompletion = None

    async def _ensure_initialized(self):
        """초기화"""
        if self._evaluator is not None:
            return

        # Evaluator 초기화
        self._evaluator = create_reference_free_evaluator(
            model=self.model,
            enable_faithfulness=True,
            enable_answer_relevance=True,
            enable_context_relevance=True,
        )

        # Search tool 초기화
        from internal_ops_service.tools.enhanced_search import create_enhanced_search_tool

        self._search_tool = create_enhanced_search_tool(
            collection_name=self.collection_name,
            config_dict={
                "query_rewriting_enabled": True,
                "hybrid_search_enabled": True,
                "reranking_enabled": True,
            }
        )

        # LLM 초기화
        from litellm import acompletion
        self._acompletion = acompletion

        logger.info("RAG Evaluation Runner initialized")

    async def _generate_response(
        self,
        query: str,
        contexts: List[str]
    ) -> str:
        """컨텍스트 기반 응답 생성"""
        context_text = "\n\n".join(contexts)

        prompt = f"""Based on the following context, answer the user's question.
If the context doesn't contain enough information, say so.
Be concise and factual.

Context:
{context_text}

Question: {query}

Answer:"""

        response = await self._acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    async def evaluate_query(
        self,
        query: str
    ) -> Dict[str, Any]:
        """단일 쿼리 평가"""
        await self._ensure_initialized()

        # 1. 검색
        search_results = await self._search_tool.search(query, top_k=5)
        contexts = [r.content for r in search_results]

        if not contexts:
            return {
                "query": query,
                "error": "No search results",
                "contexts": [],
                "response": "",
                "evaluation": None
            }

        # 2. 응답 생성
        response = await self._generate_response(query, contexts)

        # 3. 평가
        eval_result = await self._evaluator.evaluate(
            query=query,
            response=response,
            contexts=contexts
        )

        result = {
            "query": query,
            "contexts": contexts,
            "response": response,
            "evaluation": {
                "faithfulness": eval_result.faithfulness,
                "answer_relevance": eval_result.answer_relevance,
                "context_relevance": eval_result.context_relevance,
                "overall_score": eval_result.overall_score,
                "all_passed": eval_result.all_passed,
            }
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"-"*60)
            print(f"Response: {response[:200]}...")
            print(f"-"*60)
            print(f"Faithfulness: {eval_result.faithfulness:.3f}")
            print(f"Answer Relevance: {eval_result.answer_relevance:.3f}")
            print(f"Context Relevance: {eval_result.context_relevance:.3f}")
            print(f"Overall: {eval_result.overall_score:.3f} ({'PASS' if eval_result.all_passed else 'FAIL'})")

        return result

    async def evaluate_batch(
        self,
        queries: List[str]
    ) -> Dict[str, Any]:
        """배치 평가"""
        await self._ensure_initialized()

        print(f"\nEvaluating {len(queries)} queries...")

        results = []
        samples = []

        for i, query in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {query[:50]}...")

            try:
                result = await self.evaluate_query(query)
                results.append(result)

                if result.get("evaluation"):
                    samples.append(EvaluationSample(
                        query=query,
                        response=result["response"],
                        contexts=result["contexts"]
                    ))
            except Exception as e:
                logger.error(f"Failed to evaluate query '{query}': {e}")
                results.append({
                    "query": query,
                    "error": str(e)
                })

        # 집계
        valid_evals = [r["evaluation"] for r in results if r.get("evaluation")]

        if valid_evals:
            summary = {
                "total_queries": len(queries),
                "successful_evals": len(valid_evals),
                "mean_faithfulness": sum(e["faithfulness"] or 0 for e in valid_evals) / len(valid_evals),
                "mean_answer_relevance": sum(e["answer_relevance"] or 0 for e in valid_evals) / len(valid_evals),
                "mean_context_relevance": sum(e["context_relevance"] or 0 for e in valid_evals) / len(valid_evals),
                "mean_overall": sum(e["overall_score"] for e in valid_evals) / len(valid_evals),
                "pass_rate": sum(1 for e in valid_evals if e["all_passed"]) / len(valid_evals),
            }
        else:
            summary = {
                "total_queries": len(queries),
                "successful_evals": 0,
                "error": "No successful evaluations"
            }

        return {
            "summary": summary,
            "results": results,
            "evaluated_at": datetime.now().isoformat()
        }

    def print_summary(self, batch_result: Dict[str, Any]):
        """결과 요약 출력"""
        summary = batch_result.get("summary", {})

        print("\n" + "=" * 60)
        print("RAG Evaluation Summary")
        print("=" * 60)
        print(f"Total Queries: {summary.get('total_queries', 0)}")
        print(f"Successful Evaluations: {summary.get('successful_evals', 0)}")
        print("-" * 60)

        if summary.get('successful_evals', 0) > 0:
            print(f"{'Metric':<25} {'Mean Score':<15}")
            print("-" * 60)
            print(f"{'Faithfulness':<25} {summary.get('mean_faithfulness', 0):.3f}")
            print(f"{'Answer Relevance':<25} {summary.get('mean_answer_relevance', 0):.3f}")
            print(f"{'Context Relevance':<25} {summary.get('mean_context_relevance', 0):.3f}")
            print("-" * 60)
            print(f"{'Overall':<25} {summary.get('mean_overall', 0):.3f}")
            print(f"{'Pass Rate':<25} {summary.get('pass_rate', 0):.1%}")
        else:
            print("No successful evaluations to summarize")

        print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system quality"
    )
    parser.add_argument(
        "--queries",
        type=str,
        help="JSON file with queries to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="internal_ops_notion",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-1.5-flash",
        help="LLM model for evaluation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # 쿼리 로드
    if args.queries:
        with open(args.queries, "r", encoding="utf-8") as f:
            data = json.load(f)
            queries = data if isinstance(data, list) else data.get("queries", [])
    else:
        queries = DEFAULT_SAMPLE_QUERIES
        print(f"Using {len(queries)} default sample queries")

    # 평가 실행
    runner = RAGEvaluationRunner(
        collection_name=args.collection,
        model=args.model,
        verbose=args.verbose
    )

    batch_result = await runner.evaluate_batch(queries)

    # 결과 출력
    runner.print_summary(batch_result)

    # 결과 저장
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(batch_result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
