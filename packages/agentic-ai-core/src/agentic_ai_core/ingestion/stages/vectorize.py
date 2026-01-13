"""
Vectorize Stage

벡터화 결정 및 실행 스테이지
"""

from typing import Optional, List
import logging

from .base import Stage
from ..context import PipelineContext, VectorizationDecision
from ...schema import Chunk
from ...rag import Chunker, Embedder, create_chunker, ChunkingStrategy, ChunkerConfig

logger = logging.getLogger(__name__)


class VectorizeStage(Stage):
    """벡터화 스테이지"""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
        min_relevance: float = 0.3,
        always_vectorize_types: Optional[List[str]] = None
    ):
        super().__init__("VectorizeStage")
        self.embedder = embedder
        self.chunker = chunker or create_chunker(
            ChunkingStrategy.SEMANTIC,
            ChunkerConfig(chunk_size=500, chunk_overlap=50)
        )
        self.min_relevance = min_relevance
        self.always_vectorize_types = always_vectorize_types or ["policy", "faq"]

    async def process(self, context: PipelineContext) -> PipelineContext:
        """벡터화 실행"""
        content = context.raw_content or ""
        if not content:
            return context

        # 벡터화 결정
        decision = self._decide_vectorization(context)
        context.vectorization_decision = decision

        if not decision.should_vectorize:
            logger.debug(f"Skipping vectorization: {decision.reason}")
            return context

        # 청킹
        chunks = await self._create_chunks(context, decision)
        context.chunks = chunks

        # 임베딩 생성 (embedder가 있을 때만)
        if self.embedder and chunks:
            await self._generate_embeddings(context, chunks)

        logger.debug(
            f"Vectorized document: {len(chunks)} chunks, "
            f"granularity={decision.granularity}"
        )

        return context

    def _decide_vectorization(self, context: PipelineContext) -> VectorizationDecision:
        """벡터화 여부 결정"""
        doc_type = context.document_type

        # 강제 벡터화 타입
        if doc_type and doc_type.value in self.always_vectorize_types:
            return VectorizationDecision(
                should_vectorize=True,
                reason=f"Document type {doc_type.value} always vectorized",
                granularity="chunks",
                priority=8
            )

        # 관련성 기반 결정
        relevance = context.overall_relevance

        if relevance < self.min_relevance:
            return VectorizationDecision(
                should_vectorize=False,
                reason=f"Low relevance score: {relevance:.2f}",
                granularity="none",
                priority=0
            )

        # 관련성에 따른 세분화 결정
        if relevance >= 0.7:
            granularity = "chunks"
            priority = 10
        elif relevance >= 0.5:
            granularity = "chunks"
            priority = 7
        elif relevance >= 0.3:
            granularity = "summary"
            priority = 4
        else:
            granularity = "metadata"
            priority = 2

        # ROI 계산
        roi = self._calculate_roi(context, relevance)

        return VectorizationDecision(
            should_vectorize=True,
            reason=f"Relevance: {relevance:.2f}, ROI: {roi:.2f}",
            granularity=granularity,
            priority=priority
        )

    def _calculate_roi(self, context: PipelineContext, relevance: float) -> float:
        """
        벡터화 ROI 계산

        ROI = (Expected_Query_Value × Query_Probability) / Storage_Cost
        """
        # 예상 쿼리 가치 (관련성 기반)
        query_value = relevance * 10  # 기본 가치

        # 쿼리 확률 (문서 타입 기반)
        query_prob = 0.5  # 기본값
        if context.document_type:
            type_probs = {
                "policy": 0.8,
                "faq": 0.9,
                "meeting_note": 0.4,
                "technical_doc": 0.6,
                "announcement": 0.3,
            }
            query_prob = type_probs.get(context.document_type.value, 0.5)

        # 저장 비용 (콘텐츠 길이 기반)
        content_length = len(context.raw_content or "")
        storage_cost = max(content_length / 10000, 0.1)

        return (query_value * query_prob) / storage_cost

    async def _create_chunks(
        self,
        context: PipelineContext,
        decision: VectorizationDecision
    ) -> List[Chunk]:
        """청크 생성"""
        content = context.raw_content or ""

        if decision.granularity == "full":
            # 전체 문서를 하나의 청크로
            return [Chunk(
                document_id=context.source_item.id,
                content=content,
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                token_count=len(content.split()),
                metadata={
                    "source_type": context.source_item.source_type.value,
                    "document_type": context.document_type.value if context.document_type else "unknown",
                }
            )]

        if decision.granularity == "summary":
            # 요약 생성 (간단한 버전: 첫 부분만)
            summary = self._create_summary(content)
            return [Chunk(
                document_id=context.source_item.id,
                content=summary,
                chunk_index=0,
                start_char=0,
                end_char=len(summary),
                token_count=len(summary.split()),
                metadata={
                    "source_type": context.source_item.source_type.value,
                    "is_summary": True,
                }
            )]

        if decision.granularity == "metadata":
            # 메타데이터만
            meta_text = self._metadata_to_text(context)
            return [Chunk(
                document_id=context.source_item.id,
                content=meta_text,
                chunk_index=0,
                start_char=0,
                end_char=len(meta_text),
                metadata={
                    "source_type": context.source_item.source_type.value,
                    "is_metadata_only": True,
                }
            )]

        # chunks: 일반 청킹
        text_chunks = self.chunker.chunk(content)

        chunks = []
        for i, tc in enumerate(text_chunks):
            chunk = Chunk(
                document_id=context.source_item.id,
                content=tc.content,
                chunk_index=i,
                start_char=tc.metadata.start_char,
                end_char=tc.metadata.end_char,
                header_context=tc.metadata.header_context,
                section_name=tc.metadata.section_name,
                token_count=tc.metadata.token_count,
                metadata={
                    "source_type": context.source_item.source_type.value,
                    "document_type": context.document_type.value if context.document_type else "unknown",
                    **(context.inferred_metadata or {})
                }
            )
            chunks.append(chunk)

        return chunks

    def _create_summary(self, content: str, max_length: int = 500) -> str:
        """간단한 요약 생성"""
        # 첫 번째 섹션 또는 첫 부분
        lines = content.strip().split('\n')
        summary_lines = []
        total_length = 0

        for line in lines:
            if total_length + len(line) > max_length:
                break
            summary_lines.append(line)
            total_length += len(line)

        return '\n'.join(summary_lines)

    def _metadata_to_text(self, context: PipelineContext) -> str:
        """메타데이터를 텍스트로 변환"""
        parts = []

        if context.inferred_metadata.get("title"):
            parts.append(f"Title: {context.inferred_metadata['title']}")

        if context.document_type:
            parts.append(f"Type: {context.document_type.value}")

        if context.inferred_metadata.get("topics"):
            parts.append(f"Topics: {', '.join(context.inferred_metadata['topics'])}")

        if context.inferred_metadata.get("keywords"):
            parts.append(f"Keywords: {', '.join(context.inferred_metadata['keywords'][:10])}")

        return '\n'.join(parts)

    async def _generate_embeddings(
        self,
        context: PipelineContext,
        chunks: List[Chunk]
    ) -> None:
        """임베딩 생성"""
        if not self.embedder:
            return

        texts = [chunk.content for chunk in chunks]

        try:
            results = await self.embedder.embed_batch(texts)

            for chunk, result in zip(chunks, results):
                chunk.embedding = result.embedding
                chunk.embedding_model = result.model

            context.embeddings = {
                chunk.id: chunk.embedding
                for chunk in chunks
                if chunk.embedding
            }

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # 임베딩 실패해도 청크는 유지
