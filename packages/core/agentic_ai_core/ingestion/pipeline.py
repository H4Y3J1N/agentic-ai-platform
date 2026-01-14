"""
Ingestion Pipeline

데이터 수집 파이프라인 오케스트레이터
"""

from typing import List, Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio

from .context import PipelineContext, SourceItem, ProcessingStatus, StageError
from .stages.base import Stage

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    context: PipelineContext
    success: bool
    duration_ms: float
    stages_executed: List[str]
    errors: List[StageError]


@dataclass
class BatchResult:
    """배치 처리 결과"""
    total: int
    successful: int
    failed: int
    skipped: int
    results: List[PipelineResult]
    duration_ms: float


class Pipeline:
    """데이터 수집 파이프라인"""

    def __init__(
        self,
        name: str = "default",
        stages: Optional[List[Stage]] = None,
        error_handler: Optional[Callable[[StageError, PipelineContext], Awaitable[None]]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.name = name
        self.stages = stages or []
        self.error_handler = error_handler
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 메트릭
        self._total_processed = 0
        self._total_errors = 0
        self._stage_timings: Dict[str, List[float]] = {}

    def add_stage(self, stage: Stage) -> "Pipeline":
        """스테이지 추가"""
        self.stages.append(stage)
        return self

    def remove_stage(self, stage_name: str) -> "Pipeline":
        """스테이지 제거"""
        self.stages = [s for s in self.stages if s.name != stage_name]
        return self

    async def process(
        self,
        source_item: SourceItem,
        initial_content: Optional[str] = None
    ) -> PipelineResult:
        """단일 아이템 처리"""
        start_time = datetime.now()
        stages_executed = []
        errors = []

        # 컨텍스트 초기화
        context = PipelineContext(
            source_item=source_item,
            raw_content=initial_content
        )

        try:
            # 각 스테이지 실행
            for stage in self.stages:
                if context.should_skip:
                    logger.debug(f"Skipping remaining stages for {source_item.id}")
                    break

                if stage.should_skip(context):
                    logger.debug(f"Stage {stage.name} skipped")
                    continue

                stage_start = datetime.now()

                try:
                    context = await self._execute_stage_with_retry(stage, context)
                    stages_executed.append(stage.name)

                    # 스테이지 타이밍 기록
                    stage_duration = (datetime.now() - stage_start).total_seconds() * 1000
                    if stage.name not in self._stage_timings:
                        self._stage_timings[stage.name] = []
                    self._stage_timings[stage.name].append(stage_duration)

                except Exception as e:
                    error = StageError(
                        stage=stage.name,
                        error=str(e),
                        timestamp=datetime.now()
                    )
                    errors.append(error)
                    context.errors.append(error)

                    if self.error_handler:
                        await self.error_handler(error, context)

                    # 치명적 오류면 중단
                    if stage.is_critical:
                        context.status = ProcessingStatus.FAILED
                        break

            # 최종 상태 결정
            if context.status != ProcessingStatus.FAILED:
                if errors:
                    context.status = ProcessingStatus.PARTIAL
                else:
                    context.status = ProcessingStatus.COMPLETED

            self._total_processed += 1
            if errors:
                self._total_errors += len(errors)

        except Exception as e:
            logger.error(f"Pipeline failed for {source_item.id}: {e}")
            context.status = ProcessingStatus.FAILED
            errors.append(StageError(
                stage="pipeline",
                error=str(e),
                timestamp=datetime.now()
            ))

        duration = (datetime.now() - start_time).total_seconds() * 1000

        return PipelineResult(
            context=context,
            success=context.status in [ProcessingStatus.COMPLETED, ProcessingStatus.PARTIAL],
            duration_ms=duration,
            stages_executed=stages_executed,
            errors=errors
        )

    async def _execute_stage_with_retry(
        self,
        stage: Stage,
        context: PipelineContext
    ) -> PipelineContext:
        """재시도 로직과 함께 스테이지 실행"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return await stage.process(context)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Stage {stage.name} failed (attempt {attempt + 1}), retrying..."
                    )
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise last_error

    async def process_batch(
        self,
        items: List[SourceItem],
        contents: Optional[Dict[str, str]] = None,
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> BatchResult:
        """배치 처리"""
        start_time = datetime.now()
        contents = contents or {}
        results = []
        successful = 0
        failed = 0
        skipped = 0

        # 세마포어로 동시성 제어
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(item: SourceItem) -> PipelineResult:
            async with semaphore:
                content = contents.get(item.id)
                return await self.process(item, content)

        # 비동기 배치 처리
        tasks = [process_with_semaphore(item) for item in items]

        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            if result.success:
                if result.context.should_skip:
                    skipped += 1
                else:
                    successful += 1
            else:
                failed += 1

            if progress_callback:
                await progress_callback(i + 1, len(items))

        duration = (datetime.now() - start_time).total_seconds() * 1000

        return BatchResult(
            total=len(items),
            successful=successful,
            failed=failed,
            skipped=skipped,
            results=results,
            duration_ms=duration
        )

    def get_stats(self) -> Dict[str, Any]:
        """파이프라인 통계"""
        avg_timings = {}
        for stage_name, timings in self._stage_timings.items():
            if timings:
                avg_timings[stage_name] = {
                    "avg_ms": sum(timings) / len(timings),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                    "count": len(timings)
                }

        return {
            "name": self.name,
            "stages": [s.name for s in self.stages],
            "total_processed": self._total_processed,
            "total_errors": self._total_errors,
            "stage_timings": avg_timings
        }

    def reset_stats(self) -> None:
        """통계 초기화"""
        self._total_processed = 0
        self._total_errors = 0
        self._stage_timings.clear()


class PipelineBuilder:
    """파이프라인 빌더"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.stages: List[Stage] = []
        self.error_handler = None
        self.max_retries = 3
        self.retry_delay = 1.0

    def add_stage(self, stage: Stage) -> "PipelineBuilder":
        """스테이지 추가"""
        self.stages.append(stage)
        return self

    def with_stages(self, stages: List[Stage]) -> "PipelineBuilder":
        """여러 스테이지 추가"""
        self.stages.extend(stages)
        return self

    def with_error_handler(
        self,
        handler: Callable[[StageError, PipelineContext], Awaitable[None]]
    ) -> "PipelineBuilder":
        """에러 핸들러 설정"""
        self.error_handler = handler
        return self

    def with_retry(
        self,
        max_retries: int = 3,
        delay: float = 1.0
    ) -> "PipelineBuilder":
        """재시도 설정"""
        self.max_retries = max_retries
        self.retry_delay = delay
        return self

    def build(self) -> Pipeline:
        """파이프라인 생성"""
        return Pipeline(
            name=self.name,
            stages=self.stages,
            error_handler=self.error_handler,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )


def create_default_pipeline(
    vector_store=None,
    embedder=None,
    graph_store=None,
    document_store=None
) -> Pipeline:
    """기본 파이프라인 생성"""
    from .stages import (
        ParseStage,
        ExtractStage,
        InferStage,
        ScoreStage,
        VectorizeStage,
        StoreStage,
    )

    builder = PipelineBuilder("default")

    # 기본 스테이지들
    builder.add_stage(ParseStage())
    builder.add_stage(ExtractStage())
    builder.add_stage(InferStage())
    builder.add_stage(ScoreStage())
    builder.add_stage(VectorizeStage(embedder=embedder))
    builder.add_stage(StoreStage(
        vector_store=vector_store,
        graph_store=graph_store,
        document_store=document_store
    ))

    return builder.build()


def create_lightweight_pipeline() -> Pipeline:
    """경량 파이프라인 생성 (벡터화 없음)"""
    from .stages import (
        ParseStage,
        ExtractStage,
        InferStage,
    )

    builder = PipelineBuilder("lightweight")

    builder.add_stage(ParseStage())
    builder.add_stage(ExtractStage())
    builder.add_stage(InferStage())

    return builder.build()


def create_indexing_pipeline(
    vector_store,
    embedder,
    document_store=None
) -> Pipeline:
    """인덱싱 전용 파이프라인"""
    from .stages import (
        ParseStage,
        InferStage,
        VectorizeStage,
        StoreStage,
    )

    builder = PipelineBuilder("indexing")

    builder.add_stage(ParseStage())
    builder.add_stage(InferStage())
    builder.add_stage(VectorizeStage(
        embedder=embedder,
        min_relevance=0.0  # 모든 문서 벡터화
    ))
    builder.add_stage(StoreStage(
        vector_store=vector_store,
        document_store=document_store,
        store_entities=False,
        store_relationships=False
    ))

    return builder.build()
