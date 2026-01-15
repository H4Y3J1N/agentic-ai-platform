"""
Pipeline Stage Base

파이프라인 스테이지 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from ..context import PipelineContext, StageError

logger = logging.getLogger(__name__)


class Stage(ABC):
    """파이프라인 스테이지 베이스 클래스"""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        스테이지 처리

        Args:
            context: 파이프라인 컨텍스트

        Returns:
            업데이트된 컨텍스트
        """
        pass

    def should_skip(self, context: PipelineContext) -> bool:
        """
        스킵 조건 확인

        Args:
            context: 파이프라인 컨텍스트

        Returns:
            스킵 여부
        """
        return context.should_skip or context.has_fatal_errors()

    async def __call__(self, context: PipelineContext) -> PipelineContext:
        """스테이지 실행"""
        # 스킵 조건 확인
        if self.should_skip(context):
            logger.debug(f"Skipping stage {self.name}: {context.skip_reason}")
            return context

        context.mark_stage_start(self.name)
        logger.debug(f"Starting stage: {self.name}")

        try:
            result = await self.process(context)
            context.mark_stage_complete(self.name)
            logger.debug(f"Completed stage: {self.name}")
            return result

        except Exception as e:
            error = StageError(
                stage_name=self.name,
                error_type=type(e).__name__,
                message=str(e),
                is_fatal=self._is_fatal_error(e)
            )
            context.add_error(error)
            logger.error(f"Stage {self.name} failed: {e}")

            if error.is_fatal:
                raise

            return context

    def _is_fatal_error(self, error: Exception) -> bool:
        """
        치명적 오류 판단

        서브클래스에서 오버라이드 가능
        """
        # 기본: 모든 오류는 비치명적
        return False


class ConditionalStage(Stage):
    """조건부 실행 스테이지"""

    @abstractmethod
    def condition(self, context: PipelineContext) -> bool:
        """
        실행 조건

        Args:
            context: 파이프라인 컨텍스트

        Returns:
            실행 여부
        """
        pass

    def should_skip(self, context: PipelineContext) -> bool:
        """조건 확인 후 스킵 결정"""
        if super().should_skip(context):
            return True
        return not self.condition(context)


class CompositeStage(Stage):
    """복합 스테이지 (여러 스테이지 순차 실행)"""

    def __init__(self, stages: list[Stage], name: Optional[str] = None):
        super().__init__(name)
        self.stages = stages

    async def process(self, context: PipelineContext) -> PipelineContext:
        """순차적으로 모든 스테이지 실행"""
        for stage in self.stages:
            context = await stage(context)
            if context.should_skip or context.has_fatal_errors():
                break
        return context


class ParallelStage(Stage):
    """병렬 스테이지 (독립적 스테이지 동시 실행)"""

    def __init__(self, stages: list[Stage], name: Optional[str] = None):
        super().__init__(name)
        self.stages = stages

    async def process(self, context: PipelineContext) -> PipelineContext:
        """병렬로 모든 스테이지 실행"""
        import asyncio

        tasks = [stage(context) for stage in self.stages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 병합 (마지막 성공 결과 사용)
        for result in results:
            if isinstance(result, PipelineContext):
                context = result
            elif isinstance(result, Exception):
                context.add_error(StageError(
                    stage_name="ParallelStage",
                    error_type=type(result).__name__,
                    message=str(result),
                    is_fatal=False
                ))

        return context
