"""
Langfuse Client

LLM 관측성(Observability)을 위한 Langfuse 클라이언트
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager
from functools import wraps
import os
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class LangfuseConfig:
    """Langfuse 설정"""
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: str = "https://cloud.langfuse.com"
    enabled: bool = True
    debug: bool = False
    flush_interval: float = 1.0  # seconds


class LangfuseClient:
    """
    Langfuse 클라이언트

    LLM 호출, 에이전트 실행, RAG 파이프라인 추적을 위한 관측성 클라이언트
    """

    def __init__(self, config: Optional[LangfuseConfig] = None):
        self.config = config or LangfuseConfig()
        self._client = None
        self._initialized = False

        # 환경 변수에서 설정 로드
        if not self.config.public_key:
            self.config.public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        if not self.config.secret_key:
            self.config.secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if os.environ.get("LANGFUSE_HOST"):
            self.config.host = os.environ.get("LANGFUSE_HOST")

    def _ensure_initialized(self) -> bool:
        """Langfuse 클라이언트 초기화 (lazy loading)"""
        if self._initialized:
            return self._client is not None

        self._initialized = True

        if not self.config.enabled:
            logger.info("Langfuse is disabled")
            return False

        if not self.config.public_key or not self.config.secret_key:
            logger.warning(
                "Langfuse credentials not found. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
            )
            return False

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
                debug=self.config.debug
            )

            logger.info(f"Langfuse initialized: host={self.config.host}")
            return True

        except ImportError:
            logger.warning(
                "langfuse package not found. "
                "Install with: pip install langfuse"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Langfuse 사용 가능 여부"""
        return self._ensure_initialized()

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional["Trace"]:
        """
        새 트레이스 생성

        Args:
            name: 트레이스 이름
            user_id: 사용자 ID
            session_id: 세션 ID
            metadata: 추가 메타데이터
            tags: 태그 목록

        Returns:
            Trace 객체 (Langfuse 미사용시 None)
        """
        if not self._ensure_initialized():
            return None

        try:
            trace = self._client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or []
            )
            return Trace(trace, self)
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None

    @asynccontextmanager
    async def trace_context(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        트레이스 컨텍스트 매니저

        Usage:
            async with langfuse.trace_context("my_operation") as trace:
                # do something
                trace.log_generation(...)
        """
        trace = self.create_trace(name, user_id, session_id, metadata, tags)
        try:
            yield trace
        except Exception as e:
            if trace:
                trace.update(
                    metadata={"error": str(e)},
                    level="ERROR"
                )
            raise
        finally:
            if trace:
                trace.end()
            self.flush()

    def flush(self):
        """버퍼된 이벤트 전송"""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")

    def shutdown(self):
        """클라이언트 종료"""
        if self._client:
            try:
                self._client.shutdown()
                logger.info("Langfuse shutdown complete")
            except Exception as e:
                logger.error(f"Error during Langfuse shutdown: {e}")


class Trace:
    """Langfuse Trace 래퍼"""

    def __init__(self, trace, client: LangfuseClient):
        self._trace = trace
        self._client = client
        self._start_time = datetime.now()

    @property
    def trace_id(self) -> str:
        """트레이스 ID"""
        return self._trace.id if self._trace else ""

    def update(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        output: Optional[Any] = None
    ):
        """트레이스 업데이트"""
        if self._trace:
            try:
                kwargs = {}
                if metadata:
                    kwargs["metadata"] = metadata
                if level:
                    kwargs["level"] = level
                if output is not None:
                    kwargs["output"] = output
                self._trace.update(**kwargs)
            except Exception as e:
                logger.error(f"Failed to update trace: {e}")

    def end(self):
        """트레이스 종료"""
        pass  # Langfuse handles this automatically

    def span(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        input: Optional[Any] = None
    ) -> "Span":
        """
        스팬 생성 (일반 작업 추적)

        Args:
            name: 스팬 이름
            metadata: 메타데이터
            input: 입력 데이터
        """
        if not self._trace:
            return Span(None)

        try:
            span = self._trace.span(
                name=name,
                metadata=metadata or {},
                input=input
            )
            return Span(span)
        except Exception as e:
            logger.error(f"Failed to create span: {e}")
            return Span(None)

    def generation(
        self,
        name: str,
        model: str,
        input: Any,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        level: str = "DEFAULT"
    ) -> "Generation":
        """
        LLM Generation 로깅

        Args:
            name: Generation 이름
            model: 모델 이름 (e.g., "gpt-4o-mini")
            input: 입력 (프롬프트/메시지)
            output: 출력 (응답)
            metadata: 추가 메타데이터
            usage: 토큰 사용량 {"input": int, "output": int, "total": int}
            model_parameters: 모델 파라미터 (temperature 등)
            level: 로그 레벨
        """
        if not self._trace:
            return Generation(None)

        try:
            gen = self._trace.generation(
                name=name,
                model=model,
                input=input,
                output=output,
                metadata=metadata or {},
                usage=usage,
                model_parameters=model_parameters,
                level=level
            )
            return Generation(gen)
        except Exception as e:
            logger.error(f"Failed to create generation: {e}")
            return Generation(None)

    def event(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        level: str = "DEFAULT"
    ):
        """
        이벤트 로깅

        Args:
            name: 이벤트 이름
            metadata: 메타데이터
            input: 입력
            output: 출력
            level: 로그 레벨
        """
        if not self._trace:
            return

        try:
            self._trace.event(
                name=name,
                metadata=metadata or {},
                input=input,
                output=output,
                level=level
            )
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def score(
        self,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """
        스코어 기록

        Args:
            name: 스코어 이름 (e.g., "relevance", "quality")
            value: 스코어 값 (0.0 ~ 1.0)
            comment: 코멘트
        """
        if not self._trace:
            return

        try:
            self._trace.score(
                name=name,
                value=value,
                comment=comment
            )
        except Exception as e:
            logger.error(f"Failed to log score: {e}")


class Span:
    """Langfuse Span 래퍼"""

    def __init__(self, span):
        self._span = span
        self._start_time = datetime.now()

    def update(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[Any] = None,
        level: Optional[str] = None
    ):
        """스팬 업데이트"""
        if self._span:
            try:
                kwargs = {}
                if metadata:
                    kwargs["metadata"] = metadata
                if output is not None:
                    kwargs["output"] = output
                if level:
                    kwargs["level"] = level
                self._span.update(**kwargs)
            except Exception as e:
                logger.error(f"Failed to update span: {e}")

    def end(self, output: Optional[Any] = None):
        """스팬 종료"""
        if self._span:
            try:
                if output is not None:
                    self._span.end(output=output)
                else:
                    self._span.end()
            except Exception as e:
                logger.error(f"Failed to end span: {e}")

    def generation(
        self,
        name: str,
        model: str,
        input: Any,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None
    ) -> "Generation":
        """스팬 내에서 Generation 로깅"""
        if not self._span:
            return Generation(None)

        try:
            gen = self._span.generation(
                name=name,
                model=model,
                input=input,
                output=output,
                metadata=metadata or {},
                usage=usage
            )
            return Generation(gen)
        except Exception as e:
            logger.error(f"Failed to create generation in span: {e}")
            return Generation(None)


class Generation:
    """Langfuse Generation 래퍼"""

    def __init__(self, generation):
        self._generation = generation

    def update(
        self,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
        level: Optional[str] = None
    ):
        """Generation 업데이트"""
        if self._generation:
            try:
                kwargs = {}
                if output is not None:
                    kwargs["output"] = output
                if metadata:
                    kwargs["metadata"] = metadata
                if usage:
                    kwargs["usage"] = usage
                if level:
                    kwargs["level"] = level
                self._generation.update(**kwargs)
            except Exception as e:
                logger.error(f"Failed to update generation: {e}")

    def end(self, output: Optional[Any] = None, usage: Optional[Dict[str, int]] = None):
        """Generation 종료"""
        if self._generation:
            try:
                kwargs = {}
                if output is not None:
                    kwargs["output"] = output
                if usage:
                    kwargs["usage"] = usage
                self._generation.end(**kwargs)
            except Exception as e:
                logger.error(f"Failed to end generation: {e}")


# 전역 클라이언트 인스턴스
_default_client: Optional[LangfuseClient] = None


def get_langfuse_client() -> LangfuseClient:
    """전역 Langfuse 클라이언트 반환"""
    global _default_client
    if _default_client is None:
        _default_client = LangfuseClient()
    return _default_client


def trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """
    함수 데코레이터: 함수 실행을 트레이스로 래핑

    Usage:
        @trace("my_function")
        async def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_langfuse_client()
            async with client.trace_context(name, user_id, session_id, metadata, tags) as t:
                result = await func(*args, **kwargs)
                if t:
                    t.update(output=str(result)[:1000] if result else None)
                return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = get_langfuse_client()
            trace_obj = client.create_trace(name, user_id, session_id, metadata, tags)
            try:
                result = func(*args, **kwargs)
                if trace_obj:
                    trace_obj.update(output=str(result)[:1000] if result else None)
                return result
            except Exception as e:
                if trace_obj:
                    trace_obj.update(metadata={"error": str(e)}, level="ERROR")
                raise
            finally:
                if trace_obj:
                    trace_obj.end()
                client.flush()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
