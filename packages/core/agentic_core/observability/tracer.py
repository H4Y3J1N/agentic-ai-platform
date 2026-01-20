"""
Tracer

에이전트 및 RAG 파이프라인 추적을 위한 통합 Tracer
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging

from .langfuse_client import LangfuseClient, Trace, get_langfuse_client

logger = logging.getLogger(__name__)


@dataclass
class TracerConfig:
    """Tracer 설정"""
    enabled: bool = True
    trace_llm_calls: bool = True
    trace_retrievals: bool = True
    trace_tool_calls: bool = True
    include_inputs: bool = True
    include_outputs: bool = True
    max_output_length: int = 2000


class Tracer:
    """
    통합 Tracer

    에이전트 실행, LLM 호출, RAG 검색 등을 추적
    """

    def __init__(
        self,
        config: Optional[TracerConfig] = None,
        langfuse_client: Optional[LangfuseClient] = None
    ):
        self.config = config or TracerConfig()
        self._langfuse = langfuse_client or get_langfuse_client()
        self._current_trace: Optional[Trace] = None

    @property
    def is_enabled(self) -> bool:
        """추적 활성화 여부"""
        return self.config.enabled and self._langfuse.is_available

    def start_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Trace]:
        """
        새 트레이스 시작

        Args:
            name: 트레이스 이름 (e.g., "agent_execution", "rag_query")
            user_id: 사용자 ID
            session_id: 세션 ID
            metadata: 추가 메타데이터
            tags: 태그
        """
        if not self.is_enabled:
            return None

        self._current_trace = self._langfuse.create_trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags
        )
        return self._current_trace

    def end_trace(self, output: Optional[Any] = None):
        """현재 트레이스 종료"""
        if self._current_trace:
            if output is not None:
                self._current_trace.update(output=self._truncate_output(output))
            self._current_trace.end()
            self._langfuse.flush()
            self._current_trace = None

    @asynccontextmanager
    async def trace(
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
            async with tracer.trace("my_agent") as t:
                result = await agent.execute(...)
                t.log_output(result)
        """
        trace = self.start_trace(name, user_id, session_id, metadata, tags)
        context = TraceContext(trace, self)
        try:
            yield context
        except Exception as e:
            context.log_error(str(e))
            raise
        finally:
            self.end_trace(context.output)

    def log_llm_call(
        self,
        name: str,
        model: str,
        messages: List[Dict[str, str]],
        response: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        model_parameters: Optional[Dict[str, Any]] = None
    ):
        """
        LLM 호출 로깅

        Args:
            name: 호출 이름
            model: 모델명
            messages: 입력 메시지
            response: 응답
            usage: 토큰 사용량
            model_parameters: 모델 파라미터
        """
        if not self.config.trace_llm_calls or not self._current_trace:
            return

        input_data = messages if self.config.include_inputs else "[REDACTED]"
        output_data = self._truncate_output(response) if self.config.include_outputs else "[REDACTED]"

        self._current_trace.generation(
            name=name,
            model=model,
            input=input_data,
            output=output_data,
            usage=usage,
            model_parameters=model_parameters
        )

    def log_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        source: str = "vector_store",
        top_k: int = 5
    ):
        """
        RAG 검색 로깅

        Args:
            query: 검색 쿼리
            results: 검색 결과
            source: 검색 소스
            top_k: 검색 개수
        """
        if not self.config.trace_retrievals or not self._current_trace:
            return

        # 결과 요약
        result_summary = [
            {
                "id": r.get("id", ""),
                "score": r.get("score", 0),
                "snippet": r.get("content", "")[:200] + "..." if r.get("content") else ""
            }
            for r in results[:top_k]
        ]

        self._current_trace.span(
            name=f"retrieval_{source}",
            metadata={
                "source": source,
                "top_k": top_k,
                "result_count": len(results)
            },
            input=query if self.config.include_inputs else "[REDACTED]"
        ).end(output=result_summary if self.config.include_outputs else "[REDACTED]")

    def log_tool_call(
        self,
        tool_name: str,
        input_args: Dict[str, Any],
        output: Any,
        error: Optional[str] = None
    ):
        """
        Tool 호출 로깅

        Args:
            tool_name: Tool 이름
            input_args: 입력 인자
            output: 출력
            error: 에러 메시지
        """
        if not self.config.trace_tool_calls or not self._current_trace:
            return

        span = self._current_trace.span(
            name=f"tool_{tool_name}",
            metadata={"tool": tool_name, "has_error": error is not None},
            input=input_args if self.config.include_inputs else "[REDACTED]"
        )

        if error:
            span.update(level="ERROR", metadata={"error": error})
            span.end(output={"error": error})
        else:
            span.end(output=self._truncate_output(output) if self.config.include_outputs else "[REDACTED]")

    def log_event(
        self,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "DEFAULT"
    ):
        """
        커스텀 이벤트 로깅

        Args:
            name: 이벤트 이름
            data: 이벤트 데이터
            level: 로그 레벨
        """
        if not self._current_trace:
            return

        self._current_trace.event(
            name=name,
            metadata=data,
            level=level
        )

    def score(
        self,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """
        스코어 기록

        Args:
            name: 스코어 이름
            value: 값 (0.0 ~ 1.0)
            comment: 코멘트
        """
        if self._current_trace:
            self._current_trace.score(name, value, comment)

    def _truncate_output(self, output: Any) -> Any:
        """출력 길이 제한"""
        if output is None:
            return None
        str_output = str(output)
        if len(str_output) > self.config.max_output_length:
            return str_output[:self.config.max_output_length] + "..."
        return str_output

    def flush(self):
        """버퍼 플러시"""
        self._langfuse.flush()


class TraceContext:
    """트레이스 컨텍스트"""

    def __init__(self, trace: Optional[Trace], tracer: Tracer):
        self._trace = trace
        self._tracer = tracer
        self.output: Optional[Any] = None
        self._error: Optional[str] = None

    @property
    def trace_id(self) -> str:
        """트레이스 ID"""
        return self._trace.trace_id if self._trace else ""

    def log_output(self, output: Any):
        """출력 기록"""
        self.output = output

    def log_error(self, error: str):
        """에러 기록"""
        self._error = error
        if self._trace:
            self._trace.update(
                metadata={"error": error},
                level="ERROR"
            )

    def log_llm(
        self,
        name: str,
        model: str,
        messages: List[Dict[str, str]],
        response: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        model_parameters: Optional[Dict[str, Any]] = None
    ):
        """LLM 호출 로깅 (위임)"""
        self._tracer.log_llm_call(name, model, messages, response, usage, model_parameters)

    def log_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        source: str = "vector_store",
        top_k: int = 5
    ):
        """검색 로깅 (위임)"""
        self._tracer.log_retrieval(query, results, source, top_k)

    def log_tool(
        self,
        tool_name: str,
        input_args: Dict[str, Any],
        output: Any,
        error: Optional[str] = None
    ):
        """Tool 호출 로깅 (위임)"""
        self._tracer.log_tool_call(tool_name, input_args, output, error)

    def event(
        self,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "DEFAULT"
    ):
        """이벤트 로깅 (위임)"""
        self._tracer.log_event(name, data, level)

    def score(self, name: str, value: float, comment: Optional[str] = None):
        """스코어 기록 (위임)"""
        self._tracer.score(name, value, comment)


# 전역 tracer 인스턴스
_default_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """전역 Tracer 반환"""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer()
    return _default_tracer
