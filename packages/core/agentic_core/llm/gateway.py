"""
LLM Gateway

LiteLLM 기반 통합 LLM 게이트웨이
- 100+ LLM 프로바이더 지원 (OpenAI, Anthropic, Ollama, Azure, Bedrock 등)
- 통일된 인터페이스
- 비용 추적, 폴백, 로드밸런싱
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging
import time

import litellm
from litellm import acompletion, completion

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """지원 프로바이더"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE = "azure"
    BEDROCK = "bedrock"
    VERTEX = "vertex_ai"
    GROQ = "groq"
    TOGETHER = "together_ai"


@dataclass
class Message:
    """채팅 메시지"""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str
    model: str
    provider: str

    # 토큰 사용량
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # 메타데이터
    latency_ms: float = 0.0
    cost: Optional[float] = None
    finish_reason: Optional[str] = None

    # 원본 응답 (디버깅용)
    raw_response: Optional[Any] = None


@dataclass
class GatewayConfig:
    """게이트웨이 설정"""
    # 기본 모델
    default_model: str = "gpt-4o-mini"

    # 요청 설정
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 60.0

    # 재시도 설정
    max_retries: int = 3
    retry_delay: float = 1.0

    # 폴백 모델 (기본 모델 실패 시)
    fallback_models: List[str] = field(default_factory=list)

    # 비용 추적
    track_cost: bool = True

    # 캐싱
    enable_cache: bool = False
    cache_ttl: int = 3600


class LLMGateway:
    """
    통합 LLM 게이트웨이

    사용 예시:
        gateway = LLMGateway()

        # 단순 호출
        response = await gateway.chat("안녕하세요")

        # 메시지 리스트
        response = await gateway.chat([
            Message(role="system", content="당신은 도움이 되는 AI입니다."),
            Message(role="user", content="파이썬이란?")
        ])

        # 모델 지정
        response = await gateway.chat("설명해줘", model="claude-3-5-sonnet-20241022")
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self._configure_litellm()

        # 메트릭
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._latencies: List[float] = []

    def _configure_litellm(self) -> None:
        """LiteLLM 설정"""
        litellm.set_verbose = False
        litellm.drop_params = True  # 지원하지 않는 파라미터 자동 제거

        if self.config.max_retries:
            litellm.num_retries = self.config.max_retries

    async def chat(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        LLM 채팅 호출

        Args:
            messages: 메시지 (문자열, Message 리스트, 또는 dict 리스트)
            model: 모델 이름 (None이면 기본값)
            temperature: 온도 (None이면 기본값)
            max_tokens: 최대 토큰 (None이면 기본값)
            **kwargs: 추가 LiteLLM 파라미터

        Returns:
            LLMResponse
        """
        # 메시지 정규화
        normalized_messages = self._normalize_messages(messages)

        # 파라미터 설정
        effective_model = model or self.config.default_model
        effective_temp = temperature if temperature is not None else self.config.temperature
        effective_max_tokens = max_tokens or self.config.max_tokens

        start_time = time.time()

        try:
            response = await self._call_with_fallback(
                messages=normalized_messages,
                model=effective_model,
                temperature=effective_temp,
                max_tokens=effective_max_tokens,
                **kwargs
            )

            latency = (time.time() - start_time) * 1000

            # 응답 파싱
            llm_response = self._parse_response(response, effective_model, latency)

            # 메트릭 업데이트
            self._update_metrics(llm_response)

            return llm_response

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def _call_with_fallback(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> Any:
        """폴백 로직 포함 호출"""
        models_to_try = [model] + self.config.fallback_models
        last_error = None

        for m in models_to_try:
            try:
                logger.debug(f"Trying model: {m}")
                response = await acompletion(
                    model=m,
                    messages=messages,
                    timeout=self.config.timeout,
                    **kwargs
                )
                return response
            except Exception as e:
                logger.warning(f"Model {m} failed: {e}")
                last_error = e
                continue

        raise last_error or Exception("All models failed")

    def _normalize_messages(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """메시지 정규화"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        result = []
        for msg in messages:
            if isinstance(msg, Message):
                result.append(msg.to_dict())
            elif isinstance(msg, dict):
                result.append(msg)
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

        return result

    def _parse_response(
        self,
        response: Any,
        model: str,
        latency_ms: float
    ) -> LLMResponse:
        """LiteLLM 응답 파싱"""
        choice = response.choices[0]
        usage = response.usage or {}

        # 비용 계산
        cost = None
        if self.config.track_cost:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                pass

        # 프로바이더 추출
        provider = self._extract_provider(model)

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or model,
            provider=provider,
            prompt_tokens=getattr(usage, 'prompt_tokens', 0),
            completion_tokens=getattr(usage, 'completion_tokens', 0),
            total_tokens=getattr(usage, 'total_tokens', 0),
            latency_ms=latency_ms,
            cost=cost,
            finish_reason=choice.finish_reason,
            raw_response=response
        )

    def _extract_provider(self, model: str) -> str:
        """모델명에서 프로바이더 추출"""
        if model.startswith("gpt-") or model.startswith("o1"):
            return ModelProvider.OPENAI.value
        elif model.startswith("claude-"):
            return ModelProvider.ANTHROPIC.value
        elif model.startswith("ollama/"):
            return ModelProvider.OLLAMA.value
        elif model.startswith("azure/"):
            return ModelProvider.AZURE.value
        elif model.startswith("bedrock/"):
            return ModelProvider.BEDROCK.value
        elif model.startswith("groq/"):
            return ModelProvider.GROQ.value
        elif model.startswith("together_ai/"):
            return ModelProvider.TOGETHER.value
        else:
            return "unknown"

    def _update_metrics(self, response: LLMResponse) -> None:
        """메트릭 업데이트"""
        self._total_requests += 1
        self._total_tokens += response.total_tokens
        if response.cost:
            self._total_cost += response.cost
        self._latencies.append(response.latency_ms)

        # 최근 100개만 유지
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]

    def chat_sync(
        self,
        messages: Union[str, List[Message], List[Dict[str, str]]],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """동기 호출 (테스트/스크립트용)"""
        normalized_messages = self._normalize_messages(messages)
        effective_model = model or self.config.default_model

        start_time = time.time()

        response = completion(
            model=effective_model,
            messages=normalized_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            timeout=self.config.timeout
        )

        latency = (time.time() - start_time) * 1000
        llm_response = self._parse_response(response, effective_model, latency)
        self._update_metrics(llm_response)

        return llm_response

    def get_stats(self) -> Dict[str, Any]:
        """게이트웨이 통계"""
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0

        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "default_model": self.config.default_model,
        }

    def reset_stats(self) -> None:
        """통계 초기화"""
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._latencies.clear()


# 편의 함수
_default_gateway: Optional[LLMGateway] = None


def get_gateway(config: Optional[GatewayConfig] = None) -> LLMGateway:
    """기본 게이트웨이 인스턴스 반환 (싱글톤)"""
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = LLMGateway(config)
    return _default_gateway


async def chat(
    messages: Union[str, List[Message], List[Dict[str, str]]],
    model: Optional[str] = None,
    **kwargs
) -> LLMResponse:
    """빠른 채팅 호출"""
    return await get_gateway().chat(messages, model=model, **kwargs)
