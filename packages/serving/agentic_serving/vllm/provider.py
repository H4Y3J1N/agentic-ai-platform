"""
vLLM Provider

vLLM 서버와 통신하는 LLM Provider 구현
OpenAI-compatible API 사용
"""

from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass, field
import httpx
from openai import AsyncOpenAI

from .config import VLLMConfig


@dataclass
class VLLMResponse:
    """vLLM 응답"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    lora_adapter: Optional[str] = None  # 사용된 LoRA 어댑터


@dataclass
class VLLMStreamChunk:
    """vLLM 스트리밍 청크"""
    content: str
    finish_reason: Optional[str] = None
    is_final: bool = False


class VLLMProvider:
    """
    vLLM Provider

    vLLM 서버(OpenAI-compatible API)와 통신하여
    텍스트 생성, 스트리밍, LoRA 어댑터 스왑을 지원합니다.
    """

    def __init__(self, config: VLLMConfig):
        self.config = config
        self._client: Optional[AsyncOpenAI] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> AsyncOpenAI:
        """OpenAI 클라이언트 lazy initialization"""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    async def _get_http_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 lazy initialization (비표준 API용)"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/v1"),
                timeout=self.config.timeout,
            )
        return self._http_client

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        lora_adapter: Optional[str] = None,
        **kwargs,
    ) -> VLLMResponse:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            model: 모델 이름 (기본값: config.default_model)
            max_tokens: 최대 토큰 수
            temperature: 샘플링 temperature
            top_p: nucleus sampling p 값
            stop: 중단 토큰 리스트
            lora_adapter: 사용할 LoRA 어댑터 이름
            **kwargs: 추가 생성 파라미터

        Returns:
            VLLMResponse: 생성 결과
        """
        client = await self._get_client()

        # 파라미터 설정
        request_model = model or self.config.default_model
        if not request_model:
            raise ValueError("Model name is required")

        extra_body = kwargs.pop("extra_body", {})

        # LoRA 어댑터 설정 (vLLM 확장)
        if lora_adapter and self.config.enable_lora_routing:
            # vLLM은 model 파라미터에 adapter 이름을 포함
            # 예: "base-model:lora-adapter-name"
            request_model = f"{request_model}:{lora_adapter}"

        response = await client.completions.create(
            model=request_model,
            prompt=prompt,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature or self.config.default_temperature,
            top_p=top_p,
            stop=stop,
            extra_body=extra_body if extra_body else None,
            **kwargs,
        )

        choice = response.choices[0]
        return VLLMResponse(
            content=choice.text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason,
            lora_adapter=lora_adapter,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        lora_adapter: Optional[str] = None,
        **kwargs,
    ) -> VLLMResponse:
        """
        채팅 완성

        Args:
            messages: 메시지 리스트 [{"role": "user", "content": "..."}]
            model: 모델 이름
            max_tokens: 최대 토큰 수
            temperature: 샘플링 temperature
            top_p: nucleus sampling p 값
            stop: 중단 토큰 리스트
            lora_adapter: 사용할 LoRA 어댑터 이름
            **kwargs: 추가 생성 파라미터

        Returns:
            VLLMResponse: 생성 결과
        """
        client = await self._get_client()

        request_model = model or self.config.default_model
        if not request_model:
            raise ValueError("Model name is required")

        # LoRA 어댑터 설정
        if lora_adapter and self.config.enable_lora_routing:
            request_model = f"{request_model}:{lora_adapter}"

        response = await client.chat.completions.create(
            model=request_model,
            messages=messages,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature or self.config.default_temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

        choice = response.choices[0]
        return VLLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason,
            lora_adapter=lora_adapter,
        )

    async def stream(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        lora_adapter: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[VLLMStreamChunk]:
        """
        스트리밍 텍스트 생성

        Args:
            prompt: 입력 프롬프트
            model: 모델 이름
            max_tokens: 최대 토큰 수
            temperature: 샘플링 temperature
            lora_adapter: 사용할 LoRA 어댑터 이름

        Yields:
            VLLMStreamChunk: 스트리밍 청크
        """
        client = await self._get_client()

        request_model = model or self.config.default_model
        if not request_model:
            raise ValueError("Model name is required")

        if lora_adapter and self.config.enable_lora_routing:
            request_model = f"{request_model}:{lora_adapter}"

        stream = await client.completions.create(
            model=request_model,
            prompt=prompt,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature or self.config.default_temperature,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                yield VLLMStreamChunk(
                    content=choice.text or "",
                    finish_reason=choice.finish_reason,
                    is_final=choice.finish_reason is not None,
                )

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        lora_adapter: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[VLLMStreamChunk]:
        """
        스트리밍 채팅 완성

        Args:
            messages: 메시지 리스트
            model: 모델 이름
            max_tokens: 최대 토큰 수
            temperature: 샘플링 temperature
            lora_adapter: 사용할 LoRA 어댑터 이름

        Yields:
            VLLMStreamChunk: 스트리밍 청크
        """
        client = await self._get_client()

        request_model = model or self.config.default_model
        if not request_model:
            raise ValueError("Model name is required")

        if lora_adapter and self.config.enable_lora_routing:
            request_model = f"{request_model}:{lora_adapter}"

        stream = await client.chat.completions.create(
            model=request_model,
            messages=messages,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature or self.config.default_temperature,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                yield VLLMStreamChunk(
                    content=delta.content or "" if delta else "",
                    finish_reason=choice.finish_reason,
                    is_final=choice.finish_reason is not None,
                )

    async def list_models(self) -> List[Dict[str, Any]]:
        """서버에 로드된 모델 목록 조회"""
        client = await self._get_client()
        response = await client.models.list()
        return [{"id": m.id, "object": m.object} for m in response.data]

    async def health_check(self) -> bool:
        """서버 헬스 체크"""
        try:
            http_client = await self._get_http_client()
            response = await http_client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_server_info(self) -> Dict[str, Any]:
        """서버 정보 조회 (vLLM 확장 API)"""
        try:
            http_client = await self._get_http_client()
            response = await http_client.get("/version")
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}

    async def close(self):
        """리소스 정리"""
        if self._client:
            await self._client.close()
            self._client = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
