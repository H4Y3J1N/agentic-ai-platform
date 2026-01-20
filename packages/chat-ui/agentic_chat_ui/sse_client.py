"""
SSE Client

Server-Sent Events 클라이언트 및 스트리밍 유틸리티
"""

from dataclasses import dataclass
from typing import AsyncIterator, Optional, Dict, Any, Callable
import json
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class SSEConfig:
    """SSE 클라이언트 설정"""
    base_url: str = "http://localhost:8000"
    timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class SSEClient:
    """Server-Sent Events 클라이언트"""

    def __init__(self, config: Optional[SSEConfig] = None):
        self.config = config or SSEConfig()
        self._client = None

    async def _ensure_client(self):
        """HTTP 클라이언트 초기화"""
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("httpx is required. Install with: pip install httpx")

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout)
            )

    async def close(self):
        """클라이언트 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def stream(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> AsyncIterator[str]:
        """
        SSE 스트림 연결

        Args:
            endpoint: API 엔드포인트
            data: 요청 데이터
            method: HTTP 메서드

        Yields:
            스트림 청크
        """
        await self._ensure_client()

        try:
            async with self._client.stream(
                method,
                endpoint,
                json=data,
                headers={"Accept": "text/event-stream"}
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # "data: " 제거

                        # [DONE] 시그널 처리
                        if data_str.strip() == "[DONE]":
                            break

                        # JSON 파싱 시도
                        try:
                            parsed = json.loads(data_str)
                            if isinstance(parsed, dict):
                                content = parsed.get("content", parsed.get("text", ""))
                                if content:
                                    yield content
                            else:
                                yield data_str
                        except json.JSONDecodeError:
                            # 일반 텍스트
                            yield data_str

        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            raise

    async def stream_with_events(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        이벤트 타입을 포함한 SSE 스트림

        Yields:
            {"event": "message", "data": "..."}
        """
        await self._ensure_client()

        try:
            async with self._client.stream(
                method,
                endpoint,
                json=data,
                headers={"Accept": "text/event-stream"}
            ) as response:
                response.raise_for_status()

                current_event = "message"
                current_data = []

                async for line in response.aiter_lines():
                    line = line.strip()

                    if not line:
                        # 빈 줄 = 이벤트 완료
                        if current_data:
                            data_str = "\n".join(current_data)
                            yield {
                                "event": current_event,
                                "data": data_str
                            }
                            current_data = []
                            current_event = "message"
                        continue

                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                    elif line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str != "[DONE]":
                            current_data.append(data_str)

                # 마지막 이벤트 처리
                if current_data:
                    yield {
                        "event": current_event,
                        "data": "\n".join(current_data)
                    }

        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield {"event": "error", "data": str(e)}


async def stream_chat_response(
    api_url: str,
    message: str,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    on_chunk: Optional[Callable[[str], None]] = None
) -> AsyncIterator[str]:
    """
    채팅 응답 스트리밍 헬퍼 함수

    Args:
        api_url: 전체 API URL (예: http://localhost:8000/agent/chat/stream)
        message: 사용자 메시지
        session_id: 세션 ID
        context: 추가 컨텍스트
        timeout: 타임아웃 (초)
        on_chunk: 청크 콜백 함수

    Yields:
        응답 청크
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required. Install with: pip install httpx")

    # URL 파싱
    from urllib.parse import urlparse
    parsed = urlparse(api_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    endpoint = parsed.path

    request_data = {
        "message": message,
        "session_id": session_id,
        "context": context or {}
    }

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout)
    ) as client:
        async with client.stream(
            "POST",
            endpoint,
            json=request_data,
            headers={"Accept": "text/event-stream"}
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        parsed_data = json.loads(data_str)
                        content = parsed_data.get("content", parsed_data.get("text", ""))
                    except json.JSONDecodeError:
                        content = data_str

                    if content:
                        if on_chunk:
                            on_chunk(content)
                        yield content


def sync_stream_chat_response(
    api_url: str,
    message: str,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0
):
    """
    동기 방식 채팅 스트리밍 (Streamlit용)

    Streamlit은 async를 직접 지원하지 않으므로 동기 래퍼 제공
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required. Install with: pip install httpx")

    from urllib.parse import urlparse
    parsed = urlparse(api_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    endpoint = parsed.path

    request_data = {
        "message": message,
        "session_id": session_id,
        "context": context or {}
    }

    with httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(timeout)
    ) as client:
        with client.stream(
            "POST",
            endpoint,
            json=request_data,
            headers={"Accept": "text/event-stream"}
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        parsed_data = json.loads(data_str)
                        content = parsed_data.get("content", parsed_data.get("text", ""))
                    except json.JSONDecodeError:
                        content = data_str

                    if content:
                        yield content
