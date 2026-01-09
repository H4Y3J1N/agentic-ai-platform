"""
SSE (Server-Sent Events) Response Helper
"""

from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Any, Dict
import asyncio
import json


class SSEResponse:
    """SSE 응답 생성 및 관리"""
    
    @staticmethod
    async def event_generator(
        iterator: AsyncGenerator[Any, None],
        event_type: str = "message"
    ) -> AsyncGenerator[str, None]:
        """비동기 이터레이터를 SSE 포맷으로 변환"""
        try:
            async for chunk in iterator:
                data = json.dumps(chunk, ensure_ascii=False) if isinstance(chunk, dict) else str(chunk)
                yield f"event: {event_type}\ndata: {data}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    @classmethod
    def create_stream(cls, generator: AsyncGenerator[str, None]) -> StreamingResponse:
        """SSE StreamingResponse 생성"""
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
