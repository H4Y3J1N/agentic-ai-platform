"""
Chat Routes - SSE 스트리밍 지원
"""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from ..dependencies import get_agent_executor, get_current_user
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["chat"])


class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """채팅 응답"""
    result: str
    user_id: str
    session_id: Optional[str] = None


@router.post("/chat")
async def chat(
    request: ChatRequest,
    executor = Depends(get_agent_executor),
    user = Depends(get_current_user)
) -> ChatResponse:
    """일반 채팅 (비스트리밍)"""
    context = {
        "user_id": user.id,
        "session_id": request.session_id,
        "domain": "internal-ops",
        **(request.context or {})
    }

    result = await executor.execute(request.message, context)

    return ChatResponse(
        result=result,
        user_id=user.id,
        session_id=request.session_id
    )


@router.post("/chat/stream")
async def stream_chat_post(
    request: ChatRequest,
    executor = Depends(get_agent_executor),
    user = Depends(get_current_user)
):
    """
    SSE 기반 스트리밍 채팅 (POST)

    Streamlit 앱과 호환되는 POST 기반 SSE 엔드포인트
    """
    context = {
        "user_id": user.id,
        "session_id": request.session_id,
        "domain": "internal-ops",
        **(request.context or {})
    }

    async def generate():
        try:
            # Agent 실행 스트리밍
            async for chunk in executor.execute_stream(request.message, context):
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            # 완료 신호
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # nginx 버퍼링 비활성화
        }
    )


@router.get("/chat/stream")
async def stream_chat_get(
    task: str = Query(..., description="실행할 작업"),
    session_id: Optional[str] = Query(None),
    executor = Depends(get_agent_executor),
    user = Depends(get_current_user)
):
    """SSE 기반 스트리밍 채팅 (GET - 레거시 호환)"""
    context = {
        "user_id": user.id,
        "session_id": session_id,
        "domain": "internal-ops"
    }

    async def generate():
        # 시작 이벤트
        yield f"event: start\ndata: {json.dumps({'status': 'started', 'user_id': user.id})}\n\n"

        try:
            # Agent 실행 스트리밍
            async for chunk in executor.execute_stream(task, context):
                yield f"event: message\ndata: {json.dumps({'content': chunk})}\n\n"

            # 완료 이벤트
            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
