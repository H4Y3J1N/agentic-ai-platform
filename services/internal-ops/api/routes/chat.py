"""
Chat Routes - SSE 스트리밍 지원
"""

from fastapi import APIRouter, Depends, Query
from agentic_ai_core.api.sse_response import SSEResponse
from ..dependencies import get_agent_executor, get_current_user
import json

router = APIRouter(prefix="/agent", tags=["chat"])


@router.post("/chat")
async def chat(
    task: str,
    executor = Depends(get_agent_executor),
    user = Depends(get_current_user)
):
    """일반 채팅 (비스트리밍)"""
    context = {"user_id": user.id, "domain": "ecommerce"}
    result = await executor.execute(task, context)
    return {"result": result, "user_id": user.id}


@router.get("/chat/stream")
async def stream_chat(
    task: str = Query(..., description="실행할 작업"),
    executor = Depends(get_agent_executor),
    user = Depends(get_current_user)
):
    """SSE 기반 스트리밍 채팅"""
    
    context = {"user_id": user.id, "domain": "ecommerce"}
    
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
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return SSEResponse.create_stream(generate())
