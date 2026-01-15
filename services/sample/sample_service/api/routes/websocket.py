"""
WebSocket Routes
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from agentic_core.api import WebSocketManager
import json

router = APIRouter(tags=["websocket"])
ws_manager = WebSocketManager()


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket 채팅 엔드포인트
    
    클라이언트 메시지 형식:
    {
        "type": "message",
        "task": "주문 상태 확인",
        "user_id": 123
    }
    """
    await ws_manager.connect(websocket, session_id)
    
    try:
        while True:
            # 클라이언트 메시지 수신
            data = await websocket.receive_json()
            
            message_type = data.get("type", "message")
            
            if message_type == "ping":
                await ws_manager.send_message(session_id, {"type": "pong"})
            
            elif message_type == "message":
                task = data.get("task", "")
                user_id = data.get("user_id")
                
                # TODO: Agent 실행 로직 추가
                # result = await agent_executor.execute(task, {"user_id": user_id})
                
                # 임시 응답
                await ws_manager.send_message(session_id, {
                    "type": "response",
                    "content": f"Received: {task}",
                    "session_id": session_id
                })
            
            elif message_type == "close":
                break
    
    except WebSocketDisconnect:
        pass
    
    finally:
        ws_manager.disconnect(websocket, session_id)


@router.get("/ws/stats")
async def websocket_stats():
    """WebSocket 연결 통계"""
    return {
        "total_connections": ws_manager.get_connection_count(),
        "sessions": len(ws_manager.active_connections)
    }
