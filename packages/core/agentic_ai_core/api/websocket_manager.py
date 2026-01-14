"""
WebSocket Connection Manager
"""

from fastapi import WebSocket
from typing import Dict, List
import json


class WebSocketManager:
    """WebSocket 연결 관리"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """연결 수락"""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """연결 해제"""
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: dict):
        """세션에 메시지 전송"""
        if session_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            for conn in disconnected:
                self.disconnect(conn, session_id)
    
    async def broadcast(self, message: dict):
        """모든 연결에 브로드캐스트"""
        for session_id in list(self.active_connections.keys()):
            await self.send_message(session_id, message)
    
    def get_connection_count(self, session_id: str = None) -> int:
        """연결 수 반환"""
        if session_id:
            return len(self.active_connections.get(session_id, []))
        return sum(len(conns) for conns in self.active_connections.values())
