"""
Internal-Ops Agentic AI Service - FastAPI Application
Notion RAG 검색 및 내부 지식 관리
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import chat, websocket, health, knowledge
from .middleware.rate_limit_middleware import RateLimitMiddleware
import signal
import asyncio

app = FastAPI(
    title="Internal-Ops Agentic AI Service",
    description="내부 운영 AI 어시스턴트 - Notion RAG 검색, 지식 Q&A",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
app.add_middleware(RateLimitMiddleware, capacity=100, refill_rate=10)

# 라우터
app.include_router(chat.router)
app.include_router(websocket.router)
app.include_router(health.router)
app.include_router(knowledge.router)


# Graceful Shutdown 핸들러
def handle_sigterm(signum, frame):
    """SIGTERM 수신 시 Graceful Shutdown"""
    print("SIGTERM received, starting graceful shutdown...")
    health.set_shutdown()
    asyncio.create_task(graceful_shutdown())


async def graceful_shutdown():
    """Graceful Shutdown 프로세스"""
    print("Waiting for active connections to complete (30s)...")
    await asyncio.sleep(30)
    print("Graceful shutdown complete")


# SIGTERM 핸들러 등록
signal.signal(signal.SIGTERM, handle_sigterm)


@app.on_event("startup")
async def startup():
    """서비스 시작"""
    print("Internal-Ops Agentic AI Service starting...")

    # 의존성 연결 (PostgreSQL, Redis, ChromaDB)
    await asyncio.sleep(1)

    health.set_ready(True)
    print("Service is ready to accept traffic")


@app.on_event("shutdown")
async def shutdown():
    """서비스 종료"""
    print("Internal-Ops Agentic AI Service shutting down...")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Internal-Ops Agentic AI",
        "version": "1.0.0",
        "features": [
            "Notion RAG Search",
            "Knowledge Q&A",
            "SSE Streaming",
            "WebSocket"
        ],
        "endpoints": {
            "knowledge_search": "/knowledge/search",
            "knowledge_ask": "/knowledge/ask",
            "knowledge_page": "/knowledge/page/{page_id}",
            "chat": "/agent/chat",
            "stream": "/agent/chat/stream",
            "websocket": "/ws/chat/{session_id}",
            "health": "/health",
            "docs": "/docs"
        }
    }
