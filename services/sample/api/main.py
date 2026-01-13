"""
E-commerce Agentic AI Service - FastAPI Application
무중단 배포 지원 (Graceful Shutdown, Health Checks)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import chat, websocket, health
from .middleware.rate_limit_middleware import RateLimitMiddleware
import signal
import asyncio

app = FastAPI(
    title="E-commerce Agentic AI Service",
    description="무중단 배포 지원 - SSE, WebSocket, Rate Limiting",
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


# Graceful Shutdown 핸들러
def handle_sigterm(signum, frame):
    """SIGTERM 수신 시 Graceful Shutdown"""
    print("🛑 SIGTERM received, starting graceful shutdown...")
    health.set_shutdown()
    
    # Connection Draining: 새 요청 차단, 기존 요청 처리 완료 대기
    asyncio.create_task(graceful_shutdown())


async def graceful_shutdown():
    """Graceful Shutdown 프로세스"""
    print("⏳ Waiting for active connections to complete (30s)...")
    await asyncio.sleep(30)  # Connection Draining 시간
    print("✅ Graceful shutdown complete")


# SIGTERM 핸들러 등록
signal.signal(signal.SIGTERM, handle_sigterm)


@app.on_event("startup")
async def startup():
    """서비스 시작"""
    print("🚀 E-commerce Agentic AI Service starting...")
    
    # 의존성 연결 (PostgreSQL, Redis, Milvus)
    await asyncio.sleep(1)  # 초기화 시뮬레이션
    
    # Ready 상태로 전환
    health.set_ready(True)
    print("✅ Service is ready to accept traffic")


@app.on_event("shutdown")
async def shutdown():
    """서비스 종료"""
    print("👋 E-commerce Agentic AI Service shutting down...")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "E-commerce Agentic AI",
        "version": "1.0.0",
        "features": ["SSE", "WebSocket", "Rate Limiting", "Zero-Downtime Deployment"],
        "endpoints": {
            "chat": "/agent/chat",
            "stream": "/agent/chat/stream",
            "websocket": "/ws/chat/{session_id}",
            "health": "/health (liveness)",
            "readiness": "/readiness (readiness)",
            "startup": "/startup (startup)",
            "docs": "/docs"
        }
    }
