"""
Health Check Endpoints - K8s Probes
"""

from fastapi import APIRouter, status, Response
from typing import Dict, Any
import time

router = APIRouter(tags=["health"])

_startup_time = time.time()
_is_ready = False
_is_shutting_down = False


@router.get("/health")
async def liveness_probe():
    """Liveness Probe - Pod 재시작 기준"""
    if _is_shutting_down:
        return Response(status_code=503)
    return {"status": "healthy"}


@router.get("/readiness")
async def readiness_probe():
    """Readiness Probe - 트래픽 라우팅 기준"""
    if _is_shutting_down or not _is_ready:
        return Response(status_code=503)
    
    checks = await check_dependencies()
    if not all(checks.values()):
        return Response(status_code=503)
    
    return {"status": "ready", "checks": checks}


@router.get("/startup")
async def startup_probe():
    """Startup Probe - 초기화 완료 기준"""
    if not _is_ready:
        return Response(status_code=503)
    return {"status": "started", "elapsed": round(time.time() - _startup_time, 2)}


async def check_dependencies() -> Dict[str, bool]:
    """의존성 서비스 체크"""
    return {
        "postgres": True,  # TODO: 실제 체크
        "redis": True,
        "milvus": True
    }


def set_ready(ready: bool):
    global _is_ready
    _is_ready = ready


def set_shutdown():
    global _is_shutting_down
    _is_shutting_down = True
