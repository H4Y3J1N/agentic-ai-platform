"""
Rate Limiting Middleware
"""

from fastapi import Request, HTTPException
from agentic_ai_core.security.rate_limiting.token_bucket import TokenBucket
from starlette.middleware.base import BaseHTTPMiddleware
import time


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate Limiting 미들웨어"""
    
    def __init__(self, app, capacity: int = 100, refill_rate: float = 10):
        """
        Args:
            capacity: 버킷 최대 용량
            refill_rate: 초당 리필 속도
        """
        super().__init__(app)
        self.rate_limiter = TokenBucket(capacity=capacity, refill_rate=refill_rate)
    
    async def dispatch(self, request: Request, call_next):
        # 사용자 ID 추출 (인증된 경우)
        user_id = self._get_user_id(request)
        
        # Rate Limit 확인
        if not await self.rate_limiter.allow_request(user_id):
            remaining_tokens = self.rate_limiter.get_remaining_tokens(user_id)
            
            raise HTTPException(
                status_code=429,
                detail="Too Many Requests",
                headers={
                    "X-RateLimit-Limit": str(self.rate_limiter.capacity),
                    "X-RateLimit-Remaining": str(int(remaining_tokens)),
                    "Retry-After": "1"
                }
            )
        
        # 요청 처리
        response = await call_next(request)
        
        # Rate Limit 헤더 추가
        remaining_tokens = self.rate_limiter.get_remaining_tokens(user_id)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.capacity)
        response.headers["X-RateLimit-Remaining"] = str(int(remaining_tokens))
        
        return response
    
    def _get_user_id(self, request: Request) -> str:
        """사용자 ID 추출"""
        # 인증된 사용자
        if hasattr(request.state, "user"):
            return f"user_{request.state.user.id}"
        
        # 미인증 사용자는 IP 기반
        client_ip = request.client.host if request.client else "unknown"
        return f"ip_{client_ip}"
