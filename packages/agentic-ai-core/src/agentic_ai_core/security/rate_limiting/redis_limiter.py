"""
Redis-based Distributed Rate Limiting
"""

import redis.asyncio as redis
from typing import Optional
import time


class RedisRateLimiter:
    """Redis 기반 분산 Rate Limiter"""
    
    def __init__(self, redis_url: str, prefix: str = "rate_limit"):
        """
        Args:
            redis_url: Redis 연결 URL
            prefix: Redis 키 프리픽스
        """
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
    
    async def allow_request(
        self,
        user_id: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """
        Sliding Window 알고리즘으로 요청 허용 여부 확인
        
        Args:
            user_id: 사용자 ID
            max_requests: 윈도우 내 최대 요청 수
            window_seconds: 윈도우 크기 (초)
        """
        key = f"{self.prefix}:{user_id}"
        now = time.time()
        window_start = now - window_seconds
        
        # 파이프라인으로 원자적 실행
        pipe = self.redis.pipeline()
        
        # 오래된 요청 제거
        pipe.zremrangebyscore(key, 0, window_start)
        
        # 현재 요청 수 확인
        pipe.zcard(key)
        
        # 새 요청 추가
        pipe.zadd(key, {str(now): now})
        
        # 만료 시간 설정
        pipe.expire(key, window_seconds)
        
        results = await pipe.execute()
        request_count = results[1]
        
        return request_count < max_requests
    
    async def get_request_count(
        self,
        user_id: str,
        window_seconds: int
    ) -> int:
        """현재 윈도우 내 요청 수 반환"""
        key = f"{self.prefix}:{user_id}"
        now = time.time()
        window_start = now - window_seconds
        
        return await self.redis.zcount(key, window_start, now)
    
    async def reset(self, user_id: str):
        """특정 사용자 제한 초기화"""
        key = f"{self.prefix}:{user_id}"
        await self.redis.delete(key)
    
    async def close(self):
        """Redis 연결 종료"""
        await self.redis.close()
