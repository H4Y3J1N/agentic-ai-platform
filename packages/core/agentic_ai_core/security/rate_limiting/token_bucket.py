"""
Token Bucket Rate Limiting Algorithm
"""

from datetime import datetime, timedelta
from typing import Dict
import asyncio


class TokenBucket:
    """Token Bucket 알고리즘 기반 Rate Limiter"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: 버킷 최대 용량 (토큰 수)
            refill_rate: 초당 토큰 리필 속도
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, dict] = {}
    
    async def allow_request(self, user_id: str) -> bool:
        """요청 허용 여부 확인"""
        now = datetime.now()
        
        if user_id not in self.buckets:
            self.buckets[user_id] = {
                "tokens": self.capacity,
                "last_refill": now
            }
        
        bucket = self.buckets[user_id]
        
        # 토큰 리필
        elapsed = (now - bucket["last_refill"]).total_seconds()
        refill_amount = elapsed * self.refill_rate
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + refill_amount)
        bucket["last_refill"] = now
        
        # 토큰 소비
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False
    
    def get_remaining_tokens(self, user_id: str) -> float:
        """남은 토큰 수 반환"""
        if user_id not in self.buckets:
            return self.capacity
        return self.buckets[user_id]["tokens"]
    
    def reset(self, user_id: str = None):
        """버킷 초기화"""
        if user_id:
            if user_id in self.buckets:
                del self.buckets[user_id]
        else:
            self.buckets.clear()
