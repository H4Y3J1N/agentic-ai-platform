"""
API Dependencies - FastAPI Dependency Injection
"""

from fastapi import Header, HTTPException
from typing import Optional
from dataclasses import dataclass


@dataclass
class User:
    """Current user context"""
    id: str
    name: str = "Anonymous"


@dataclass
class AgentExecutor:
    """Agent executor placeholder"""

    async def execute(self, task: str, context: dict) -> str:
        """Execute task (placeholder)"""
        return f"Processed: {task}"

    async def execute_stream(self, task: str, context: dict):
        """Stream task execution (placeholder)"""
        yield f"Processing: {task}"
        yield "Done"


_executor = None


def get_agent_executor() -> AgentExecutor:
    """Get or create agent executor instance"""
    global _executor
    if _executor is None:
        _executor = AgentExecutor()
    return _executor


async def get_current_user(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    authorization: Optional[str] = Header(None)
) -> User:
    """Extract current user from request headers"""
    user_id = x_user_id or "anonymous"
    return User(id=user_id)
