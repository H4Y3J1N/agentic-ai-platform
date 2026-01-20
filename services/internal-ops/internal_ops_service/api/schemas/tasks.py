"""
Task API Request/Response Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class TaskCreateRequest(BaseModel):
    """Task creation request"""
    title: str = Field(..., description="Task title (태스크 이름)", min_length=1)
    project_id: Optional[str] = Field(None, description="Direct project page ID to link")
    project_name: Optional[str] = Field(None, description="Project name to search and link")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional Notion properties")


class TaskCreateResponse(BaseModel):
    """Task creation response"""
    id: str
    title: str
    url: str
    project_id: Optional[str] = None
    project_title: Optional[str] = None


class ProjectInfo(BaseModel):
    """Project summary info"""
    id: str
    title: str
    url: str


class ProjectSearchRequest(BaseModel):
    """Project search request"""
    query: str = Field(..., description="Project name search query", min_length=1)
    limit: int = Field(5, ge=1, le=20, description="Max results to return")


class ProjectSearchResponse(BaseModel):
    """Project search response"""
    query: str
    projects: List[ProjectInfo]
    total: int


class ProjectListResponse(BaseModel):
    """Project list response"""
    projects: List[ProjectInfo]
    total: int


class TaskDbSchemaResponse(BaseModel):
    """Task database schema response"""
    id: str
    title: str
    properties: Dict[str, Dict[str, Any]]
