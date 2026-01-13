"""
Knowledge API Request/Response Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class KnowledgeSearchRequest(BaseModel):
    """Knowledge search request"""
    query: str = Field(..., description="Natural language search query", min_length=1)
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")


class SearchResult(BaseModel):
    """Single search result"""
    id: str
    content: str
    title: str
    url: str
    relevance_score: float
    last_edited: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchResponse(BaseModel):
    """Knowledge search response"""
    query: str
    results: List[SearchResult]
    total: int


class KnowledgeAskRequest(BaseModel):
    """Knowledge Q&A request"""
    question: str = Field(..., description="Question to answer", min_length=1)


class KnowledgeAskResponse(BaseModel):
    """Knowledge Q&A response"""
    question: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class PageContentResponse(BaseModel):
    """Page content response"""
    id: str
    title: str
    url: str
    content: str
    last_edited: str
    created: str
    icon: Optional[str] = None
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None


class PageSummaryResponse(BaseModel):
    """Page summary response"""
    id: str
    title: str
    url: str
    last_edited: str
    created: str
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None
    icon: Optional[str] = None
    archived: bool = False


class CollectionStatsResponse(BaseModel):
    """Collection statistics response"""
    collection_name: str
    document_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
