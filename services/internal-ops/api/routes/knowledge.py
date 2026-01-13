"""
Knowledge API Routes - Notion RAG Search and Q&A
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import logging

from ..schemas.knowledge import (
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    SearchResult,
    KnowledgeAskResponse,
    PageContentResponse,
    PageSummaryResponse,
    CollectionStatsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# Lazy-loaded instances
_knowledge_agent = None
_search_tool = None
_page_tool = None


def get_knowledge_agent():
    """Get or create KnowledgeAgent instance"""
    global _knowledge_agent
    if _knowledge_agent is None:
        from src.internal_ops_service.agents import KnowledgeAgent
        _knowledge_agent = KnowledgeAgent()
    return _knowledge_agent


def get_search_tool():
    """Get or create NotionSearchTool instance"""
    global _search_tool
    if _search_tool is None:
        from src.internal_ops_service.tools import NotionSearchTool
        _search_tool = NotionSearchTool()
    return _search_tool


def get_page_tool():
    """Get or create NotionPageTool instance"""
    global _page_tool
    if _page_tool is None:
        from src.internal_ops_service.tools import NotionPageTool
        _page_tool = NotionPageTool()
    return _page_tool


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(request: KnowledgeSearchRequest):
    """
    Semantic search over internal knowledge base.

    Uses RAG to find relevant documents from indexed Notion pages.
    """
    try:
        tool = get_search_tool()
        results = await tool.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        formatted_results = [
            SearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                title=r.get("title", "Untitled"),
                url=r.get("url", ""),
                relevance_score=r.get("relevance_score", 0.0),
                last_edited=r.get("last_edited"),
                metadata=r.get("metadata", {})
            )
            for r in results
        ]

        return KnowledgeSearchResponse(
            query=request.query,
            results=formatted_results,
            total=len(formatted_results)
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=KnowledgeAskResponse)
async def ask_knowledge(
    question: str = Query(..., description="Question to answer", min_length=1)
):
    """
    Ask a question and get an AI-generated answer with citations.

    The answer is generated using RAG over the indexed knowledge base.
    """
    try:
        agent = get_knowledge_agent()
        answer = await agent.execute(question)

        return KnowledgeAskResponse(
            question=question,
            answer=answer,
            sources=[]
        )

    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ask/stream")
async def ask_knowledge_stream(
    question: str = Query(..., description="Question to answer", min_length=1)
):
    """
    Ask a question with streaming response.

    Returns Server-Sent Events with the AI-generated answer.
    """
    async def generate():
        try:
            agent = get_knowledge_agent()

            yield f"event: start\ndata: {json.dumps({'status': 'started', 'question': question})}\n\n"

            async for chunk in agent.execute_stream(question):
                yield f"event: message\ndata: {json.dumps({'content': chunk})}\n\n"

            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/page/{page_id}", response_model=PageContentResponse)
async def get_page_content(
    page_id: str,
    include_children: bool = Query(True, description="Include nested content")
):
    """
    Get full content of a specific Notion page.

    Requires NOTION_API_KEY environment variable.
    """
    try:
        tool = get_page_tool()
        content = await tool.get_page_content(
            page_id=page_id,
            include_children=include_children
        )

        return PageContentResponse(**content)

    except Exception as e:
        logger.error(f"Page fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/page/{page_id}/summary", response_model=PageSummaryResponse)
async def get_page_summary(page_id: str):
    """
    Get page metadata without full content.
    """
    try:
        tool = get_page_tool()
        summary = await tool.get_page_summary(page_id)

        return PageSummaryResponse(**summary)

    except Exception as e:
        logger.error(f"Page summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=CollectionStatsResponse)
async def get_collection_stats():
    """
    Get statistics about the indexed document collection.
    """
    try:
        tool = get_search_tool()
        stats = await tool.get_collection_stats()

        return CollectionStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
