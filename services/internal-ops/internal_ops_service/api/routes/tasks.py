"""
Task Management API Routes - Create and manage tasks in Notion
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging

from ..schemas.tasks import (
    TaskCreateRequest,
    TaskCreateResponse,
    ProjectInfo,
    ProjectSearchResponse,
    ProjectListResponse,
    TaskDbSchemaResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])

# Lazy-loaded instance
_task_tool = None


def get_task_tool():
    """Get or create TaskCreationTool instance"""
    global _task_tool
    if _task_tool is None:
        from internal_ops_service.tools import TaskCreationTool
        _task_tool = TaskCreationTool()
    return _task_tool


@router.post("/create", response_model=TaskCreateResponse)
async def create_task(request: TaskCreateRequest):
    """
    Create a new task in the Notion Task database.

    Optionally links the task to an existing project via Relation.
    You can specify either project_id (direct) or project_name (search).
    """
    try:
        tool = get_task_tool()
        result = await tool.create_task(
            title=request.title,
            project_id=request.project_id,
            project_name=request.project_name,
            additional_properties=request.properties
        )

        return TaskCreateResponse(
            id=result["id"],
            title=result["title"],
            url=result["url"],
            project_id=result.get("project_id"),
            project_title=result.get("project_title")
        )

    except ValueError as e:
        logger.error(f"Task creation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Task creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/search", response_model=ProjectSearchResponse)
async def search_projects(
    query: str = Query(..., description="Project name search query", min_length=1),
    limit: int = Query(5, ge=1, le=20, description="Max results to return")
):
    """
    Search for projects by name.

    Use this to find project IDs for linking when creating tasks.
    """
    try:
        tool = get_task_tool()
        projects = await tool.search_projects(query, limit)

        return ProjectSearchResponse(
            query=query,
            projects=[ProjectInfo(**p) for p in projects],
            total=len(projects)
        )

    except ValueError as e:
        logger.error(f"Project search validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Project search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    limit: int = Query(20, ge=1, le=100, description="Max results to return")
):
    """
    List available projects.

    Returns a list of projects that can be linked to tasks.
    """
    try:
        tool = get_task_tool()
        projects = await tool.list_projects(limit)

        return ProjectListResponse(
            projects=[ProjectInfo(**p) for p in projects],
            total=len(projects)
        )

    except ValueError as e:
        logger.error(f"Project list validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Project list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema", response_model=TaskDbSchemaResponse)
async def get_task_db_schema():
    """
    Get the Task database schema.

    Returns the available properties and their types for task creation.
    """
    try:
        tool = get_task_tool()
        schema = await tool.get_task_db_schema()

        return TaskDbSchemaResponse(
            id=schema["id"],
            title=schema["title"],
            properties=schema["properties"]
        )

    except ValueError as e:
        logger.error(f"Schema fetch validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Schema fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
