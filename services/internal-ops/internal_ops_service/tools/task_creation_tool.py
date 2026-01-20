"""
Task Creation Tool - Create tasks in Notion Task DB with Project relations
"""

from typing import Dict, Any, Optional, List
import os
import sys
import logging
import yaml
from pathlib import Path

# Add core package to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))

from agentic_core.tools import BaseTool, ToolResult, ToolConfig, ToolCapability, tool

logger = logging.getLogger(__name__)


@tool(
    name="task_creation",
    description="Create tasks in Notion Task database with project linking",
    capabilities=[ToolCapability.CREATE],
    tags=["notion", "task", "create"]
)
class TaskCreationTool(BaseTool):
    """Tool for creating tasks in Notion Task Database with Project linking"""

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        super().__init__(config)
        # Support legacy dict config
        self._extra_config = kwargs.get("legacy_config", {})

        self._client = None
        self._task_db_id = None
        self._project_db_id = None
        self._task_title_property = None
        self._task_project_relation_property = None

    async def initialize(self):
        """Lazy initialization of Notion client and database IDs"""
        if self._initialized:
            return

        from ..integrations.notion import NotionClient, NotionClientConfig

        api_key = self._extra_config.get("api_key") or os.environ.get("NOTION_API_KEY")
        if not api_key:
            raise ValueError("NOTION_API_KEY environment variable required")

        notion_config = NotionClientConfig(api_key=api_key)
        self._client = NotionClient(notion_config)

        # Load database IDs from config or environment
        self._load_database_config()

        self._initialized = True
        logger.info(
            f"TaskCreationTool initialized. "
            f"Task DB: {self._task_db_id}, Project DB: {self._project_db_id}"
        )

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute task creation (BaseTool interface).

        Args:
            title: Task title (required)
            project_id: Direct project page ID to link
            project_name: Project name to search and link
            additional_properties: Extra Notion properties

        Returns:
            ToolResult with created task info
        """
        title = kwargs.get("title", "")
        project_id = kwargs.get("project_id")
        project_name = kwargs.get("project_name")
        additional_properties = kwargs.get("additional_properties")

        if not title:
            return ToolResult.fail("Task title is required")

        try:
            result = await self.create_task(
                title=title,
                project_id=project_id,
                project_name=project_name,
                additional_properties=additional_properties
            )
            return ToolResult.ok(result, task_id=result.get("id"))
        except Exception as e:
            return ToolResult.fail(str(e))

    def _load_database_config(self):
        """Load database configuration from config file or environment"""
        # Try to load from config file
        config_path = Path(__file__).parent.parent.parent / "config" / "notion.yaml"
        file_config = {}

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}

        notion_config = file_config.get("notion", {})
        databases = notion_config.get("databases", {})
        task_creation = notion_config.get("task_creation", {})

        # Database IDs (config -> env -> error)
        self._project_db_id = (
            self._extra_config.get("project_database_id")
            or databases.get("project")
            or os.environ.get("NOTION_PROJECT_DB_ID")
        )

        self._task_db_id = (
            self._extra_config.get("task_database_id")
            or databases.get("task")
            or os.environ.get("NOTION_TASK_DB_ID")
        )

        # Property names for task creation
        self._task_title_property = (
            self._extra_config.get("task_title_property")
            or task_creation.get("task_title_property", "이름")
        )

        self._task_project_relation_property = (
            self._extra_config.get("task_project_relation_property")
            or task_creation.get("task_project_relation_property", "프로젝트")
        )

        if not self._project_db_id:
            raise ValueError(
                "Project database ID required. "
                "Set NOTION_PROJECT_DB_ID env var or configure in notion.yaml"
            )

        if not self._task_db_id:
            logger.warning(
                "Task database ID not configured. "
                "Will attempt to discover from Project DB relations."
            )

    async def _discover_task_db_id(self) -> Optional[str]:
        """Discover Task DB ID from Project DB's relation property"""
        if not self._project_db_id:
            return None

        try:
            db = await self._client.get_database(self._project_db_id)

            # Look for "관련 Task" relation property
            for prop_name, prop_config in db.properties.items():
                if "task" in prop_name.lower() or "태스크" in prop_name:
                    if prop_config.get("type") == "relation":
                        relation_config = prop_config.get("relation", {})
                        discovered_id = relation_config.get("database_id")
                        if discovered_id:
                            logger.info(
                                f"Discovered Task DB ID from '{prop_name}': {discovered_id}"
                            )
                            return discovered_id

            logger.warning("Could not discover Task DB ID from Project DB relations")
            return None

        except Exception as e:
            logger.error(f"Failed to discover Task DB ID: {e}")
            return None

    async def search_projects(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for projects by name to link tasks.

        Args:
            query: Project name search query
            limit: Max results to return

        Returns:
            List of matching projects with id, title, url
        """
        await self.initialize()

        if not self._project_db_id:
            raise ValueError("Project database ID not configured")

        projects = []
        count = 0

        # Query with title filter
        async for page in self._client.query_database(
            self._project_db_id,
            filter_conditions={
                "property": "제목",
                "title": {"contains": query}
            }
        ):
            projects.append({
                "id": page.id,
                "title": page.title,
                "url": page.url
            })
            count += 1
            if count >= limit:
                break

        logger.info(f"Found {len(projects)} projects matching '{query}'")
        return projects

    async def list_projects(
        self,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List available projects.

        Args:
            limit: Max results to return

        Returns:
            List of projects with id, title, url
        """
        await self.initialize()

        if not self._project_db_id:
            raise ValueError("Project database ID not configured")

        projects = []
        count = 0

        async for page in self._client.query_database(self._project_db_id):
            projects.append({
                "id": page.id,
                "title": page.title,
                "url": page.url
            })
            count += 1
            if count >= limit:
                break

        return projects

    async def create_task(
        self,
        title: str,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        additional_properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a task in the Task DB with optional project linking.

        Args:
            title: Task title
            project_id: Direct project page ID to link (takes precedence)
            project_name: Project name to search and link (if project_id not provided)
            additional_properties: Extra Notion properties to set

        Returns:
            Created task info with id, title, url, project_id
        """
        await self.initialize()

        # Ensure we have a Task DB ID
        if not self._task_db_id:
            self._task_db_id = await self._discover_task_db_id()
            if not self._task_db_id:
                raise ValueError(
                    "Task database ID not configured and could not be discovered. "
                    "Set NOTION_TASK_DB_ID env var or configure in notion.yaml"
                )

        # Resolve project ID from name if needed
        linked_project_id = project_id
        linked_project_title = None

        if not linked_project_id and project_name:
            projects = await self.search_projects(project_name, limit=1)
            if projects:
                linked_project_id = projects[0]["id"]
                linked_project_title = projects[0]["title"]
                logger.info(f"Resolved project '{project_name}' to ID: {linked_project_id}")
            else:
                logger.warning(f"Project not found: {project_name}")

        # Build properties
        properties = {}

        # Add project relation if we have a project ID
        if linked_project_id:
            properties[self._task_project_relation_property] = {
                "relation": [{"id": self._client._normalize_id(linked_project_id)}]
            }

        # Merge additional properties
        if additional_properties:
            properties.update(additional_properties)

        # Create the task
        page = await self._client.create_page(
            parent_id=self._task_db_id,
            parent_type="database_id",
            title=title,
            properties=properties if properties else None,
            title_property_name=self._task_title_property
        )

        logger.info(f"Created task: '{title}' (ID: {page.id})")

        result = {
            "id": page.id,
            "title": page.title,
            "url": page.url,
            "project_id": linked_project_id,
            "project_title": linked_project_title
        }

        return result

    async def get_task_db_schema(self) -> Dict[str, Any]:
        """
        Get the Task database schema to understand available properties.

        Returns:
            Database schema with property definitions
        """
        await self.initialize()

        if not self._task_db_id:
            self._task_db_id = await self._discover_task_db_id()
            if not self._task_db_id:
                raise ValueError("Task database ID not available")

        db = await self._client.get_database(self._task_db_id)

        return {
            "id": db.id,
            "title": db.title,
            "properties": {
                name: {
                    "type": prop.get("type"),
                    "id": prop.get("id")
                }
                for name, prop in db.properties.items()
            }
        }

    async def close(self):
        """Close the Notion client"""
        if self._client:
            await self._client.close()
            self._client = None
