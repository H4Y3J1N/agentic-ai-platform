"""
Notion Page Tool - Direct Notion page content retrieval
"""

from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class NotionPageTool:
    """Direct Notion page content retrieval"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.name = "NotionPageTool"
        self.description = "Retrieve full content of a specific Notion page"

        self._client = None

    async def _ensure_initialized(self):
        """Lazy initialization of Notion client"""
        if self._client is not None:
            return

        from ..integrations.notion import NotionClient, NotionClientConfig

        api_key = self.config.get("api_key") or os.environ.get("NOTION_API_KEY")
        if not api_key:
            raise ValueError("NOTION_API_KEY environment variable required")

        notion_config = NotionClientConfig(api_key=api_key)
        self._client = NotionClient(notion_config)

        logger.info("NotionPageTool initialized")

    async def get_page_content(
        self,
        page_id: str,
        include_children: bool = True
    ) -> Dict[str, Any]:
        """
        Get full content of a Notion page.

        Args:
            page_id: Notion page ID or URL
            include_children: Whether to include nested content

        Returns:
            Page content with metadata
        """
        await self._ensure_initialized()

        page = await self._client.get_page(page_id)
        blocks = await self._client.get_page_blocks(
            page_id,
            recursive=include_children
        )

        # Convert blocks to text
        content_lines = []
        for block in blocks:
            text = block.to_text(include_children=True)
            if text.strip():
                content_lines.append(text)

        content = "\n\n".join(content_lines)

        return {
            "id": page.id,
            "title": page.title,
            "url": page.url,
            "content": content,
            "last_edited": page.last_edited_time.isoformat(),
            "created": page.created_time.isoformat(),
            "icon": page.icon,
            "parent_type": page.parent_type,
            "parent_id": page.parent_id
        }

    async def get_page_summary(
        self,
        page_id: str
    ) -> Dict[str, Any]:
        """
        Get page metadata without full content.

        Args:
            page_id: Notion page ID

        Returns:
            Page metadata
        """
        await self._ensure_initialized()

        page = await self._client.get_page(page_id)

        return {
            "id": page.id,
            "title": page.title,
            "url": page.url,
            "last_edited": page.last_edited_time.isoformat(),
            "created": page.created_time.isoformat(),
            "parent_type": page.parent_type,
            "parent_id": page.parent_id,
            "icon": page.icon,
            "archived": page.archived
        }

    async def list_child_pages(
        self,
        page_id: str
    ) -> list:
        """
        List child pages of a given page.

        Args:
            page_id: Parent page ID

        Returns:
            List of child page summaries
        """
        await self._ensure_initialized()

        blocks = await self._client.get_page_blocks(page_id, recursive=False)

        child_pages = []
        for block in blocks:
            if block.type.value == "child_page":
                child_pages.append({
                    "id": block.id,
                    "title": block.content,
                    "type": "page"
                })
            elif block.type.value == "child_database":
                child_pages.append({
                    "id": block.id,
                    "title": block.content,
                    "type": "database"
                })

        return child_pages

    async def close(self):
        """Close the Notion client"""
        if self._client:
            await self._client.close()
            self._client = None
