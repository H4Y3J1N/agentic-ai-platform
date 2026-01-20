"""
Notion API Client
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import httpx
import asyncio
import logging

from .models import NotionPage, NotionBlock, NotionDatabase

logger = logging.getLogger(__name__)


@dataclass
class NotionClientConfig:
    """Notion API client configuration"""
    api_key: str
    api_version: str = "2022-06-28"
    base_url: str = "https://api.notion.com/v1"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class NotionClient:
    """Notion API wrapper with rate limiting and pagination support"""

    def __init__(self, config: NotionClientConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Notion-Version": config.api_version,
            "Content-Type": "application/json"
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=self.headers,
                timeout=self.config.timeout
            )
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API request with retry logic"""
        client = await self._get_client()

        for attempt in range(self.config.max_retries):
            try:
                if method == "GET":
                    response = await client.get(endpoint, params=params)
                elif method == "POST":
                    response = await client.post(endpoint, json=json)
                elif method == "PATCH":
                    response = await client.patch(endpoint, json=json)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise Exception("Max retries exceeded")

    async def search_pages(
        self,
        query: str = "",
        filter_type: str = "page",
        page_size: int = 100,
        start_cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search workspace pages"""
        payload: Dict[str, Any] = {
            "filter": {"value": filter_type, "property": "object"},
            "page_size": min(page_size, 100)
        }
        if query:
            payload["query"] = query
        if start_cursor:
            payload["start_cursor"] = start_cursor

        return await self._request("POST", "/search", json=payload)

    async def get_page(self, page_id: str) -> NotionPage:
        """Get page metadata"""
        page_id = self._normalize_id(page_id)
        data = await self._request("GET", f"/pages/{page_id}")
        return NotionPage.from_api_response(data)

    async def get_page_blocks(
        self,
        block_id: str,
        recursive: bool = True
    ) -> List[NotionBlock]:
        """Get all blocks in a page (with optional recursion for nested blocks)"""
        block_id = self._normalize_id(block_id)
        blocks = []
        start_cursor = None

        while True:
            params: Dict[str, Any] = {"page_size": 100}
            if start_cursor:
                params["start_cursor"] = start_cursor

            data = await self._request(
                "GET",
                f"/blocks/{block_id}/children",
                params=params
            )

            for block_data in data.get("results", []):
                block = NotionBlock.from_api_response(block_data)
                blocks.append(block)

                # Recursively fetch children if block has them
                if recursive and block.has_children:
                    try:
                        children = await self.get_page_blocks(block.id, recursive=True)
                        block.children = children
                    except Exception as e:
                        logger.warning(f"Failed to fetch children for block {block.id}: {e}")

            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")

        return blocks

    async def iterate_all_pages(self) -> AsyncIterator[NotionPage]:
        """Iterate through all accessible pages in workspace"""
        start_cursor = None

        while True:
            result = await self.search_pages(
                filter_type="page",
                page_size=100,
                start_cursor=start_cursor
            )

            for page_data in result.get("results", []):
                yield NotionPage.from_api_response(page_data)

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

    async def get_database(self, database_id: str) -> NotionDatabase:
        """Get database schema"""
        database_id = self._normalize_id(database_id)
        data = await self._request("GET", f"/databases/{database_id}")
        return NotionDatabase.from_api_response(data)

    async def query_database(
        self,
        database_id: str,
        filter_conditions: Optional[Dict] = None,
        sorts: Optional[List[Dict]] = None,
        page_size: int = 100
    ) -> AsyncIterator[NotionPage]:
        """Query database rows as pages"""
        database_id = self._normalize_id(database_id)
        start_cursor = None

        while True:
            payload: Dict[str, Any] = {"page_size": min(page_size, 100)}
            if filter_conditions:
                payload["filter"] = filter_conditions
            if sorts:
                payload["sorts"] = sorts
            if start_cursor:
                payload["start_cursor"] = start_cursor

            data = await self._request(
                "POST",
                f"/databases/{database_id}/query",
                json=payload
            )

            for row_data in data.get("results", []):
                yield NotionPage.from_api_response(row_data)

            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")

    async def create_page(
        self,
        parent_id: str,
        parent_type: str,
        title: str,
        properties: Optional[Dict[str, Any]] = None,
        content_blocks: Optional[List[Dict]] = None,
        title_property_name: str = "title"
    ) -> NotionPage:
        """
        Create a new page in Notion.

        Args:
            parent_id: Database ID or Page ID
            parent_type: "database_id" or "page_id"
            title: Page title
            properties: Additional properties (relations, selects, etc.)
            content_blocks: Page content blocks
            title_property_name: Name of the title property (default "title",
                               but some DBs use different names like "이름" or "제목")

        Returns:
            Created NotionPage
        """
        parent_id = self._normalize_id(parent_id)

        # Build properties dict with title
        page_properties = {
            title_property_name: {"title": [{"text": {"content": title}}]}
        }

        # Merge additional properties
        if properties:
            page_properties.update(properties)

        payload: Dict[str, Any] = {
            "parent": {parent_type: parent_id},
            "properties": page_properties
        }

        if content_blocks:
            payload["children"] = content_blocks

        data = await self._request("POST", "/pages", json=payload)
        return NotionPage.from_api_response(data)

    async def update_page(
        self,
        page_id: str,
        properties: Dict[str, Any]
    ) -> NotionPage:
        """Update page properties (Post-MVP)"""
        page_id = self._normalize_id(page_id)
        data = await self._request(
            "PATCH",
            f"/pages/{page_id}",
            json={"properties": properties}
        )
        return NotionPage.from_api_response(data)

    def _normalize_id(self, id_or_url: str) -> str:
        """Normalize page/database ID (remove dashes, extract from URL)"""
        # Handle URL input
        if "notion.so" in id_or_url or "notion.site" in id_or_url:
            # Extract ID from URL: notion.so/Page-Title-abc123def456
            parts = id_or_url.rstrip("/").split("-")
            if parts:
                id_str = parts[-1].split("?")[0]
                return id_str.replace("-", "")

        return id_or_url.replace("-", "")

    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
