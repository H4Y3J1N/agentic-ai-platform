"""
Notion Sync Manager - Batch synchronization of Notion content
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .client import NotionClient
from .models import NotionPage, NotionBlock

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Sync configuration"""
    # Page filters
    include_archived: bool = False
    page_ids_whitelist: Optional[List[str]] = None
    page_ids_blacklist: Optional[List[str]] = None

    # Database filters
    database_ids: Optional[List[str]] = None

    # Content options
    include_child_pages: bool = True
    max_depth: int = 5

    # Performance
    batch_size: int = 10
    concurrency: int = 5


@dataclass
class SyncedDocument:
    """Document ready for indexing"""
    id: str
    title: str
    content: str
    url: str
    source_type: str  # "page" or "database_row"
    parent_id: Optional[str]
    last_edited: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotionSyncManager:
    """Manages batch synchronization of Notion content"""

    def __init__(
        self,
        client: NotionClient,
        sync_config: Optional[SyncConfig] = None
    ):
        self.client = client
        self.config = sync_config or SyncConfig()
        self._seen_pages: set = set()

    async def sync_all(self) -> AsyncIterator[SyncedDocument]:
        """Sync all accessible pages from workspace"""
        logger.info("Starting full Notion workspace sync")
        self._seen_pages.clear()

        page_count = 0
        async for page in self.client.iterate_all_pages():
            if not self._should_sync_page(page):
                continue

            if page.id in self._seen_pages:
                continue
            self._seen_pages.add(page.id)

            try:
                doc = await self._process_page(page)
                if doc:
                    page_count += 1
                    if page_count % 10 == 0:
                        logger.info(f"Processed {page_count} pages...")
                    yield doc
            except Exception as e:
                logger.error(f"Error processing page {page.id} ({page.title}): {e}")

        # Sync specific databases if configured
        if self.config.database_ids:
            for db_id in self.config.database_ids:
                async for doc in self._sync_database(db_id):
                    yield doc

        logger.info(f"Full sync completed. Processed {page_count} pages.")

    async def sync_incremental(
        self,
        since: datetime
    ) -> AsyncIterator[SyncedDocument]:
        """Sync pages modified since given timestamp"""
        logger.info(f"Starting incremental sync since {since}")
        self._seen_pages.clear()

        page_count = 0
        async for page in self.client.iterate_all_pages():
            if page.last_edited_time <= since:
                continue

            if not self._should_sync_page(page):
                continue

            if page.id in self._seen_pages:
                continue
            self._seen_pages.add(page.id)

            try:
                doc = await self._process_page(page)
                if doc:
                    page_count += 1
                    yield doc
            except Exception as e:
                logger.error(f"Error processing page {page.id}: {e}")

        logger.info(f"Incremental sync completed. Updated {page_count} pages.")

    async def sync_pages(
        self,
        page_ids: List[str]
    ) -> AsyncIterator[SyncedDocument]:
        """Sync specific pages by ID"""
        logger.info(f"Syncing {len(page_ids)} specific pages")

        for page_id in page_ids:
            try:
                page = await self.client.get_page(page_id)
                doc = await self._process_page(page)
                if doc:
                    yield doc
            except Exception as e:
                logger.error(f"Error syncing page {page_id}: {e}")

    def _should_sync_page(self, page: NotionPage) -> bool:
        """Check if page should be synced based on filters"""
        # Skip archived unless configured
        if page.archived and not self.config.include_archived:
            return False

        # Whitelist check
        if self.config.page_ids_whitelist:
            return page.id in self.config.page_ids_whitelist

        # Blacklist check
        if self.config.page_ids_blacklist:
            if page.id in self.config.page_ids_blacklist:
                return False

        return True

    async def _process_page(self, page: NotionPage) -> Optional[SyncedDocument]:
        """Process a page into a SyncedDocument"""
        try:
            blocks = await self.client.get_page_blocks(
                page.id,
                recursive=self.config.include_child_pages
            )
        except Exception as e:
            logger.warning(f"Failed to fetch blocks for page {page.id}: {e}")
            blocks = []

        content = self._blocks_to_text(blocks)

        if not content.strip():
            logger.debug(f"Skipping empty page: {page.title}")
            return None

        return SyncedDocument(
            id=page.id,
            title=page.title,
            content=content,
            url=page.url,
            source_type="page",
            parent_id=page.parent_id,
            last_edited=page.last_edited_time,
            metadata={
                "notion_page_id": page.id,
                "notion_url": page.url,
                "title": page.title,
                "parent_type": page.parent_type,
                "parent_id": page.parent_id,
                "created_time": page.created_time.isoformat(),
                "last_edited_time": page.last_edited_time.isoformat(),
                "icon": page.icon,
                "source": "notion",
                "domain": "internal-ops"
            }
        )

    def _blocks_to_text(self, blocks: List[NotionBlock]) -> str:
        """Convert blocks to plain text for indexing"""
        lines = []

        for block in blocks:
            text = block.to_text(include_children=True)
            if text.strip():
                lines.append(text)

        return "\n\n".join(lines)

    async def _sync_database(
        self,
        database_id: str
    ) -> AsyncIterator[SyncedDocument]:
        """Sync all rows from a database"""
        try:
            db = await self.client.get_database(database_id)
            logger.info(f"Syncing database: {db.title}")

            async for row in self.client.query_database(database_id):
                if row.id in self._seen_pages:
                    continue
                self._seen_pages.add(row.id)

                try:
                    doc = await self._process_page(row)
                    if doc:
                        doc.source_type = "database_row"
                        doc.metadata["database_id"] = database_id
                        doc.metadata["database_title"] = db.title
                        yield doc
                except Exception as e:
                    logger.error(f"Error processing database row {row.id}: {e}")

        except Exception as e:
            logger.error(f"Error syncing database {database_id}: {e}")
