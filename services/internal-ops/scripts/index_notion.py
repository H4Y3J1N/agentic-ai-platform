#!/usr/bin/env python3
"""
Notion Indexing CLI

Usage:
    python index_notion.py full              # Full workspace sync
    python index_notion.py incremental       # Since last sync
    python index_notion.py pages PAGE_ID...  # Specific pages
    python index_notion.py status            # Show sync status
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project paths
SERVICE_ROOT = Path(__file__).parent.parent  # services/internal-ops
PROJECT_ROOT = SERVICE_ROOT.parent.parent     # agentic-ai-platform
sys.path.insert(0, str(SERVICE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NotionIndexer:
    """CLI for indexing Notion content into ChromaDB"""

    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else (
            SERVICE_ROOT / "config"
        )
        self.notion_config = self._load_config("notion.yaml")
        self.rag_config = self._load_config("rag.yaml")
        self._last_sync_file = PROJECT_ROOT / ".notion_last_sync"

        self.client = None
        self.sync_manager = None
        self.chroma_client = None
        self.collection = None

    def _load_config(self, filename: str) -> dict:
        """Load configuration from YAML"""
        config_path = self.config_dir / filename
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Expand environment variables
        return self._expand_env_vars(config)

    def _expand_env_vars(self, obj):
        """Recursively expand ${VAR} patterns in config"""
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                # Handle default values: ${VAR:default}
                if ":" in var_name:
                    var_name, default = var_name.split(":", 1)
                    return os.environ.get(var_name, default)
                return os.environ.get(var_name, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        return obj

    async def initialize(self):
        """Initialize all components"""
        from internal_ops_service.integrations.notion import (
            NotionClient,
            NotionClientConfig,
            NotionSyncManager,
            SyncConfig
        )

        # Check for API key
        api_key = self.notion_config.get("notion", {}).get("api_key")
        if not api_key or api_key.startswith("${"):
            api_key = os.environ.get("NOTION_API_KEY")

        if not api_key:
            raise ValueError(
                "NOTION_API_KEY environment variable is required. "
                "Get your API key from https://www.notion.so/my-integrations"
            )

        # Initialize Notion client
        notion_config = NotionClientConfig(api_key=api_key)
        self.client = NotionClient(notion_config)

        # Initialize sync manager
        sync_settings = self.notion_config.get("notion", {}).get("sync", {})
        sync_config = SyncConfig(
            include_archived=sync_settings.get("include_archived", False),
            page_ids_whitelist=sync_settings.get("page_ids_whitelist") or None,
            page_ids_blacklist=sync_settings.get("page_ids_blacklist") or None,
            database_ids=sync_settings.get("database_ids") or None,
            include_child_pages=sync_settings.get("include_child_pages", True),
            max_depth=sync_settings.get("max_depth", 5),
        )
        self.sync_manager = NotionSyncManager(self.client, sync_config)

        # Initialize ChromaDB
        await self._init_chroma()

        logger.info("Indexer initialized successfully")

    async def _init_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )

        rag_settings = self.rag_config.get("rag", {}).get("vector_store", {})
        persist_dir = rag_settings.get("persist_dir", "./data/chroma")
        collection_name = rag_settings.get("collection_name", "internal_ops_notion")

        # Create persist directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"ChromaDB collection '{collection_name}' ready. "
                   f"Current count: {self.collection.count()}")

    async def _get_embeddings(self, texts: list) -> list:
        """Get embeddings for texts using OpenAI"""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        client = openai.OpenAI(api_key=api_key)

        embedding_config = self.rag_config.get("rag", {}).get("embedding", {})
        model = embedding_config.get("model", "text-embedding-3-small")

        # Batch embeddings (OpenAI allows up to 2048 texts per request)
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=model,
                input=batch
            )
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _chunk_text(self, text: str, metadata: dict) -> list:
        """Chunk text into smaller pieces"""
        chunk_config = self.rag_config.get("rag", {}).get("chunking", {})
        chunk_size = chunk_config.get("chunk_size", 500)
        chunk_overlap = chunk_config.get("chunk_overlap", 50)

        chunks = []

        if len(text) <= chunk_size:
            chunks.append({"text": text, "metadata": metadata})
        else:
            start = 0
            chunk_index = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index
                }
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

                start = end - chunk_overlap
                chunk_index += 1

        return chunks

    async def full_sync(self):
        """Run full workspace sync"""
        logger.info("Starting full Notion sync...")

        documents = []
        metadatas = []
        ids = []

        doc_count = 0
        async for doc in self.sync_manager.sync_all():
            # Chunk the document
            chunks = self._chunk_text(doc.content, doc.metadata)

            for i, chunk in enumerate(chunks):
                doc_id = f"{doc.id}_{i}"
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                ids.append(doc_id)
                doc_count += 1

            # Batch index every 50 chunks
            if len(documents) >= 50:
                await self._index_batch(ids, documents, metadatas)
                documents = []
                metadatas = []
                ids = []

        # Index remaining
        if documents:
            await self._index_batch(ids, documents, metadatas)

        self._save_last_sync()
        logger.info(f"Full sync completed. Indexed {doc_count} chunks.")

    async def incremental_sync(self):
        """Sync changes since last sync"""
        last_sync = self._load_last_sync()
        logger.info(f"Starting incremental sync since {last_sync}...")

        documents = []
        metadatas = []
        ids = []
        doc_count = 0

        async for doc in self.sync_manager.sync_incremental(since=last_sync):
            chunks = self._chunk_text(doc.content, doc.metadata)

            for i, chunk in enumerate(chunks):
                doc_id = f"{doc.id}_{i}"
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                ids.append(doc_id)
                doc_count += 1

            if len(documents) >= 50:
                await self._index_batch(ids, documents, metadatas)
                documents = []
                metadatas = []
                ids = []

        if documents:
            await self._index_batch(ids, documents, metadatas)

        self._save_last_sync()
        logger.info(f"Incremental sync completed. Updated {doc_count} chunks.")

    async def sync_pages(self, page_ids: list):
        """Sync specific pages"""
        logger.info(f"Syncing {len(page_ids)} specific pages...")

        documents = []
        metadatas = []
        ids = []
        doc_count = 0

        async for doc in self.sync_manager.sync_pages(page_ids):
            chunks = self._chunk_text(doc.content, doc.metadata)

            for i, chunk in enumerate(chunks):
                doc_id = f"{doc.id}_{i}"
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                ids.append(doc_id)
                doc_count += 1

        if documents:
            await self._index_batch(ids, documents, metadatas)

        logger.info(f"Synced {doc_count} chunks from {len(page_ids)} pages.")

    async def _index_batch(self, ids: list, documents: list, metadatas: list):
        """Index a batch of documents into ChromaDB"""
        if not documents:
            return

        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = await self._get_embeddings(documents)

        # Upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Indexed {len(documents)} chunks. "
                   f"Total in collection: {self.collection.count()}")

    def _save_last_sync(self):
        """Save current timestamp as last sync time"""
        self._last_sync_file.write_text(datetime.utcnow().isoformat())

    def _load_last_sync(self) -> datetime:
        """Load last sync timestamp"""
        if self._last_sync_file.exists():
            ts = self._last_sync_file.read_text().strip()
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        # Default to 30 days ago
        return datetime.utcnow() - timedelta(days=30)

    async def show_status(self):
        """Show sync status"""
        last_sync = self._load_last_sync() if self._last_sync_file.exists() else None

        print("\n" + "=" * 50)
        print("Notion Indexer Status")
        print("=" * 50)
        print(f"Last sync: {last_sync or 'Never'}")
        print(f"Config directory: {self.config_dir}")

        if self.collection:
            print(f"ChromaDB collection count: {self.collection.count()}")

        print("\nNotion Config:")
        sync_config = self.notion_config.get("notion", {}).get("sync", {})
        print(f"  - Include archived: {sync_config.get('include_archived', False)}")
        print(f"  - Max depth: {sync_config.get('max_depth', 5)}")
        print("=" * 50 + "\n")

    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Notion Indexing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python index_notion.py full              # Full workspace sync
  python index_notion.py incremental       # Sync changes since last run
  python index_notion.py pages abc123 def456  # Sync specific pages
  python index_notion.py status            # Show sync status

Environment Variables:
  NOTION_API_KEY      Notion integration API key (required)
  OPENAI_API_KEY      OpenAI API key for embeddings (required)
  CHROMA_PERSIST_DIR  ChromaDB storage directory (optional)
        """
    )
    parser.add_argument(
        "command",
        choices=["full", "incremental", "pages", "status"],
        help="Sync command to run"
    )
    parser.add_argument(
        "page_ids",
        nargs="*",
        help="Page IDs for 'pages' command"
    )
    parser.add_argument(
        "--config-dir",
        help="Path to config directory"
    )

    args = parser.parse_args()

    indexer = NotionIndexer(config_dir=args.config_dir)

    try:
        await indexer.initialize()

        if args.command == "full":
            await indexer.full_sync()
        elif args.command == "incremental":
            await indexer.incremental_sync()
        elif args.command == "pages":
            if not args.page_ids:
                print("Error: page_ids required for 'pages' command")
                sys.exit(1)
            await indexer.sync_pages(args.page_ids)
        elif args.command == "status":
            await indexer.show_status()

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        await indexer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
