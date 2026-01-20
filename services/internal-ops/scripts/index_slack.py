#!/usr/bin/env python3
"""
Slack Indexing CLI

Usage:
    python index_slack.py full                  # Full workspace sync
    python index_slack.py incremental           # Since last sync
    python index_slack.py channels CH_ID...     # Specific channels
    python index_slack.py status                # Show sync status
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project paths
SERVICE_ROOT = Path(__file__).parent.parent  # services/internal-ops
PROJECT_ROOT = SERVICE_ROOT.parent.parent     # agentic-ai-platform
sys.path.insert(0, str(SERVICE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "core"))

import yaml

# Import core chunker and embedder
from agentic_core.rag import (
    RecursiveChunker,
    ChunkerConfig,
    TextChunk,
    EmbeddingModel,
    create_embedder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SlackIndexer:
    """CLI for indexing Slack messages into ChromaDB"""

    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else (
            SERVICE_ROOT / "config"
        )
        self.slack_config = self._load_config("slack.yaml")
        self.rag_config = self._load_config("rag.yaml")
        self._last_sync_file = SERVICE_ROOT / ".slack_last_sync"

        self.client = None
        self.sync_manager = None
        self.chroma_client = None
        self.collection = None

        # Initialize chunker from core package
        chunking_config = self.rag_config.get("rag", {}).get("chunking", {})
        self.chunker = RecursiveChunker(ChunkerConfig(
            chunk_size=chunking_config.get("chunk_size", 500),
            chunk_overlap=chunking_config.get("chunk_overlap", 50),
            min_chunk_size=chunking_config.get("min_chunk_size", 100),
            max_chunk_size=chunking_config.get("max_chunk_size", 1000),
            respect_sentence_boundary=True
        ))
        logger.info(f"Initialized RecursiveChunker with chunk_size={self.chunker.config.chunk_size}")

        # Initialize embedder from core package (Google Gemini by default)
        embedding_config = self.rag_config.get("rag", {}).get("embedding", {})
        embedding_model = embedding_config.get("model", EmbeddingModel.GEMINI_EMBEDDING.value)
        self.embedder = create_embedder(model=embedding_model)
        logger.info(f"Initialized Embedder: {embedding_model}")

    def _load_config(self, filename: str) -> dict:
        """Load configuration from YAML"""
        config_path = self.config_dir / filename
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        return self._expand_env_vars(config)

    def _expand_env_vars(self, obj):
        """Recursively expand ${VAR} patterns in config"""
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
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
        from internal_ops_service.integrations.slack import (
            SlackClient,
            SlackClientConfig,
            SlackSyncManager,
            SlackSyncConfig
        )

        slack_settings = self.slack_config.get("slack", {})

        # Check for bot token
        bot_token = slack_settings.get("bot_token")
        if not bot_token or bot_token.startswith("${"):
            bot_token = os.environ.get("SLACK_BOT_TOKEN")

        if not bot_token:
            raise ValueError(
                "SLACK_BOT_TOKEN environment variable is required. "
                "Create a Slack app at https://api.slack.com/apps"
            )

        # User token (optional, for search)
        user_token = slack_settings.get("user_token")
        if user_token and user_token.startswith("${"):
            user_token = os.environ.get("SLACK_USER_TOKEN")

        # Workspace URL
        workspace_url = slack_settings.get("workspace_url", "")
        if workspace_url.startswith("${"):
            workspace_url = os.environ.get("SLACK_WORKSPACE_URL", "")

        # Initialize Slack client
        client_config = SlackClientConfig(
            bot_token=bot_token,
            user_token=user_token if user_token else None
        )
        self.client = SlackClient(client_config)

        # Test authentication
        try:
            auth_info = await self.client.test_auth()
            logger.info(f"Authenticated as: {auth_info.get('user', 'unknown')} "
                       f"in workspace: {auth_info.get('team', 'unknown')}")
        except Exception as e:
            raise ValueError(f"Slack authentication failed: {e}")

        # Initialize sync manager
        sync_settings = slack_settings.get("sync", {})
        sync_config = SlackSyncConfig(
            include_private=sync_settings.get("include_private", True),
            include_threads=sync_settings.get("include_threads", True),
            include_archived=sync_settings.get("include_archived", False),
            max_messages_per_channel=sync_settings.get("max_messages_per_channel", 10000),
            lookback_days=sync_settings.get("lookback_days", 90),
            batch_size=sync_settings.get("batch_size", 100),
            channel_whitelist=sync_settings.get("channel_whitelist", []),
            channel_blacklist=sync_settings.get("channel_blacklist", []),
            exclude_bot_messages=sync_settings.get("exclude_bot_messages", False)
        )
        self.sync_manager = SlackSyncManager(
            self.client,
            sync_config,
            workspace_url=workspace_url
        )

        # Initialize ChromaDB
        await self._init_chroma()

        logger.info("Slack indexer initialized successfully")

    async def _init_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )

        # Get settings from slack config or rag config
        slack_vector = self.slack_config.get("slack", {}).get("vector_store", {})
        rag_vector = self.rag_config.get("rag", {}).get("vector_store", {})

        persist_dir = (
            slack_vector.get("persist_dir") or
            rag_vector.get("persist_dir") or
            os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
        )
        collection_name = slack_vector.get("collection_name", "internal_ops_slack")

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
        """Get embeddings for texts using core embedder (Gemini by default)"""
        if not texts:
            return []

        # Use batch embedding from core embedder
        results = await self.embedder.embed_batch(texts)
        return [result.embedding for result in results]

    async def full_sync(self):
        """Run full workspace sync"""
        logger.info("Starting full Slack sync...")

        documents = []
        metadatas = []
        ids = []
        doc_count = 0

        async for doc in self.sync_manager.sync_all():
            doc_dict = doc.to_dict()

            documents.append(doc_dict["content"])
            metadatas.append(doc_dict["metadata"])
            ids.append(doc_dict["id"])
            doc_count += 1

            # Batch index every 50 documents
            if len(documents) >= 50:
                await self._index_batch(ids, documents, metadatas)
                documents = []
                metadatas = []
                ids = []

        # Index remaining
        if documents:
            await self._index_batch(ids, documents, metadatas)

        self._save_last_sync()
        logger.info(f"Full sync completed. Indexed {doc_count} messages.")

    async def incremental_sync(self):
        """Sync changes since last sync"""
        last_sync = self._load_last_sync()
        logger.info(f"Starting incremental sync since {last_sync}...")

        documents = []
        metadatas = []
        ids = []
        doc_count = 0

        async for doc in self.sync_manager.sync_incremental(since=last_sync):
            doc_dict = doc.to_dict()

            documents.append(doc_dict["content"])
            metadatas.append(doc_dict["metadata"])
            ids.append(doc_dict["id"])
            doc_count += 1

            if len(documents) >= 50:
                await self._index_batch(ids, documents, metadatas)
                documents = []
                metadatas = []
                ids = []

        if documents:
            await self._index_batch(ids, documents, metadatas)

        self._save_last_sync()
        logger.info(f"Incremental sync completed. Updated {doc_count} messages.")

    async def sync_channels(self, channel_ids: list):
        """Sync specific channels"""
        logger.info(f"Syncing {len(channel_ids)} specific channels...")

        documents = []
        metadatas = []
        ids = []
        doc_count = 0

        async for doc in self.sync_manager.sync_channels(channel_ids):
            doc_dict = doc.to_dict()

            documents.append(doc_dict["content"])
            metadatas.append(doc_dict["metadata"])
            ids.append(doc_dict["id"])
            doc_count += 1

            if len(documents) >= 50:
                await self._index_batch(ids, documents, metadatas)
                documents = []
                metadatas = []
                ids = []

        if documents:
            await self._index_batch(ids, documents, metadatas)

        logger.info(f"Synced {doc_count} messages from {len(channel_ids)} channels.")

    def _chunk_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """
        Chunk documents using core RecursiveChunker.

        Long messages are split into smaller chunks while preserving metadata.
        Short messages are kept as-is.
        """
        chunked_ids = []
        chunked_docs = []
        chunked_metas = []

        for doc_id, content, meta in zip(ids, documents, metadatas):
            # Skip short content (no need to chunk)
            if len(content) <= self.chunker.config.max_chunk_size:
                chunked_ids.append(doc_id)
                chunked_docs.append(content)
                chunked_metas.append(meta)
                continue

            # Chunk long content
            chunks: List[TextChunk] = self.chunker.chunk(content)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_meta = {
                    **meta,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "parent_id": doc_id,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                }
                chunked_ids.append(chunk_id)
                chunked_docs.append(chunk.content)
                chunked_metas.append(chunk_meta)

        if len(chunked_docs) > len(documents):
            logger.info(
                f"Chunking: {len(documents)} messages -> {len(chunked_docs)} chunks"
            )

        return chunked_ids, chunked_docs, chunked_metas

    async def _index_batch(self, ids: list, documents: list, metadatas: list):
        """Index a batch of documents into ChromaDB with chunking"""
        if not documents:
            return

        # Apply chunking using core RecursiveChunker
        chunked_ids, chunked_docs, chunked_metas = self._chunk_documents(
            ids, documents, metadatas
        )

        logger.info(f"Generating embeddings for {len(chunked_docs)} chunks...")
        embeddings = await self._get_embeddings(chunked_docs)

        # Upsert to ChromaDB
        self.collection.upsert(
            ids=chunked_ids,
            documents=chunked_docs,
            embeddings=embeddings,
            metadatas=chunked_metas
        )

        logger.info(f"Indexed {len(chunked_docs)} chunks from {len(documents)} messages. "
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
        # Default to lookback_days from config
        lookback = self.slack_config.get("slack", {}).get("sync", {}).get("lookback_days", 30)
        return datetime.utcnow() - timedelta(days=lookback)

    async def show_status(self):
        """Show sync status"""
        last_sync = self._load_last_sync() if self._last_sync_file.exists() else None

        print("\n" + "=" * 50)
        print("Slack Indexer Status")
        print("=" * 50)
        print(f"Last sync: {last_sync or 'Never'}")
        print(f"Config directory: {self.config_dir}")

        if self.collection:
            print(f"ChromaDB collection count: {self.collection.count()}")

        print("\nSlack Config:")
        sync_config = self.slack_config.get("slack", {}).get("sync", {})
        print(f"  - Include private channels: {sync_config.get('include_private', True)}")
        print(f"  - Include threads: {sync_config.get('include_threads', True)}")
        print(f"  - Lookback days: {sync_config.get('lookback_days', 90)}")
        print(f"  - Max messages/channel: {sync_config.get('max_messages_per_channel', 10000)}")

        # Try to get workspace info
        if self.client:
            try:
                team_info = await self.client.get_team_info()
                print(f"\nWorkspace: {team_info.get('name', 'Unknown')}")
                print(f"Domain: {team_info.get('domain', 'Unknown')}.slack.com")
            except Exception:
                pass

        print("=" * 50 + "\n")

    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Slack Indexing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python index_slack.py full                  # Full workspace sync
  python index_slack.py incremental           # Sync changes since last run
  python index_slack.py channels C123 C456    # Sync specific channels
  python index_slack.py status                # Show sync status

Environment Variables:
  SLACK_BOT_TOKEN       Slack bot token (required, xoxb-...)
  SLACK_USER_TOKEN      Slack user token for search (optional, xoxp-...)
  SLACK_WORKSPACE_URL   Workspace URL for permalinks (optional)
  GEMINI_API_KEY        Google Gemini API key for embeddings (required)
  GOOGLE_API_KEY        Alternative Google API key (used if GEMINI_API_KEY not set)
  CHROMA_PERSIST_DIR    ChromaDB storage directory (optional)
        """
    )
    parser.add_argument(
        "command",
        choices=["full", "incremental", "channels", "status"],
        help="Sync command to run"
    )
    parser.add_argument(
        "channel_ids",
        nargs="*",
        help="Channel IDs for 'channels' command"
    )
    parser.add_argument(
        "--config-dir",
        help="Path to config directory"
    )

    args = parser.parse_args()

    indexer = SlackIndexer(config_dir=args.config_dir)

    try:
        await indexer.initialize()

        if args.command == "full":
            await indexer.full_sync()
        elif args.command == "incremental":
            await indexer.incremental_sync()
        elif args.command == "channels":
            if not args.channel_ids:
                print("Error: channel_ids required for 'channels' command")
                sys.exit(1)
            await indexer.sync_channels(args.channel_ids)
        elif args.command == "status":
            await indexer.show_status()

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        await indexer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
