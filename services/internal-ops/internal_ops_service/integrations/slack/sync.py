"""
Slack Sync Manager - Batch synchronization of Slack messages for RAG indexing
"""

from typing import List, Dict, Any, Optional, AsyncIterator, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import re

from .client import SlackClient
from .models import SlackMessage, SlackChannel, SlackUser, SyncedSlackDocument

logger = logging.getLogger(__name__)


@dataclass
class SlackSyncConfig:
    """Slack sync configuration"""
    include_private: bool = True
    include_threads: bool = True
    include_archived: bool = False
    max_messages_per_channel: int = 10000
    lookback_days: int = 90
    batch_size: int = 100
    channel_whitelist: List[str] = field(default_factory=list)  # Empty = all
    channel_blacklist: List[str] = field(default_factory=list)
    exclude_bot_messages: bool = False


class SlackSyncManager:
    """
    Manages batch synchronization of Slack messages for RAG indexing.

    Features:
    - Full workspace sync
    - Incremental sync (since timestamp)
    - Channel-specific sync
    - Thread sync
    - User name resolution with caching
    - Deduplication
    """

    def __init__(
        self,
        client: SlackClient,
        config: Optional[SlackSyncConfig] = None,
        workspace_url: str = ""
    ):
        self.client = client
        self.config = config or SlackSyncConfig()
        self.workspace_url = workspace_url.rstrip("/")

        # Caches
        self._user_cache: Dict[str, SlackUser] = {}
        self._channel_cache: Dict[str, SlackChannel] = {}
        self._seen_messages: Set[str] = set()

    async def sync_all(self) -> AsyncIterator[SyncedSlackDocument]:
        """
        Sync all accessible channels.

        Yields:
            SyncedSlackDocument for each message
        """
        logger.info("Starting full Slack workspace sync")
        self._seen_messages.clear()

        oldest = datetime.utcnow() - timedelta(days=self.config.lookback_days)
        doc_count = 0
        channel_count = 0

        async for channel_data in self.client.list_channels(
            exclude_archived=not self.config.include_archived
        ):
            channel = SlackChannel.from_api_response(channel_data)

            if not self._should_sync_channel(channel):
                continue

            channel_count += 1
            self._channel_cache[channel.id] = channel
            logger.info(f"Syncing channel: #{channel.name} ({channel.id})")

            async for doc in self._sync_channel(channel, oldest):
                doc_count += 1
                yield doc

        logger.info(
            f"Slack sync completed. "
            f"Channels: {channel_count}, Documents: {doc_count}"
        )

    async def sync_incremental(
        self,
        since: datetime
    ) -> AsyncIterator[SyncedSlackDocument]:
        """
        Sync messages since a given timestamp.

        Args:
            since: Only sync messages after this time

        Yields:
            SyncedSlackDocument for each new message
        """
        logger.info(f"Starting incremental Slack sync since {since}")
        self._seen_messages.clear()

        doc_count = 0
        channel_count = 0

        async for channel_data in self.client.list_channels(
            exclude_archived=True
        ):
            channel = SlackChannel.from_api_response(channel_data)

            if not self._should_sync_channel(channel):
                continue

            channel_count += 1
            self._channel_cache[channel.id] = channel

            async for doc in self._sync_channel(channel, since):
                doc_count += 1
                yield doc

        logger.info(
            f"Incremental sync completed. "
            f"Channels: {channel_count}, Documents: {doc_count}"
        )

    async def sync_channels(
        self,
        channel_ids: List[str],
        oldest: Optional[datetime] = None
    ) -> AsyncIterator[SyncedSlackDocument]:
        """
        Sync specific channels.

        Args:
            channel_ids: List of channel IDs to sync
            oldest: Only sync messages after this time

        Yields:
            SyncedSlackDocument for each message
        """
        logger.info(f"Syncing {len(channel_ids)} specific channels")
        self._seen_messages.clear()

        if oldest is None:
            oldest = datetime.utcnow() - timedelta(days=self.config.lookback_days)

        doc_count = 0

        for channel_id in channel_ids:
            try:
                channel_data = await self.client.get_channel_info(channel_id)
                channel = SlackChannel.from_api_response(channel_data)
                self._channel_cache[channel.id] = channel

                logger.info(f"Syncing channel: #{channel.name}")

                async for doc in self._sync_channel(channel, oldest):
                    doc_count += 1
                    yield doc

            except Exception as e:
                logger.error(f"Failed to sync channel {channel_id}: {e}")

        logger.info(f"Channel sync completed. Documents: {doc_count}")

    def _should_sync_channel(self, channel: SlackChannel) -> bool:
        """Check if channel should be synced based on config"""
        # Check archived
        if channel.is_archived and not self.config.include_archived:
            return False

        # Check private
        if channel.is_private and not self.config.include_private:
            return False

        # Check whitelist
        if self.config.channel_whitelist:
            if channel.id not in self.config.channel_whitelist and \
               channel.name not in self.config.channel_whitelist:
                return False

        # Check blacklist
        if channel.id in self.config.channel_blacklist or \
           channel.name in self.config.channel_blacklist:
            return False

        return True

    async def _sync_channel(
        self,
        channel: SlackChannel,
        oldest: datetime
    ) -> AsyncIterator[SyncedSlackDocument]:
        """Sync a single channel"""
        message_count = 0

        async for msg_data in self.client.get_channel_history(
            channel.id,
            oldest=oldest,
            limit=self.config.max_messages_per_channel
        ):
            message = SlackMessage.from_api_response(msg_data, channel.id)

            # Skip if already seen
            msg_id = f"{channel.id}_{message.ts}"
            if msg_id in self._seen_messages:
                continue
            self._seen_messages.add(msg_id)

            # Skip bot messages if configured
            if self.config.exclude_bot_messages and message.subtype == "bot_message":
                continue

            # Skip messages without meaningful text
            if not self._has_meaningful_content(message.text):
                continue

            # Process main message
            doc = await self._process_message(message, channel)
            yield doc
            message_count += 1

            # Sync thread replies if enabled
            if self.config.include_threads and message.is_thread_parent:
                try:
                    replies = await self.client.get_thread_replies(
                        channel.id,
                        message.ts
                    )

                    # Skip first message (parent already processed)
                    for reply_data in replies[1:]:
                        reply = SlackMessage.from_api_response(reply_data, channel.id)

                        reply_id = f"{channel.id}_{reply.ts}"
                        if reply_id in self._seen_messages:
                            continue
                        self._seen_messages.add(reply_id)

                        if self._has_meaningful_content(reply.text):
                            reply_doc = await self._process_message(
                                reply, channel, is_thread=True
                            )
                            yield reply_doc
                            message_count += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to sync thread {message.ts} in #{channel.name}: {e}"
                    )

        logger.debug(f"Synced {message_count} messages from #{channel.name}")

    def _has_meaningful_content(self, text: str) -> bool:
        """Check if message has meaningful content for indexing"""
        if not text:
            return False

        # Remove mentions and links for check
        cleaned = re.sub(r'<[@#][^>]+>', '', text)
        cleaned = re.sub(r'<https?://[^>]+>', '', cleaned)
        cleaned = cleaned.strip()

        # Minimum content length
        return len(cleaned) >= 3

    async def _process_message(
        self,
        message: SlackMessage,
        channel: SlackChannel,
        is_thread: bool = False
    ) -> SyncedSlackDocument:
        """Process a message into a SyncedSlackDocument"""
        # Resolve user name
        user_name = None
        if message.user_id:
            user = await self._get_user(message.user_id)
            if user:
                user_name = user.display_name or user.real_name or user.name

        # Clean message text
        content = self._clean_message_text(message.text)

        # Build permalink
        url = self._build_permalink(channel.id, message.ts)

        # Build metadata
        metadata = {
            "has_files": len(message.files) > 0,
            "has_attachments": len(message.attachments) > 0,
            "reaction_count": sum(r.get("count", 0) for r in message.reactions),
            "subtype": message.subtype,
        }

        return SyncedSlackDocument(
            id=f"{channel.id}_{message.ts}",
            channel_id=channel.id,
            channel_name=channel.name,
            message_ts=message.ts,
            user_id=message.user_id,
            user_name=user_name,
            content=content,
            timestamp=message.timestamp,
            thread_ts=message.thread_ts,
            is_thread=is_thread,
            url=url,
            metadata=metadata
        )

    def _clean_message_text(self, text: str) -> str:
        """Clean Slack message text for indexing"""
        if not text:
            return ""

        # Replace user mentions with placeholder
        text = re.sub(r'<@([A-Z0-9]+)>', r'@user', text)

        # Replace channel mentions with placeholder
        text = re.sub(r'<#([A-Z0-9]+)\|([^>]+)>', r'#\2', text)
        text = re.sub(r'<#([A-Z0-9]+)>', r'#channel', text)

        # Extract link text or URL
        text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
        text = re.sub(r'<(https?://[^>]+)>', r'\1', text)

        # Remove special commands
        text = re.sub(r'<!([^>]+)>', r'@\1', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _build_permalink(self, channel_id: str, ts: str) -> str:
        """Build Slack message permalink"""
        if not self.workspace_url:
            return ""

        # Convert timestamp to permalink format (remove dot)
        ts_formatted = ts.replace(".", "")
        return f"{self.workspace_url}/archives/{channel_id}/p{ts_formatted}"

    async def _get_user(self, user_id: str) -> Optional[SlackUser]:
        """Get user with caching"""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            user_data = await self.client.get_user_info(user_id)
            user = SlackUser.from_api_response(user_data)
            self._user_cache[user_id] = user
            return user
        except Exception as e:
            logger.debug(f"Failed to get user {user_id}: {e}")
            return None

    def clear_caches(self):
        """Clear all caches"""
        self._user_cache.clear()
        self._channel_cache.clear()
        self._seen_messages.clear()
