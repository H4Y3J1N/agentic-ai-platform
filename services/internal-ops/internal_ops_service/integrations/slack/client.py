"""
Slack API Client

Uses slack-sdk for async operations with built-in rate limiting and pagination.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class SlackClientConfig:
    """Slack client configuration"""
    bot_token: str
    user_token: Optional[str] = None  # For search.messages (requires user token)
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class SlackClient:
    """
    Async Slack API wrapper using slack-sdk.

    Provides methods for:
    - Listing channels
    - Getting channel history
    - Getting thread replies
    - Getting user info
    - Searching messages (requires user token)
    """

    def __init__(self, config: SlackClientConfig):
        self.config = config
        self._client = None
        self._user_client = None

    async def _get_client(self):
        """Get or create Slack AsyncWebClient"""
        if self._client is None:
            try:
                from slack_sdk.web.async_client import AsyncWebClient
            except ImportError:
                raise ImportError(
                    "slack-sdk is required. Install with: pip install slack-sdk"
                )

            self._client = AsyncWebClient(
                token=self.config.bot_token,
                timeout=self.config.timeout
            )
        return self._client

    async def _get_user_client(self):
        """Get or create user-token client for search operations"""
        if self._user_client is None:
            if not self.config.user_token:
                return None

            try:
                from slack_sdk.web.async_client import AsyncWebClient
            except ImportError:
                raise ImportError(
                    "slack-sdk is required. Install with: pip install slack-sdk"
                )

            self._user_client = AsyncWebClient(
                token=self.config.user_token,
                timeout=self.config.timeout
            )
        return self._user_client

    async def _retry_request(self, coro, operation_name: str):
        """Execute request with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                return await coro
            except Exception as e:
                error_str = str(e)

                # Handle rate limiting
                if "rate_limited" in error_str.lower() or "ratelimited" in error_str.lower():
                    retry_after = 60  # Default retry after
                    logger.warning(
                        f"Rate limited on {operation_name}. "
                        f"Waiting {retry_after}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(retry_after)
                    continue

                # Other errors
                if attempt == self.config.max_retries - 1:
                    logger.error(f"{operation_name} failed after {self.config.max_retries} attempts: {e}")
                    raise

                logger.warning(
                    f"{operation_name} error (attempt {attempt + 1}): {e}"
                )
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise Exception(f"Max retries exceeded for {operation_name}")

    async def list_channels(
        self,
        types: str = "public_channel,private_channel",
        exclude_archived: bool = True,
        limit: int = 200
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        List all accessible channels with pagination.

        Args:
            types: Channel types to include (comma-separated)
            exclude_archived: Whether to exclude archived channels
            limit: Max channels per page

        Yields:
            Channel data dictionaries
        """
        client = await self._get_client()
        cursor = None
        total_count = 0

        while True:
            response = await self._retry_request(
                client.conversations_list(
                    types=types,
                    exclude_archived=exclude_archived,
                    limit=min(limit, 200),
                    cursor=cursor
                ),
                "conversations_list"
            )

            for channel in response.get("channels", []):
                yield channel
                total_count += 1

            # Check for more pages
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        logger.info(f"Listed {total_count} channels")

    async def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """
        Get channel information.

        Args:
            channel_id: Slack channel ID

        Returns:
            Channel info dictionary
        """
        client = await self._get_client()

        response = await self._retry_request(
            client.conversations_info(channel=channel_id),
            f"conversations_info({channel_id})"
        )

        return response.get("channel", {})

    async def get_channel_history(
        self,
        channel_id: str,
        oldest: Optional[datetime] = None,
        latest: Optional[datetime] = None,
        limit: int = 1000
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Get channel message history with pagination.

        Args:
            channel_id: Slack channel ID
            oldest: Only fetch messages after this time
            latest: Only fetch messages before this time
            limit: Max messages to return

        Yields:
            Message data dictionaries
        """
        client = await self._get_client()
        cursor = None
        count = 0

        while count < limit:
            kwargs: Dict[str, Any] = {
                "channel": channel_id,
                "limit": min(200, limit - count),
            }
            if cursor:
                kwargs["cursor"] = cursor
            if oldest:
                kwargs["oldest"] = str(oldest.timestamp())
            if latest:
                kwargs["latest"] = str(latest.timestamp())

            response = await self._retry_request(
                client.conversations_history(**kwargs),
                f"conversations_history({channel_id})"
            )

            for message in response.get("messages", []):
                yield message
                count += 1
                if count >= limit:
                    break

            # Check for more pages
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        logger.debug(f"Fetched {count} messages from channel {channel_id}")

    async def get_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get all replies in a thread.

        Args:
            channel_id: Slack channel ID
            thread_ts: Thread parent timestamp
            limit: Max replies to return

        Returns:
            List of message dictionaries (including parent)
        """
        client = await self._get_client()
        messages = []
        cursor = None

        while len(messages) < limit:
            kwargs: Dict[str, Any] = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": min(200, limit - len(messages)),
            }
            if cursor:
                kwargs["cursor"] = cursor

            response = await self._retry_request(
                client.conversations_replies(**kwargs),
                f"conversations_replies({channel_id}, {thread_ts})"
            )

            for message in response.get("messages", []):
                messages.append(message)
                if len(messages) >= limit:
                    break

            # Check for more pages
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return messages

    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get user information.

        Args:
            user_id: Slack user ID

        Returns:
            User info dictionary
        """
        client = await self._get_client()

        response = await self._retry_request(
            client.users_info(user=user_id),
            f"users_info({user_id})"
        )

        return response.get("user", {})

    async def list_users(self, limit: int = 1000) -> AsyncIterator[Dict[str, Any]]:
        """
        List all users in workspace.

        Args:
            limit: Max users to return

        Yields:
            User data dictionaries
        """
        client = await self._get_client()
        cursor = None
        count = 0

        while count < limit:
            kwargs: Dict[str, Any] = {"limit": min(200, limit - count)}
            if cursor:
                kwargs["cursor"] = cursor

            response = await self._retry_request(
                client.users_list(**kwargs),
                "users_list"
            )

            for user in response.get("members", []):
                yield user
                count += 1
                if count >= limit:
                    break

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

    async def search_messages(
        self,
        query: str,
        count: int = 20,
        sort: str = "timestamp",
        sort_dir: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        Search messages across workspace.

        NOTE: Requires user token with search:read scope.
        Bot tokens cannot use search.messages.

        Args:
            query: Search query
            count: Max results
            sort: Sort by "timestamp" or "score"
            sort_dir: Sort direction "asc" or "desc"

        Returns:
            List of matching message dictionaries
        """
        user_client = await self._get_user_client()
        if not user_client:
            logger.warning("search_messages requires user_token (search:read scope)")
            return []

        response = await self._retry_request(
            user_client.search_messages(
                query=query,
                count=count,
                sort=sort,
                sort_dir=sort_dir
            ),
            f"search_messages({query})"
        )

        return response.get("messages", {}).get("matches", [])

    async def get_team_info(self) -> Dict[str, Any]:
        """
        Get workspace/team information.

        Returns:
            Team info dictionary
        """
        client = await self._get_client()

        response = await self._retry_request(
            client.team_info(),
            "team_info"
        )

        return response.get("team", {})

    async def test_auth(self) -> Dict[str, Any]:
        """
        Test authentication and get bot info.

        Returns:
            Auth test response
        """
        client = await self._get_client()

        response = await self._retry_request(
            client.auth_test(),
            "auth_test"
        )

        return dict(response)

    async def close(self):
        """Close the clients"""
        self._client = None
        self._user_client = None
