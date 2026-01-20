# Slack Integration Module
from .client import SlackClient, SlackClientConfig
from .models import (
    SlackUser,
    SlackChannel,
    SlackMessage,
    SlackThread,
    SyncedSlackDocument,
    MessageType
)
from .sync import SlackSyncManager, SlackSyncConfig

__all__ = [
    # Client
    "SlackClient",
    "SlackClientConfig",
    # Models
    "SlackUser",
    "SlackChannel",
    "SlackMessage",
    "SlackThread",
    "SyncedSlackDocument",
    "MessageType",
    # Sync
    "SlackSyncManager",
    "SlackSyncConfig",
]
