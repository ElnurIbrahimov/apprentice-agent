"""Slack Tool — read channels, search messages, and post via Slack SDK.

Uses slack_sdk WebClient (synchronous) with a Bot Token.
Gives AURA awareness of your Slack workspace — team conversations,
mentions, channel activity — without manual checking.

Setup:
    1. Create a Slack App at api.slack.com/apps
    2. Add OAuth Scopes: channels:read, channels:history, chat:write, search:read, users:read
    3. Install to workspace and copy Bot User OAuth Token
    4. Set SLACK_BOT_TOKEN in .env

Config:
    SLACK_BOT_TOKEN — Bot token starting with 'xoxb-'
"""

import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


def _ts_to_dt(ts: str) -> str:
    """Convert Slack timestamp to readable datetime."""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


class SlackTool:
    """Read Slack channels and messages, search conversations, post messages."""

    name = "slack"
    description = "Read Slack channels/messages, search conversations, post messages — team awareness from AURA"

    def __init__(self):
        self._token = os.getenv("SLACK_BOT_TOKEN", "")
        self._client = None
        if SLACK_SDK_AVAILABLE and self._token:
            self._client = WebClient(token=self._token)
            logger.info("[Slack] Client initialized")
        elif not self._token:
            logger.info("[Slack] No SLACK_BOT_TOKEN set — configure in .env")

    def _check(self) -> Optional[Dict]:
        if not SLACK_SDK_AVAILABLE:
            return {"success": False, "error": "slack_sdk not installed. Run: pip install slack_sdk"}
        if not self._token:
            return {"success": False, "error": "SLACK_BOT_TOKEN not set in .env — create a Slack app at api.slack.com/apps"}
        if not self._client:
            return {"success": False, "error": "Slack client not initialized"}
        return None

    def _call(self, method_fn, **kwargs) -> tuple:
        try:
            resp = method_fn(**kwargs)
            return resp, None
        except SlackApiError as e:
            return None, f"Slack API error: {e.response['error']}"
        except Exception as e:
            return None, str(e)

    # ------------------------------------------------------------------ #

    def auth_test(self) -> Dict:
        """Test authentication and get bot identity."""
        err = self._check()
        if err:
            return err
        resp, error = self._call(self._client.auth_test)
        if error:
            return {"success": False, "error": error}
        return {
            "success": True,
            "bot_name": resp["bot_id"],
            "user": resp["user"],
            "team": resp["team"],
            "url": resp["url"],
        }

    def list_channels(self, types: str = "public_channel", limit: int = 100) -> Dict:
        """List channels in the workspace.

        Args:
            types: Channel types: 'public_channel', 'private_channel', 'mpim', 'im'
            limit: Max channels to return
        """
        err = self._check()
        if err:
            return err
        resp, error = self._call(
            self._client.conversations_list,
            types=types,
            limit=limit,
            exclude_archived=True,
        )
        if error:
            return {"success": False, "error": error}
        channels = resp.get("channels", [])
        return {
            "success": True,
            "count": len(channels),
            "channels": [
                {
                    "id": c["id"],
                    "name": c.get("name", ""),
                    "topic": c.get("topic", {}).get("value", ""),
                    "purpose": c.get("purpose", {}).get("value", ""),
                    "num_members": c.get("num_members", 0),
                }
                for c in channels
            ],
        }

    def read_channel(self, channel: str, limit: int = 20) -> Dict:
        """Read recent messages from a channel.

        Args:
            channel: Channel ID (C...) or name (#general)
            limit: Number of messages to fetch
        """
        err = self._check()
        if err:
            return err

        # Resolve channel name to ID if needed
        channel_id = channel
        if channel.startswith("#"):
            channel_id = self._resolve_channel_name(channel[1:])
            if not channel_id:
                return {"success": False, "error": f"Channel '{channel}' not found"}

        resp, error = self._call(
            self._client.conversations_history,
            channel=channel_id,
            limit=limit,
        )
        if error:
            return {"success": False, "error": error}

        messages = resp.get("messages", [])
        enriched = []
        for m in messages:
            if m.get("subtype") in ("channel_join", "channel_leave", "bot_message"):
                continue
            enriched.append({
                "user": self._resolve_user(m.get("user", "")),
                "text": m.get("text", ""),
                "timestamp": _ts_to_dt(m.get("ts", "")),
                "thread_replies": m.get("reply_count", 0),
                "reactions": [f"{r['name']}×{r['count']}" for r in m.get("reactions", [])],
            })

        return {
            "success": True,
            "channel": channel,
            "count": len(enriched),
            "messages": enriched,
        }

    def _resolve_channel_name(self, name: str) -> Optional[str]:
        """Find a channel ID by name."""
        try:
            resp = self._client.conversations_list(types="public_channel,private_channel", limit=200)
            for c in resp.get("channels", []):
                if c.get("name") == name:
                    return c["id"]
        except Exception:
            pass
        return None

    def _resolve_user(self, user_id: str) -> str:
        """Get display name for a user ID."""
        if not user_id:
            return "unknown"
        try:
            resp = self._client.users_info(user=user_id)
            user = resp.get("user", {})
            return user.get("real_name") or user.get("name") or user_id
        except Exception:
            return user_id

    def send_message(self, channel: str, text: str) -> Dict:
        """Send a message to a channel.

        Args:
            channel: Channel ID or name (#general)
            text: Message text (supports Slack markdown)
        """
        err = self._check()
        if err:
            return err
        if channel.startswith("#"):
            channel_id = self._resolve_channel_name(channel[1:])
            if not channel_id:
                return {"success": False, "error": f"Channel '{channel}' not found"}
            channel = channel_id
        resp, error = self._call(self._client.chat_postMessage, channel=channel, text=text)
        if error:
            return {"success": False, "error": error}
        return {
            "success": True,
            "sent": text[:100],
            "channel": channel,
            "ts": resp.get("ts"),
        }

    def search_messages(self, query: str, count: int = 20) -> Dict:
        """Search messages across the entire workspace.

        Args:
            query: Search query (supports Slack search modifiers)
            count: Number of results
        """
        err = self._check()
        if err:
            return err
        resp, error = self._call(self._client.search_messages, query=query, count=count)
        if error:
            return {"success": False, "error": error}
        matches = resp.get("messages", {}).get("matches", [])
        return {
            "success": True,
            "query": query,
            "total": resp.get("messages", {}).get("total", 0),
            "count": len(matches),
            "results": [
                {
                    "user": m.get("username", ""),
                    "text": m.get("text", ""),
                    "channel": m.get("channel", {}).get("name", ""),
                    "timestamp": _ts_to_dt(m.get("ts", "")),
                    "permalink": m.get("permalink", ""),
                }
                for m in matches
            ],
        }

    def get_mentions(self, limit: int = 20) -> Dict:
        """Get recent messages that mention the bot."""
        return self.search_messages(f"<@{self._get_bot_id()}>", limit)

    def _get_bot_id(self) -> str:
        try:
            resp = self._client.auth_test()
            return resp.get("user_id", "")
        except Exception:
            return ""

    def get_unread_summary(self, channels: Optional[List[str]] = None, limit_per_channel: int = 5) -> Dict:
        """Summarize recent activity across key channels.

        Args:
            channels: List of channel IDs or names (fetches top channels if None)
            limit_per_channel: Messages to read per channel
        """
        err = self._check()
        if err:
            return err

        if not channels:
            all_ch = self.list_channels()
            channels = [c["id"] for c in all_ch.get("channels", [])[:5]]

        summaries = []
        for ch in channels:
            msgs = self.read_channel(ch, limit_per_channel)
            if msgs.get("success") and msgs.get("messages"):
                summaries.append({
                    "channel": ch,
                    "recent_count": len(msgs["messages"]),
                    "latest": msgs["messages"][0] if msgs["messages"] else None,
                })
        return {
            "success": True,
            "channels_checked": len(channels),
            "summaries": summaries,
        }

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a Slack action."""
        a = action.lower().strip()
        if "auth" in a or "test" in a or "connect" in a:
            return self.auth_test()
        if "list" in a or ("channel" in a and "all" in a):
            return self.list_channels(kwargs.get("types", "public_channel"))
        if "read" in a or "history" in a:
            return self.read_channel(kwargs.get("channel") or "", kwargs.get("limit", 20))
        if "send" in a or "post" in a:
            return self.send_message(kwargs.get("channel") or "", kwargs.get("text") or kwargs.get("message") or "")
        if "search" in a:
            return self.search_messages(kwargs.get("query") or action, kwargs.get("count", 20))
        if "mention" in a:
            return self.get_mentions(kwargs.get("limit", 20))
        if "summary" in a or "unread" in a:
            return self.get_unread_summary(kwargs.get("channels"), kwargs.get("limit_per_channel", 5))
        return self.list_channels()
