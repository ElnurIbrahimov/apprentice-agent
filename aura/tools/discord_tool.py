"""Discord Tool — read channels, messages, and post via Discord REST API.

Uses Discord's HTTP API with a Bot Token. No persistent connection needed —
each call is a direct REST request, so it's lightweight and reliable.

Setup:
    1. Create a bot at https://discord.com/developers/applications
    2. Add it to your server with 'Read Messages' + 'Send Messages' permissions
    3. Copy the Bot Token to DISCORD_BOT_TOKEN in .env

Config:
    DISCORD_BOT_TOKEN — Bot token from Discord Developer Portal
"""

import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"
DEFAULT_TIMEOUT = 10


class DiscordTool:
    """Read Discord channels and messages, send messages — team awareness from AURA."""

    name = "discord"
    description = "Read Discord servers/channels/messages and send messages — team awareness and communication"

    def __init__(self):
        self._token = os.getenv("DISCORD_BOT_TOKEN", "")
        self._headers = {
            "Authorization": f"Bot {self._token}",
            "Content-Type": "application/json",
        }
        if self._token:
            logger.info("[Discord] Bot token configured")
        else:
            logger.info("[Discord] No DISCORD_BOT_TOKEN set — configure in .env")

    def _check(self) -> Optional[Dict]:
        if not self._token:
            return {"success": False, "error": "DISCORD_BOT_TOKEN not set in .env — create a bot at discord.com/developers"}
        if not REQUESTS_AVAILABLE:
            return {"success": False, "error": "requests not installed"}
        return None

    def _get(self, path: str, params: Optional[Dict] = None) -> tuple:
        err = self._check()
        if err:
            return None, err["error"]
        try:
            resp = requests.get(f"{DISCORD_API}{path}", headers=self._headers, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 401:
                return None, "Invalid bot token — check DISCORD_BOT_TOKEN"
            if resp.status_code == 403:
                return None, "Bot lacks permission for this action — check server permissions"
            if resp.status_code == 404:
                return None, f"Not found: {path}"
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            return None, str(e)

    def _post(self, path: str, data: Dict) -> tuple:
        err = self._check()
        if err:
            return None, err["error"]
        try:
            resp = requests.post(f"{DISCORD_API}{path}", headers=self._headers, json=data, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 401:
                return None, "Invalid bot token"
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            return None, str(e)

    # ------------------------------------------------------------------ #

    def list_servers(self) -> Dict:
        """List all servers (guilds) the bot is in."""
        data, err = self._get("/users/@me/guilds")
        if err:
            return {"success": False, "error": err}
        return {
            "success": True,
            "count": len(data or []),
            "servers": [{"id": g["id"], "name": g["name"]} for g in (data or [])],
        }

    def list_channels(self, server_id: str) -> Dict:
        """List channels in a server.

        Args:
            server_id: Discord server/guild ID
        """
        data, err = self._get(f"/guilds/{server_id}/channels")
        if err:
            return {"success": False, "error": err}
        channels = [
            {
                "id": c["id"],
                "name": c.get("name", ""),
                "type": {0: "text", 2: "voice", 4: "category", 5: "announcement"}.get(c["type"], "other"),
                "topic": c.get("topic", ""),
            }
            for c in (data or [])
            if c["type"] in (0, 5)  # text and announcement only
        ]
        return {"success": True, "server_id": server_id, "count": len(channels), "channels": channels}

    def read_messages(self, channel_id: str, limit: int = 20) -> Dict:
        """Read recent messages from a channel.

        Args:
            channel_id: Discord channel ID
            limit: Number of messages to retrieve (max 100)
        """
        data, err = self._get(f"/channels/{channel_id}/messages", {"limit": min(limit, 100)})
        if err:
            return {"success": False, "error": err}
        messages = [
            {
                "id": m["id"],
                "author": m["author"]["username"],
                "content": m["content"],
                "timestamp": m["timestamp"][:16],
                "attachments": len(m.get("attachments", [])),
                "reactions": [f"{r['emoji']['name']}×{r['count']}" for r in m.get("reactions", [])],
            }
            for m in (data or [])
        ]
        return {
            "success": True,
            "channel_id": channel_id,
            "count": len(messages),
            "messages": messages,
        }

    def send_message(self, channel_id: str, content: str) -> Dict:
        """Send a message to a channel.

        Args:
            channel_id: Discord channel ID
            content: Message text to send
        """
        if len(content) > 2000:
            content = content[:1997] + "..."
        data, err = self._post(f"/channels/{channel_id}/messages", {"content": content})
        if err:
            return {"success": False, "error": err}
        return {
            "success": True,
            "sent": content[:100],
            "message_id": data.get("id"),
            "channel_id": channel_id,
        }

    def search_messages(self, server_id: str, query: str, channel_id: Optional[str] = None) -> Dict:
        """Search messages in a server.

        Args:
            server_id: Server/guild ID
            query: Search query text
            channel_id: Limit search to specific channel (optional)
        """
        params = {"content": query, "limit": 25}
        if channel_id:
            params["channel_id"] = channel_id
        data, err = self._get(f"/guilds/{server_id}/messages/search", params)
        if err:
            # Fallback: manually read from channel and filter
            if channel_id:
                msgs = self.read_messages(channel_id, 100)
                if msgs.get("success"):
                    q = query.lower()
                    filtered = [m for m in msgs["messages"] if q in m["content"].lower()]
                    return {"success": True, "query": query, "count": len(filtered), "messages": filtered}
            return {"success": False, "error": err}
        messages = data.get("messages", [])
        return {
            "success": True,
            "query": query,
            "total_results": data.get("total_results", 0),
            "messages": [
                {"id": m[0]["id"], "author": m[0]["author"]["username"], "content": m[0]["content"], "timestamp": m[0]["timestamp"][:16]}
                for m in messages
            ],
        }

    def get_channel_info(self, channel_id: str) -> Dict:
        """Get information about a specific channel."""
        data, err = self._get(f"/channels/{channel_id}")
        if err:
            return {"success": False, "error": err}
        return {
            "success": True,
            "id": data["id"],
            "name": data.get("name"),
            "topic": data.get("topic"),
            "type": data.get("type"),
            "guild_id": data.get("guild_id"),
        }

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a Discord action."""
        a = action.lower().strip()
        if ("server" in a or "guild" in a) and "list" in a:
            return self.list_servers()
        if "channel" in a and "list" in a:
            return self.list_channels(kwargs.get("server_id") or kwargs.get("guild_id") or "")
        if "read" in a or ("message" in a and "list" in a):
            return self.read_messages(kwargs.get("channel_id") or "", kwargs.get("limit", 20))
        if "send" in a:
            return self.send_message(kwargs.get("channel_id") or "", kwargs.get("content") or kwargs.get("message") or "")
        if "search" in a:
            return self.search_messages(kwargs.get("server_id") or "", kwargs.get("query") or action, kwargs.get("channel_id"))
        if "info" in a:
            return self.get_channel_info(kwargs.get("channel_id") or "")
        return self.list_servers()
