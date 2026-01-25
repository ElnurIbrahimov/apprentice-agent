"""
Clawdbot Integration - Tool #19
Send and receive messages via WhatsApp, Telegram, Discord, Signal, iMessage.
"""

import subprocess
import json
from typing import Optional


class ClawdbotTool:
    """Tool for sending/receiving messages via various messaging platforms."""

    def __init__(self):
        self.name = "clawdbot"
        self.description = "Send/receive messages via WhatsApp, Telegram, Discord, Signal, iMessage"
        self._available = None

    def is_available(self) -> bool:
        """Check if Clawdbot CLI is installed and gateway is running."""
        if self._available is not None:
            return self._available

        try:
            result = subprocess.run(
                ["clawdbot", "status"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            self._available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            self._available = False

        return self._available

    def send_message(
        self,
        to: str,
        message: str,
        channel: str = "whatsapp"
    ) -> dict:
        """
        Send a message via Clawdbot.

        Args:
            to: Phone number (+1234567890) or username
            message: Text to send
            channel: whatsapp, telegram, discord, signal, imessage

        Returns:
            {"success": bool, "message_id": str or None, "error": str or None}
        """
        if not self.is_available():
            return {"success": False, "error": "Clawdbot not available"}

        valid_channels = ["whatsapp", "telegram", "discord", "signal", "imessage"]
        if channel.lower() not in valid_channels:
            return {"success": False, "error": f"Invalid channel. Use: {', '.join(valid_channels)}"}

        try:
            cmd = ["clawdbot", "message", "send", "--to", to, "--message", message]

            if channel.lower() != "whatsapp":
                cmd.extend(["--channel", channel.lower()])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout.strip(),
                    "message_id": self._parse_message_id(result.stdout)
                }
            else:
                return {"success": False, "error": result.stderr.strip() or "Unknown error"}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_message_id(self, output: str) -> Optional[str]:
        """Parse message ID from Clawdbot output."""
        try:
            data = json.loads(output)
            return data.get("message_id") or data.get("id")
        except (json.JSONDecodeError, AttributeError):
            return None

    def get_status(self) -> dict:
        """Get Clawdbot gateway status."""
        try:
            result = subprocess.run(
                ["clawdbot", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            return {
                "running": result.returncode == 0,
                "status": result.stdout.strip() if result.returncode == 0 else "Not running",
                "error": result.stderr.strip() if result.returncode != 0 else None
            }
        except FileNotFoundError:
            return {"running": False, "status": "Not installed", "error": "Clawdbot CLI not found"}
        except Exception as e:
            return {"running": False, "status": "Error", "error": str(e)}

    def list_channels(self) -> dict:
        """List connected messaging channels."""
        if not self.is_available():
            return {"success": False, "channels": [], "error": "Clawdbot not available"}

        try:
            result = subprocess.run(
                ["clawdbot", "channels", "list"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            if result.returncode == 0:
                try:
                    channels = json.loads(result.stdout)
                    return {"success": True, "channels": channels}
                except json.JSONDecodeError:
                    # Plain text output
                    return {"success": True, "channels": result.stdout.strip().split("\n")}
            else:
                return {"success": False, "channels": [], "error": result.stderr.strip()}

        except Exception as e:
            return {"success": False, "channels": [], "error": str(e)}

    def start_gateway(self, port: int = 18789) -> dict:
        """Start the Clawdbot gateway in background."""
        try:
            # Check if already running
            status = self.get_status()
            if status.get("running"):
                return {"success": True, "message": "Gateway already running"}

            # Start gateway in background
            subprocess.Popen(
                ["clawdbot", "gateway", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            self._available = None  # Reset cache
            return {"success": True, "message": f"Gateway starting on port {port}"}

        except FileNotFoundError:
            return {"success": False, "error": "Clawdbot CLI not found. Install it first."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_gateway(self) -> dict:
        """Stop the Clawdbot gateway."""
        try:
            result = subprocess.run(
                ["clawdbot", "gateway", "stop"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            self._available = None  # Reset cache

            if result.returncode == 0:
                return {"success": True, "message": "Gateway stopped"}
            else:
                return {"success": False, "error": result.stderr.strip() or "Failed to stop"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def pair_channel(self, channel: str) -> dict:
        """
        Initiate pairing for a channel (e.g., WhatsApp QR code).

        Args:
            channel: Channel to pair (whatsapp, telegram, etc.)

        Returns:
            Instructions or QR code data for pairing
        """
        try:
            result = subprocess.run(
                ["clawdbot", "channels", "login", "--channel", channel.lower()],
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout.strip()}
            else:
                return {"success": False, "error": result.stderr.strip() or "Pairing failed"}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Pairing timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute(self, action: str, **kwargs) -> dict:
        """
        Execute a Clawdbot action.

        Args:
            action: send_message, get_status, list_channels, start_gateway, stop_gateway, pair_channel
            **kwargs: Action-specific arguments

        Returns:
            Action result dict
        """
        actions = {
            "send_message": self.send_message,
            "get_status": self.get_status,
            "list_channels": self.list_channels,
            "start_gateway": self.start_gateway,
            "stop_gateway": self.stop_gateway,
            "pair_channel": self.pair_channel,
        }

        if action not in actions:
            return {"success": False, "error": f"Unknown action: {action}. Available: {', '.join(actions.keys())}"}

        try:
            return actions[action](**kwargs)
        except TypeError as e:
            return {"success": False, "error": f"Invalid arguments: {e}"}


# Singleton instance
clawdbot = ClawdbotTool()

# Convenience functions
def send_message(to: str, message: str, channel: str = "whatsapp") -> dict:
    """Send a message via Clawdbot."""
    return clawdbot.send_message(to, message, channel)

def get_status() -> dict:
    """Get Clawdbot gateway status."""
    return clawdbot.get_status()

def list_channels() -> dict:
    """List connected messaging channels."""
    return clawdbot.list_channels()

def start_gateway(port: int = 18789) -> dict:
    """Start the Clawdbot gateway."""
    return clawdbot.start_gateway(port)

def is_available() -> bool:
    """Check if Clawdbot is available."""
    return clawdbot.is_available()


# Tool detection keywords for agent integration
CLAWDBOT_KEYWORDS = [
    "send message", "send whatsapp", "send telegram",
    "text", "message to", "whatsapp", "telegram",
    "discord message", "imessage", "signal message",
    "clawdbot"
]

def matches_keywords(user_input: str) -> bool:
    """Check if user input matches Clawdbot keywords."""
    lower = user_input.lower()
    return any(kw in lower for kw in CLAWDBOT_KEYWORDS)
