"""
WhatsApp Integration for AURA

This uses a WebSocket bridge to a Node.js Baileys instance.
Baileys handles the WhatsApp Web connection, we handle the AI logic.

Setup:
1. Install Node.js dependencies: cd baileys_bridge && npm install
2. Run the bridge: node server.js
3. Scan QR code with WhatsApp
4. Run AURA with WhatsApp enabled
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from .base_platform import (
    BasePlatform,
    IncomingMessage,
    OutgoingMessage,
    MessageType
)

logger = logging.getLogger(__name__)


class WhatsAppBot(BasePlatform):
    """
    WhatsApp integration using Baileys (Node.js) + WebSocket bridge.

    Architecture:
    1. Node.js server runs Baileys (WhatsApp Web connection)
    2. Python connects via WebSocket
    3. Messages flow both directions through WebSocket
    """

    def __init__(self, aura_engine, config: dict):
        super().__init__(aura_engine, config)

        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets not installed. "
                "Run: pip install websockets"
            )

        self.ws_url = config.get("websocket_url", "ws://localhost:3001")
        self.session_path = config.get("session_path", "data/messaging/whatsapp_session")
        self.allowed_numbers = config.get("allowed_numbers", [])

        self.websocket = None
        self.active_chats: Dict[str, dict] = {}
        self._load_state()

        # Message queue for outgoing
        self.outgoing_queue: asyncio.Queue = asyncio.Queue()

        # Connection state
        self._connected = False
        self._qr_displayed = False

    @property
    def platform_name(self) -> str:
        return "whatsapp"

    def _load_state(self):
        """Load saved state"""
        state_file = Path("data/messaging/whatsapp_state.json")
        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.active_chats = data.get("active_chats", {})
            except Exception as e:
                logger.warning(f"Could not load WhatsApp state: {e}")

    def _save_state(self):
        """Save state"""
        state_file = Path("data/messaging/whatsapp_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "active_chats": self.active_chats,
                    "last_saved": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save WhatsApp state: {e}")

    async def start(self):
        """Start WhatsApp connection"""

        logger.info("Starting WhatsApp bot...")
        logger.info(f"Connecting to Baileys bridge at {self.ws_url}")

        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.is_running = True

            # Start message receiver
            asyncio.create_task(self._receive_loop())

            # Start message sender
            asyncio.create_task(self._send_loop())

            logger.info("WhatsApp bot connected to bridge!")
            logger.info("Waiting for WhatsApp connection...")

        except Exception as e:
            logger.error(f"Failed to connect to WhatsApp bridge: {e}")
            logger.info("")
            logger.info("Make sure the Baileys Node.js server is running!")
            logger.info("  cd aura/messaging/baileys_bridge")
            logger.info("  npm install")
            logger.info("  node server.js")
            logger.info("")
            raise

    async def stop(self):
        """Stop WhatsApp connection"""

        logger.info("Stopping WhatsApp bot...")
        self.is_running = False

        if self.websocket:
            await self.websocket.close()

        self._save_state()
        logger.info("WhatsApp bot stopped.")

    async def _receive_loop(self):
        """Receive messages from Baileys bridge"""

        while self.is_running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                msg_type = data.get("type")

                if msg_type == "message":
                    await self._handle_incoming_message(data)
                elif msg_type == "qr":
                    if not self._qr_displayed:
                        logger.info("")
                        logger.info("=" * 50)
                        logger.info("Scan this QR code with WhatsApp:")
                        logger.info("(Settings -> Linked Devices -> Link a Device)")
                        logger.info("=" * 50)
                        # QR would be displayed by the Node.js server
                        self._qr_displayed = True
                elif msg_type == "ready":
                    self._connected = True
                    logger.info("")
                    logger.info("WhatsApp connected and ready!")
                    logger.info("")
                elif msg_type == "disconnected":
                    self._connected = False
                    self._qr_displayed = False
                    logger.warning("WhatsApp disconnected!")

            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                logger.warning("[WhatsApp] Connection lost, reconnecting in 5s...")
                await asyncio.sleep(5)
                continue  # retry the outer while True loop
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                await asyncio.sleep(1)

    async def _send_loop(self):
        """Send queued messages to Baileys bridge"""

        while self.is_running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.outgoing_queue.get(),
                    timeout=1.0
                )

                # Send to Baileys
                if self.websocket:
                    await self.websocket.send(json.dumps(message))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in send loop: {e}")

    async def _handle_incoming_message(self, data: dict):
        """Handle incoming message from WhatsApp"""

        # Extract message info
        remote_jid = data.get("from", "")  # e.g., "1234567890@s.whatsapp.net"
        phone = remote_jid.split("@")[0]
        text = data.get("text", "")
        push_name = data.get("pushName", "")
        message_id = data.get("id", "")

        # Check if allowed
        # Normalize: strip '+' from both sides for comparison
        phone_normalized = phone.lstrip('+')
        allowed_normalized = {n.lstrip('+') for n in self.allowed_numbers}
        if allowed_normalized and phone_normalized not in allowed_normalized:
            logger.info(f"Ignoring message from non-allowed number: {phone}")
            return

        if not text:
            return  # Ignore non-text messages for now

        logger.info(f"[WhatsApp] {push_name} ({phone}): {text}")

        # Update active chats
        self.active_chats[remote_jid] = {
            "phone": phone,
            "name": push_name,
            "last_message": datetime.now().isoformat()
        }

        # Create standardized message
        incoming = IncomingMessage(
            platform="whatsapp",
            user_id=phone,
            chat_id=remote_jid,
            username=phone,
            display_name=push_name,
            message_type=MessageType.TEXT,
            text=text,
            media_url=None,
            timestamp=datetime.now(),
            raw_message=data
        )

        # Show typing
        await self.send_typing_indicator(remote_jid)

        # Process through AURA
        response = await self.handle_incoming(incoming)

        if response:
            await self.send_message(OutgoingMessage(
                chat_id=remote_jid,
                text=response
            ))

        self._save_state()

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send a message via Baileys bridge"""

        try:
            await self.outgoing_queue.put({
                "type": "send",
                "to": message.chat_id,
                "text": message.text
            })
            return True
        except Exception as e:
            logger.error(f"Failed to queue message: {e}")
            return False

    async def send_typing_indicator(self, chat_id: str):
        """Show typing indicator"""
        try:
            await self.outgoing_queue.put({
                "type": "typing",
                "to": chat_id
            })
        except (asyncio.QueueFull, Exception) as e:
            logger.debug(f"Could not send typing indicator: {e}")

    # ============ OVERRIDE AURA PROCESSING ============

    async def _process_with_aura(self, text: str, user_id: str) -> str:
        """Process message through AURA engine"""

        # Try fast path first (if available on the engine/agent)
        try:
            if hasattr(self.aura, 'fast_path_handler') and self.aura.fast_path_handler:
                fast_response = self.aura.fast_path_handler.try_fast_path(text)
                if fast_response:
                    return fast_response

        except Exception as e:
            logger.error(f"Fast path error: {e}")

        # Fall back to simple responses
        try:
            if hasattr(self.aura, 'process_input'):
                context = self.aura.process_input(text)
                topic = context.get("topic", "general")

                responses = {
                    "coding": ["Let me help with that!", "Interesting!"],
                    "learning": ["Happy to explain!", "Good question!"],
                    "casual": ["Hey!", "What's up?"],
                    "general": ["I hear you!", "Got it!"]
                }

                base_response = random.choice(responses.get(topic, responses["general"]))

                if hasattr(self.aura, 'process_response'):
                    response = self.aura.process_response(base_response, context)
                    return response.content

                return base_response

        except Exception as e:
            logger.error(f"AURA processing error: {e}")

        return "I'm here! What's up?"

    # ============ PROACTIVE MESSAGING ============

    async def send_to_all_active(self, message: str):
        """Send a message to all active chats"""

        for chat_id in self.active_chats:
            try:
                await self.send_proactive(chat_id, message)
                await asyncio.sleep(0.5)  # Rate limiting (WhatsApp is stricter)
            except Exception as e:
                logger.warning(f"Could not send to {chat_id}: {e}")

    def get_active_chat_ids(self) -> List[str]:
        """Get list of active chat IDs"""
        return list(self.active_chats.keys())

    @property
    def is_connected(self) -> bool:
        """Check if WhatsApp is connected"""
        return self._connected
