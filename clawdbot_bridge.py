"""
Aura-Clawdbot Bridge
Routes incoming Clawdbot messages to Aura agent, sends responses back.

Usage:
    python clawdbot_bridge.py

Requirements:
    - Clawdbot gateway running on port 18789
    - pip install websockets
"""

import asyncio
import json
import sys
from datetime import datetime

try:
    import websockets
except ImportError:
    print("Error: websockets not installed. Run: pip install websockets")
    sys.exit(1)

from apprentice_agent import ApprenticeAgent

GATEWAY_URL = "ws://127.0.0.1:18789"
RECONNECT_DELAY = 5  # seconds


class ClawdbotBridge:
    """Bridge between Clawdbot gateway and Aura agent."""

    def __init__(self):
        self.agent = None
        self.running = False

    def _get_agent(self) -> ApprenticeAgent:
        """Get or create agent instance."""
        if self.agent is None:
            print("[Bridge] Initializing Aura agent...")
            self.agent = ApprenticeAgent()
        return self.agent

    async def handle_message(self, websocket, message_data: dict):
        """Process incoming message and get Aura's response."""
        sender = message_data.get("from", "unknown")
        text = message_data.get("text", "").strip()
        channel = message_data.get("channel", "whatsapp")
        timestamp = datetime.now().strftime("%H:%M:%S")

        if not text:
            return

        print(f"[{timestamp}] [{channel}] {sender}: {text}")

        try:
            agent = self._get_agent()

            # Use chat() for conversational messages
            if agent._is_simple_query(text):
                response = agent.chat(text)
            else:
                result = agent.run(text)
                if result.get("fast_path"):
                    response = result.get("response", "Done.")
                else:
                    response = self._format_result(result)

            print(f"[{timestamp}] [Aura]: {response[:100]}{'...' if len(response) > 100 else ''}")

            # Send response back via Clawdbot
            reply = {
                "type": "message.send",
                "to": sender,
                "channel": channel,
                "text": response
            }

            await websocket.send(json.dumps(reply))

        except Exception as e:
            print(f"[{timestamp}] [Error]: {e}")
            error_reply = {
                "type": "message.send",
                "to": sender,
                "channel": channel,
                "text": f"Sorry, I encountered an error: {str(e)[:100]}"
            }
            await websocket.send(json.dumps(error_reply))

    def _format_result(self, result: dict) -> str:
        """Format agent result for messaging."""
        if result.get("completed"):
            final_eval = result.get("final_evaluation", {})
            progress = final_eval.get("progress", "") if final_eval else ""
            return progress or "Task completed."
        else:
            return f"Task incomplete after {result.get('iterations', 0)} iterations."

    async def connect(self):
        """Connect to Clawdbot gateway and route messages to Aura."""
        print(f"[Bridge] Connecting to Clawdbot gateway at {GATEWAY_URL}...")

        async with websockets.connect(GATEWAY_URL) as websocket:
            print("[Bridge] Connected! Waiting for messages...")
            self.running = True

            # Send registration message
            await websocket.send(json.dumps({
                "type": "agent.register",
                "name": "Aura",
                "capabilities": ["chat", "tasks", "web_search", "code"]
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data.get("type") == "message.received":
                        await self.handle_message(websocket, data)
                    elif data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                    elif data.get("type") == "error":
                        print(f"[Bridge] Gateway error: {data.get('message', 'Unknown')}")

                except json.JSONDecodeError:
                    print(f"[Bridge] Invalid JSON received: {message[:100]}")
                except Exception as e:
                    print(f"[Bridge] Error processing message: {e}")

    async def run(self):
        """Run bridge with auto-reconnect."""
        print("=" * 50)
        print("Aura-Clawdbot Bridge")
        print("=" * 50)

        while True:
            try:
                await self.connect()
            except websockets.exceptions.ConnectionClosed:
                print(f"[Bridge] Connection closed. Reconnecting in {RECONNECT_DELAY}s...")
            except ConnectionRefusedError:
                print(f"[Bridge] Cannot connect to gateway. Is Clawdbot running?")
                print(f"[Bridge] Start it with: clawdbot gateway --port 18789")
                print(f"[Bridge] Retrying in {RECONNECT_DELAY}s...")
            except Exception as e:
                print(f"[Bridge] Error: {e}")
                print(f"[Bridge] Retrying in {RECONNECT_DELAY}s...")

            await asyncio.sleep(RECONNECT_DELAY)


def main():
    """Start the Aura-Clawdbot bridge."""
    bridge = ClawdbotBridge()

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("\n[Bridge] Shutting down...")


if __name__ == "__main__":
    main()
