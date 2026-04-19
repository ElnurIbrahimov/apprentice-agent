"""Agent Communication Protocol (ACP) server for Aura.

Minimal functional implementation — handshake, session lifecycle, prompt
streaming. Enables external agent CLIs (Zed, Claude Code, Codex, Hermes)
to call Aura as a peer agent.

See `server.py` for the JSON-RPC 2.0 stdio server.

Pattern inspired by Hermes Agent's `acp_adapter/` (MIT, Nous Research).
Full spec: https://github.com/zed-industries/agent-client-protocol
"""

from .server import AuraACPServer, run_acp_server

__all__ = ["AuraACPServer", "run_acp_server"]
