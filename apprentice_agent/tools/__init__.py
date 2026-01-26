"""Tools available to the agent."""

from .filesystem import FileSystemTool
from .web_search import WebSearchTool
from .code_executor import CodeExecutorTool
from .screenshot import ScreenshotTool
from .vision import VisionTool
from .pdf_reader import PDFReaderTool
from .clipboard import ClipboardTool
from .voice import VoiceTool, VoiceConversation
from .image_gen import ImageGenerationTool, generate_image
from .arxiv_search import ArxivSearchTool
from .browser import BrowserTool
from .system_control import SystemControlTool
from .notifications import NotificationTool
from .tool_builder import ToolBuilderTool
from .marketplace import MarketplaceTool
from .regex_builder import RegexBuilderTool
from .git_tool import GitTool
from .personaplex import PersonaPlexTool
from .sesame_tts import SesameTTS
from .voice_manager import VoiceManager, voice_manager
from .clawdbot import ClawdbotTool, clawdbot, send_message as clawdbot_send
from .evoemo import EvoEmoTool, evoemo, analyze_emotion, get_current_mood, get_mood_emoji
from .evoemo_prompts import get_tone_modifier, get_response_style, build_adaptive_system_prompt
from .inner_monologue import InnerMonologueTool, get_monologue, THOUGHT_TYPES, THOUGHT_ICONS
from .knowledge_graph import KnowledgeGraphTool, get_knowledge_graph, seed_initial_knowledge, Node, Edge, NODE_TYPES, EDGE_TYPES
from .kg_extractor import KnowledgeExtractor, create_extractor
from .hybrid_memory import HybridMemory, create_hybrid_memory, MemoryResult

# Import FluxMind from external tools directory
import sys
from pathlib import Path
_tools_dir = Path(__file__).parent.parent.parent / "tools"
if _tools_dir.exists() and str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

try:
    from fluxmind import FluxMindTool
    FLUXMIND_AVAILABLE = True
except ImportError:
    FluxMindTool = None
    FLUXMIND_AVAILABLE = False

__all__ = [
    "FileSystemTool",
    "WebSearchTool",
    "CodeExecutorTool",
    "ScreenshotTool",
    "VisionTool",
    "PDFReaderTool",
    "ClipboardTool",
    "VoiceTool",
    "VoiceConversation",
    "ImageGenerationTool",
    "generate_image",
    "ArxivSearchTool",
    "BrowserTool",
    "SystemControlTool",
    "NotificationTool",
    "ToolBuilderTool",
    "MarketplaceTool",
    "FluxMindTool",
    "FLUXMIND_AVAILABLE",
    "RegexBuilderTool",
    "GitTool",
    "PersonaPlexTool",
    "SesameTTS",
    "VoiceManager",
    "voice_manager",
    "ClawdbotTool",
    "clawdbot",
    "clawdbot_send",
    "EvoEmoTool",
    "evoemo",
    "analyze_emotion",
    "get_current_mood",
    "get_mood_emoji",
    "get_tone_modifier",
    "get_response_style",
    "build_adaptive_system_prompt",
    "InnerMonologueTool",
    "get_monologue",
    "THOUGHT_TYPES",
    "THOUGHT_ICONS",
    # Knowledge Graph
    "KnowledgeGraphTool",
    "get_knowledge_graph",
    "seed_initial_knowledge",
    "Node",
    "Edge",
    "NODE_TYPES",
    "EDGE_TYPES",
    "KnowledgeExtractor",
    "create_extractor",
    "HybridMemory",
    "create_hybrid_memory",
    "MemoryResult",
]
