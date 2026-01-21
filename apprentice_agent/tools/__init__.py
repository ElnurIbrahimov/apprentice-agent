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
]
