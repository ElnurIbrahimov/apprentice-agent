"""Tools available to the agent.

Imports are individually wrapped in try/except so that a single broken tool
module never kills the entire tools package.  Failed imports are logged as
warnings and the symbol is set to None.
"""

import logging as _logging

_log = _logging.getLogger(__name__)
_TOOLS_IMPORT_ERRORS: list[str] = []


def _safe_import_error(name: str, exc: Exception) -> None:
    """Record and log a failed tool import."""
    msg = f"[tools/__init__] Failed to import {name}: {exc}"
    _log.warning(msg)
    _TOOLS_IMPORT_ERRORS.append(msg)


# ---------------------------------------------------------------------------
#  Core / Filesystem
# ---------------------------------------------------------------------------
try:
    from .filesystem import FileSystemTool
except Exception as _e:
    FileSystemTool = None
    _safe_import_error("FileSystemTool", _e)

# ---------------------------------------------------------------------------
#  Web / Search
# ---------------------------------------------------------------------------
try:
    from .web_search import WebSearchTool
except Exception as _e:
    WebSearchTool = None
    _safe_import_error("WebSearchTool", _e)

try:
    from .brave_search import BraveSearchTool
except Exception as _e:
    BraveSearchTool = None
    _safe_import_error("BraveSearchTool", _e)

try:
    from .tavily_tool import TavilyTool
except Exception as _e:
    TavilyTool = None
    _safe_import_error("TavilyTool", _e)

try:
    from .firecrawl_tool import FirecrawlTool
except Exception as _e:
    FirecrawlTool = None
    _safe_import_error("FirecrawlTool", _e)

# ---------------------------------------------------------------------------
#  Code / Execution
# ---------------------------------------------------------------------------
try:
    from .code_executor import CodeExecutorTool
except Exception as _e:
    CodeExecutorTool = None
    _safe_import_error("CodeExecutorTool", _e)

# ---------------------------------------------------------------------------
#  Vision / Media
# ---------------------------------------------------------------------------
try:
    from .screenshot import ScreenshotTool
except Exception as _e:
    ScreenshotTool = None
    _safe_import_error("ScreenshotTool", _e)

try:
    from .vision import VisionTool
except Exception as _e:
    VisionTool = None
    _safe_import_error("VisionTool", _e)

try:
    from .pdf_reader import PDFReaderTool
except Exception as _e:
    PDFReaderTool = None
    _safe_import_error("PDFReaderTool", _e)

# ---------------------------------------------------------------------------
#  Clipboard
# ---------------------------------------------------------------------------
try:
    from .clipboard import ClipboardTool
except Exception as _e:
    ClipboardTool = None
    _safe_import_error("ClipboardTool", _e)

# ---------------------------------------------------------------------------
#  Load Skill (progressive skill loading)
# ---------------------------------------------------------------------------
try:
    from .load_skill import LoadSkillTool
except Exception as _e:
    LoadSkillTool = None
    _safe_import_error("LoadSkillTool", _e)

# ---------------------------------------------------------------------------
#  Tool Search (deferred tool discovery)
# ---------------------------------------------------------------------------
try:
    from .tool_search import ToolSearchTool
except Exception as _e:
    ToolSearchTool = None
    _safe_import_error("ToolSearchTool", _e)

try:
    from .deferred_registry import DeferredToolRegistry, deferred_registry
except Exception as _e:
    DeferredToolRegistry = None
    deferred_registry = None
    _safe_import_error("DeferredToolRegistry", _e)

# ---------------------------------------------------------------------------
#  Voice / Audio
# ---------------------------------------------------------------------------
try:
    from .voice import VoiceTool, VoiceConversation
except Exception as _e:
    VoiceTool = None
    VoiceConversation = None
    _safe_import_error("VoiceTool/VoiceConversation", _e)

# ---------------------------------------------------------------------------
#  Image Generation
# ---------------------------------------------------------------------------
try:
    from .image_gen import ImageGenerationTool, generate_image
except Exception as _e:
    ImageGenerationTool = None
    generate_image = None
    _safe_import_error("ImageGenerationTool/generate_image", _e)

# ---------------------------------------------------------------------------
#  Arxiv
# ---------------------------------------------------------------------------
try:
    from .arxiv_search import ArxivSearchTool
except Exception as _e:
    ArxivSearchTool = None
    _safe_import_error("ArxivSearchTool", _e)

# ---------------------------------------------------------------------------
#  Browser
# ---------------------------------------------------------------------------
try:
    from .browser import BrowserTool
except Exception as _e:
    BrowserTool = None
    _safe_import_error("BrowserTool", _e)

# ---------------------------------------------------------------------------
#  System Control
# ---------------------------------------------------------------------------
try:
    from .system_control import SystemControlTool
except Exception as _e:
    SystemControlTool = None
    _safe_import_error("SystemControlTool", _e)

# ---------------------------------------------------------------------------
#  Notifications
# ---------------------------------------------------------------------------
try:
    from .notifications import NotificationTool
except Exception as _e:
    NotificationTool = None
    _safe_import_error("NotificationTool", _e)

# ---------------------------------------------------------------------------
#  Tool Builder / Marketplace
# ---------------------------------------------------------------------------
try:
    from .tool_builder import ToolBuilderTool, ToolUsageTracker, get_usage_tracker
except Exception as _e:
    ToolBuilderTool = None
    ToolUsageTracker = None
    get_usage_tracker = None
    _safe_import_error("ToolBuilderTool", _e)

try:
    from .marketplace import MarketplaceTool
except Exception as _e:
    MarketplaceTool = None
    _safe_import_error("MarketplaceTool", _e)

# RegexBuilderTool removed — LLMs handle regex natively (2026-03-18)
# RegexBuilderTool — removed

# ---------------------------------------------------------------------------
#  Git
# ---------------------------------------------------------------------------
try:
    from .git_tool import GitTool
except Exception as _e:
    GitTool = None
    _safe_import_error("GitTool", _e)

# ---------------------------------------------------------------------------
# PersonaPlex removed — using external voice provider

# ---------------------------------------------------------------------------
#  Tool Contract (ToolResult, ToolSpec, etc.)
# ---------------------------------------------------------------------------
try:
    from .tool_contract import ToolResult, ToolSpec, ToolRegistry, ToolSafety, LatencyTier, get_tool_registry
except Exception as _e:
    ToolResult = None
    ToolSpec = None
    ToolRegistry = None
    ToolSafety = None
    LatencyTier = None
    get_tool_registry = None
    _safe_import_error("ToolResult/ToolSpec/ToolRegistry", _e)

# SesameTTS removed — using external voice provider
SESAME_AVAILABLE = False

# ---------------------------------------------------------------------------
#  Voice Manager
# ---------------------------------------------------------------------------
try:
    from .voice_manager import VoiceManager, get_voice_manager
except Exception as _e:
    VoiceManager = None
    get_voice_manager = None
    _safe_import_error("VoiceManager/get_voice_manager", _e)

# ClawdbotTool removed (2026-03-18)

# ---------------------------------------------------------------------------
#  EvoEmo
# ---------------------------------------------------------------------------
try:
    from .evoemo import EvoEmoTool, evoemo, analyze_emotion, get_current_mood, get_mood_emoji
except Exception as _e:
    EvoEmoTool = None
    evoemo = None
    analyze_emotion = None
    get_current_mood = None
    get_mood_emoji = None
    _safe_import_error("EvoEmoTool", _e)

try:
    from .evoemo_prompts import get_tone_modifier, get_response_style, build_adaptive_system_prompt
except Exception as _e:
    get_tone_modifier = None
    get_response_style = None
    build_adaptive_system_prompt = None
    _safe_import_error("evoemo_prompts", _e)

# ---------------------------------------------------------------------------
#  Inner Monologue
# ---------------------------------------------------------------------------
try:
    from .inner_monologue import InnerMonologueTool, get_monologue, THOUGHT_TYPES, THOUGHT_ICONS
except Exception as _e:
    InnerMonologueTool = None
    get_monologue = None
    THOUGHT_TYPES = None
    THOUGHT_ICONS = None
    _safe_import_error("InnerMonologueTool", _e)

# ---------------------------------------------------------------------------
#  Knowledge Graph
# ---------------------------------------------------------------------------
try:
    from .knowledge_graph import KnowledgeGraphTool, get_knowledge_graph, seed_initial_knowledge, Node, Edge, NODE_TYPES, EDGE_TYPES
except Exception as _e:
    KnowledgeGraphTool = None
    get_knowledge_graph = None
    seed_initial_knowledge = None
    Node = None
    Edge = None
    NODE_TYPES = None
    EDGE_TYPES = None
    _safe_import_error("KnowledgeGraphTool", _e)

try:
    from .kg_extractor import KnowledgeExtractor, create_extractor
except Exception as _e:
    KnowledgeExtractor = None
    create_extractor = None
    _safe_import_error("KnowledgeExtractor/create_extractor", _e)

# ---------------------------------------------------------------------------
#  NeuroDream
# ---------------------------------------------------------------------------
try:
    from .neurodream import (
        NeuroDreamEngine,
        SleepPhase,
        DreamTrigger,
        DreamInsight,
        SleepSession,
        ConsolidatedPattern,
        get_neurodream,
        create_neurodream
    )
except Exception as _e:
    NeuroDreamEngine = None
    SleepPhase = None
    DreamTrigger = None
    DreamInsight = None
    SleepSession = None
    ConsolidatedPattern = None
    get_neurodream = None
    create_neurodream = None
    _safe_import_error("NeuroDreamEngine", _e)


# ---------------------------------------------------------------------------
#  A-MEM (Zettelkasten Agentic Memory) — REMOVED 2026-03-22
#  amem.py, amem_tool.py, hybrid_amem.py deleted; consolidated into UnifiedMemory
# ---------------------------------------------------------------------------
AMEMSystem = None
MemoryNote = None
get_amem = None
AMEMTool = None
get_amem_tool = None
HybridAMEMSystem = None
HybridResult = None
get_hybrid_memory = None

# ---------------------------------------------------------------------------
#  MCTS Reasoning Tree
# ---------------------------------------------------------------------------
try:
    from .mcts_reasoning import (
        MCTSReasoning,
        MCTSConfig,
        MCTSResult,
        MCTSNode,
        ThoughtType,
        NodeState,
        mcts_reason
    )
except Exception as _e:
    MCTSReasoning = None
    MCTSConfig = None
    MCTSResult = None
    MCTSNode = None
    ThoughtType = None
    NodeState = None
    mcts_reason = None
    _safe_import_error("MCTSReasoning", _e)

try:
    from .reasoning_tree_tool import ReasoningTreeTool, deep_reason
except Exception as _e:
    ReasoningTreeTool = None
    deep_reason = None
    _safe_import_error("ReasoningTreeTool/deep_reason", _e)


# ---------------------------------------------------------------------------
#  Calendar
# ---------------------------------------------------------------------------
try:
    from .calendar_tool import CalendarTool
except Exception as _e:
    CalendarTool = None
    _safe_import_error("CalendarTool", _e)

# ---------------------------------------------------------------------------
#  Code Search (grep, glob, definition finder)
# ---------------------------------------------------------------------------
try:
    from .code_search import CodeSearchTool
except Exception as _e:
    CodeSearchTool = None
    _safe_import_error("CodeSearchTool", _e)

# ---------------------------------------------------------------------------
#  Code Edit (surgical find-replace edits)
# ---------------------------------------------------------------------------
try:
    from .code_edit import CodeEditTool
except Exception as _e:
    CodeEditTool = None
    _safe_import_error("CodeEditTool", _e)

# ---------------------------------------------------------------------------
#  Codebase Index (semantic codebase indexing with incremental updates)
# ---------------------------------------------------------------------------
try:
    from .codebase_index import CodebaseIndex
except Exception as _e:
    CodebaseIndex = None
    _safe_import_error("CodebaseIndex", _e)

# ---------------------------------------------------------------------------
#  Shell Executor
# ---------------------------------------------------------------------------
try:
    from .shell_executor import ShellExecutorTool
except Exception as _e:
    ShellExecutorTool = None
    _safe_import_error("ShellExecutorTool", _e)

# ---------------------------------------------------------------------------
#  Sandbox Executor (E2B cloud + local subprocess fallback)
# ---------------------------------------------------------------------------
try:
    from ..sandbox import SandboxExecutor, ExecutionResult
except Exception as _e:
    SandboxExecutor = None
    ExecutionResult = None
    _safe_import_error("SandboxExecutor", _e)

# ---------------------------------------------------------------------------
#  Screen Reader
# ---------------------------------------------------------------------------
try:
    from .screen_reader import ScreenReaderTool
except Exception as _e:
    ScreenReaderTool = None
    _safe_import_error("ScreenReaderTool", _e)

# ---------------------------------------------------------------------------
#  Email
# ---------------------------------------------------------------------------
try:
    from .email_tool import EmailTool
except Exception as _e:
    EmailTool = None
    _safe_import_error("EmailTool", _e)

# ---------------------------------------------------------------------------
#  Spaced Repetition
# ---------------------------------------------------------------------------
try:
    from .spaced_repetition import SpacedRepetitionTool
except Exception as _e:
    SpacedRepetitionTool = None
    _safe_import_error("SpacedRepetitionTool", _e)

# ---------------------------------------------------------------------------
#  Task Manager
# ---------------------------------------------------------------------------
try:
    from .task_manager import TaskManagerTool
except Exception as _e:
    TaskManagerTool = None
    _safe_import_error("TaskManagerTool", _e)

# ---------------------------------------------------------------------------
#  API Tester
# ---------------------------------------------------------------------------
try:
    from .api_tester import APITesterTool
except Exception as _e:
    APITesterTool = None
    _safe_import_error("APITesterTool", _e)

# ---------------------------------------------------------------------------
#  Database
# ---------------------------------------------------------------------------
try:
    from .database_tool import DatabaseTool
except Exception as _e:
    DatabaseTool = None
    _safe_import_error("DatabaseTool", _e)

# ---------------------------------------------------------------------------
#  Audio Transcriber
# ---------------------------------------------------------------------------
try:
    from .audio_transcriber import AudioTranscriberTool
except Exception as _e:
    AudioTranscriberTool = None
    _safe_import_error("AudioTranscriberTool", _e)

# ---------------------------------------------------------------------------
#  Research
# ---------------------------------------------------------------------------
try:
    from .research_tool import ResearchTool
except Exception as _e:
    ResearchTool = None
    _safe_import_error("ResearchTool", _e)

# ---------------------------------------------------------------------------
#  Tier 1 Quick Wins
# ---------------------------------------------------------------------------
try:
    from .obsidian_tool import ObsidianTool
except Exception as _e:
    ObsidianTool = None
    _safe_import_error("ObsidianTool", _e)

try:
    from .github_tool import GitHubTool
except Exception as _e:
    GitHubTool = None
    _safe_import_error("GitHubTool", _e)

try:
    from .log_analyst import LogAnalystTool
except Exception as _e:
    LogAnalystTool = None
    _safe_import_error("LogAnalystTool", _e)

try:
    from .document_generator import DocumentGeneratorTool
except Exception as _e:
    DocumentGeneratorTool = None
    _safe_import_error("DocumentGeneratorTool", _e)

# ---------------------------------------------------------------------------
#  Tier 2 High-Impact
# ---------------------------------------------------------------------------
try:
    from .windows_control import WindowsControlTool
except Exception as _e:
    WindowsControlTool = None
    _safe_import_error("WindowsControlTool", _e)

# HomeAssistantTool removed (2026-03-18)

try:
    from .task_scheduler import TaskSchedulerTool
except Exception as _e:
    TaskSchedulerTool = None
    _safe_import_error("TaskSchedulerTool", _e)

# DiscordTool, SlackTool, LocalImageGenTool removed (2026-03-18)

# ---------------------------------------------------------------------------
#  Tier 3 Moonshot
# ---------------------------------------------------------------------------
# AmbientAudioTool removed (2026-03-18)

try:
    from .predictive_tasks import PredictiveTaskTool
except Exception as _e:
    PredictiveTaskTool = None
    _safe_import_error("PredictiveTaskTool", _e)

try:
    from .meeting_intel import MeetingIntelTool
except Exception as _e:
    MeetingIntelTool = None
    _safe_import_error("MeetingIntelTool", _e)

try:
    from .voice_synth import VoiceSynthTool
except Exception as _e:
    VoiceSynthTool = None
    _safe_import_error("VoiceSynthTool", _e)

try:
    from .life_logger import LifeLoggerTool
except Exception as _e:
    LifeLoggerTool = None
    _safe_import_error("LifeLoggerTool", _e)

# ---------------------------------------------------------------------------
#  Deploy
# ---------------------------------------------------------------------------
try:
    from .deploy_tool import DeployTool, get_deploy_tool
except Exception as _e:
    DeployTool = None
    get_deploy_tool = None
    _safe_import_error("DeployTool/get_deploy_tool", _e)

# ---------------------------------------------------------------------------
#  Scaffold (project generator)
# ---------------------------------------------------------------------------
try:
    from .scaffold import ScaffoldTool
except Exception as _e:
    ScaffoldTool = None
    _safe_import_error("ScaffoldTool", _e)

# ---------------------------------------------------------------------------
#  Extension Feed (captured website designs)
# ---------------------------------------------------------------------------
try:
    from .extension_feed import ExtensionFeedTool, get_feed_tool
except Exception as _e:
    ExtensionFeedTool = None
    get_feed_tool = None
    _safe_import_error("ExtensionFeedTool/get_feed_tool", _e)

# ---------------------------------------------------------------------------
#  Visual Feedback Loop (headless render + screenshot iteration)
# ---------------------------------------------------------------------------
try:
    from .visual_feedback import VisualFeedbackLoop, get_visual_feedback
except Exception as _e:
    VisualFeedbackLoop = None
    get_visual_feedback = None
    _safe_import_error("VisualFeedbackLoop/get_visual_feedback", _e)

# ---------------------------------------------------------------------------
#  Component Registry (on-demand UI component templates)
# ---------------------------------------------------------------------------
try:
    from .component_registry import ComponentRegistryTool
except Exception as _e:
    ComponentRegistryTool = None
    _safe_import_error("ComponentRegistryTool", _e)


# ---------------------------------------------------------------------------
#  __all__ — only export symbols that actually imported successfully
# ---------------------------------------------------------------------------
_ALL_SYMBOLS = [
    "FileSystemTool",
    "WebSearchTool",
    "BraveSearchTool",
    "TavilyTool",
    "FirecrawlTool",
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
    "ToolUsageTracker",
    "get_usage_tracker",
    "MarketplaceTool",
    "GitTool",
    "VoiceManager",
    "get_voice_manager",
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
    # NeuroDream
    "NeuroDreamEngine",
    "SleepPhase",
    "DreamTrigger",
    "DreamInsight",
    "SleepSession",
    "ConsolidatedPattern",
    "get_neurodream",
    "create_neurodream",
    # Tool Contract
    "ToolResult",
    "ToolSpec",
    "ToolRegistry",
    "ToolSafety",
    "LatencyTier",
    "get_tool_registry",
    # A-MEM / HybridAMEM removed 2026-03-22 (consolidated into UnifiedMemory)
    # MCTS Reasoning Tree
    "MCTSReasoning",
    "MCTSConfig",
    "MCTSResult",
    "MCTSNode",
    "ThoughtType",
    "NodeState",
    "mcts_reason",
    "ReasoningTreeTool",
    "deep_reason",
    # Calendar
    "CalendarTool",
    # Code Search & Edit
    "CodeSearchTool",
    "CodeEditTool",
    # Load Skill (progressive skill loading)
    "LoadSkillTool",
    # Codebase Index
    "CodebaseIndex",
    # Shell Executor
    "ShellExecutorTool",
    # Sandbox Executor
    "SandboxExecutor",
    "ExecutionResult",
    # Screen Reader
    "ScreenReaderTool",
    # Email
    "EmailTool",
    # Spaced Repetition
    "SpacedRepetitionTool",
    # Task Manager
    "TaskManagerTool",
    # API Tester
    "APITesterTool",
    # Database
    "DatabaseTool",
    # Audio Transcriber
    "AudioTranscriberTool",
    # Research
    "ResearchTool",
    # Tier 1 Quick Wins
    "ObsidianTool",
    "GitHubTool",
    "LogAnalystTool",
    "DocumentGeneratorTool",
    # Tier 2 High-Impact
    "WindowsControlTool",
    "TaskSchedulerTool",
    # Tier 3 Moonshot
    "PredictiveTaskTool",
    "MeetingIntelTool",
    "VoiceSynthTool",
    "LifeLoggerTool",
    # Deploy
    "DeployTool",
    "get_deploy_tool",
    # Scaffold
    "ScaffoldTool",
    # Extension Feed
    "ExtensionFeedTool",
    "get_feed_tool",
    # Tool Search / Deferred Registry
    "ToolSearchTool",
    "DeferredToolRegistry",
    "deferred_registry",
    # Load Skill
    "LoadSkillTool",
    # Visual Feedback Loop
    "VisualFeedbackLoop",
    "get_visual_feedback",
    # Component Registry
    "ComponentRegistryTool",
]

# Filter out symbols that failed to import (are None) — but keep
# SESAME_AVAILABLE which is intentionally bool.
_ALWAYS_EXPORT = set()  # No forced exports for removed modules
_ns = vars()
__all__ = [
    s for s in _ALL_SYMBOLS
    if s in _ALWAYS_EXPORT or _ns.get(s) is not None
]
