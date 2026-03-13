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
    from .tool_builder import ToolBuilderTool
except Exception as _e:
    ToolBuilderTool = None
    _safe_import_error("ToolBuilderTool", _e)

try:
    from .marketplace import MarketplaceTool
except Exception as _e:
    MarketplaceTool = None
    _safe_import_error("MarketplaceTool", _e)

# ---------------------------------------------------------------------------
#  Regex Builder
# ---------------------------------------------------------------------------
try:
    from .regex_builder import RegexBuilderTool
except Exception as _e:
    RegexBuilderTool = None
    _safe_import_error("RegexBuilderTool", _e)

# ---------------------------------------------------------------------------
#  Git
# ---------------------------------------------------------------------------
try:
    from .git_tool import GitTool
except Exception as _e:
    GitTool = None
    _safe_import_error("GitTool", _e)

# ---------------------------------------------------------------------------
#  PersonaPlex
# ---------------------------------------------------------------------------
try:
    from .personaplex import PersonaPlexTool
except Exception as _e:
    PersonaPlexTool = None
    _safe_import_error("PersonaPlexTool", _e)

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

# ---------------------------------------------------------------------------
#  SesameTTS (already guarded - requires torch)
# ---------------------------------------------------------------------------
try:
    from .sesame_tts import SesameTTS
    SESAME_AVAILABLE = True
except ImportError:
    SesameTTS = None
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

# ---------------------------------------------------------------------------
#  Clawdbot
# ---------------------------------------------------------------------------
try:
    from .clawdbot import ClawdbotTool, clawdbot, send_message as clawdbot_send
except Exception as _e:
    ClawdbotTool = None
    clawdbot = None
    clawdbot_send = None
    _safe_import_error("ClawdbotTool/clawdbot/clawdbot_send", _e)

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
#  Hybrid Memory
# ---------------------------------------------------------------------------
try:
    from .hybrid_memory import HybridMemory, create_hybrid_memory, MemoryResult
except Exception as _e:
    HybridMemory = None
    create_hybrid_memory = None
    MemoryResult = None
    _safe_import_error("HybridMemory", _e)

# ---------------------------------------------------------------------------
#  Metacognitive Guardian
# ---------------------------------------------------------------------------
try:
    from .metacog_guardian import (
        MetacognitiveGuardian,
        GuardianConfig,
        FailureType,
        InterventionType,
        FailurePrediction,
        get_guardian
    )
except Exception as _e:
    MetacognitiveGuardian = None
    GuardianConfig = None
    FailureType = None
    InterventionType = None
    FailurePrediction = None
    get_guardian = None
    _safe_import_error("MetacognitiveGuardian", _e)

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
#  MirrorMind
# ---------------------------------------------------------------------------
try:
    from .mirrormind import MirrorMind, CritiqueResult
except Exception as _e:
    MirrorMind = None
    CritiqueResult = None
    _safe_import_error("MirrorMind/CritiqueResult", _e)

# ---------------------------------------------------------------------------
#  Cognitive Theater
# ---------------------------------------------------------------------------
try:
    from .cognitive_theater import CognitiveTheater, Deliberation, is_decision_question
except Exception as _e:
    CognitiveTheater = None
    Deliberation = None
    is_decision_question = None
    _safe_import_error("CognitiveTheater", _e)

# ---------------------------------------------------------------------------
#  Reflexion
# ---------------------------------------------------------------------------
try:
    from .reflexion import (
        ReflexionEngine,
        Reflection,
        ReflexionResult,
        code_syntax_evaluator,
        function_evaluator,
        json_evaluator,
        answer_completeness_evaluator
    )
except Exception as _e:
    ReflexionEngine = None
    Reflection = None
    ReflexionResult = None
    code_syntax_evaluator = None
    function_evaluator = None
    json_evaluator = None
    answer_completeness_evaluator = None
    _safe_import_error("ReflexionEngine", _e)

# ---------------------------------------------------------------------------
#  SynapseForge
# ---------------------------------------------------------------------------
try:
    from .synapseforge import SynapseForge, SynthesizedTool
except Exception as _e:
    SynapseForge = None
    SynthesizedTool = None
    _safe_import_error("SynapseForge/SynthesizedTool", _e)

# ---------------------------------------------------------------------------
#  WorldSim
# ---------------------------------------------------------------------------
try:
    from .worldsim import WorldSim, RiskLevel, SimulationResult, quick_check
except Exception as _e:
    WorldSim = None
    RiskLevel = None
    SimulationResult = None
    quick_check = None
    _safe_import_error("WorldSim", _e)

# ---------------------------------------------------------------------------
#  A-MEM (Zettelkasten Agentic Memory)
# ---------------------------------------------------------------------------
try:
    from .amem import AMEMSystem, MemoryNote, get_amem
except Exception as _e:
    AMEMSystem = None
    MemoryNote = None
    get_amem = None
    _safe_import_error("AMEMSystem", _e)

try:
    from .amem_tool import AMEMTool, get_amem_tool
except Exception as _e:
    AMEMTool = None
    get_amem_tool = None
    _safe_import_error("AMEMTool/get_amem_tool", _e)

# ---------------------------------------------------------------------------
#  Hybrid A-MEM + KG
# ---------------------------------------------------------------------------
try:
    from .hybrid_amem import HybridAMEMSystem, HybridResult, get_hybrid_memory
except Exception as _e:
    HybridAMEMSystem = None
    HybridResult = None
    get_hybrid_memory = None
    _safe_import_error("HybridAMEMSystem", _e)

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
#  Introspection Circuit
# ---------------------------------------------------------------------------
try:
    from .introspection_circuit import (
        IntrospectionCircuit,
        IntrospectionConfig,
        IntrospectionResult,
        IntrospectionAction,
        ConfidenceLevel,
        ConfidenceSignal,
        QueryType,
        create_introspection_circuit,
        quick_confidence_check
    )
except Exception as _e:
    IntrospectionCircuit = None
    IntrospectionConfig = None
    IntrospectionResult = None
    IntrospectionAction = None
    ConfidenceLevel = None
    ConfidenceSignal = None
    QueryType = None
    create_introspection_circuit = None
    quick_confidence_check = None
    _safe_import_error("IntrospectionCircuit", _e)

try:
    from .introspection_tool import IntrospectionTool, get_introspection_tool
except Exception as _e:
    IntrospectionTool = None
    get_introspection_tool = None
    _safe_import_error("IntrospectionTool/get_introspection_tool", _e)

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
#  Shell Executor
# ---------------------------------------------------------------------------
try:
    from .shell_executor import ShellExecutorTool
except Exception as _e:
    ShellExecutorTool = None
    _safe_import_error("ShellExecutorTool", _e)

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
#  Clipboard History
# ---------------------------------------------------------------------------
try:
    from .clipboard_history import ClipboardHistoryTool
except Exception as _e:
    ClipboardHistoryTool = None
    _safe_import_error("ClipboardHistoryTool", _e)

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
    from .clipboard_memory import ClipboardMemoryTool
except Exception as _e:
    ClipboardMemoryTool = None
    _safe_import_error("ClipboardMemoryTool", _e)

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

try:
    from .home_assistant import HomeAssistantTool
except Exception as _e:
    HomeAssistantTool = None
    _safe_import_error("HomeAssistantTool", _e)

try:
    from .task_scheduler import TaskSchedulerTool
except Exception as _e:
    TaskSchedulerTool = None
    _safe_import_error("TaskSchedulerTool", _e)

try:
    from .discord_tool import DiscordTool
except Exception as _e:
    DiscordTool = None
    _safe_import_error("DiscordTool", _e)

try:
    from .slack_tool import SlackTool
except Exception as _e:
    SlackTool = None
    _safe_import_error("SlackTool", _e)

try:
    from .local_image_gen import LocalImageGenTool
except Exception as _e:
    LocalImageGenTool = None
    _safe_import_error("LocalImageGenTool", _e)

# ---------------------------------------------------------------------------
#  Tier 3 Moonshot
# ---------------------------------------------------------------------------
try:
    from .ambient_audio import AmbientAudioTool
except Exception as _e:
    AmbientAudioTool = None
    _safe_import_error("AmbientAudioTool", _e)

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
#  FluxMind (external tools directory - already guarded)
# ---------------------------------------------------------------------------
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
    "MarketplaceTool",
    "FluxMindTool",
    "FLUXMIND_AVAILABLE",
    "RegexBuilderTool",
    "GitTool",
    "PersonaPlexTool",
    "SesameTTS",
    "SESAME_AVAILABLE",
    "VoiceManager",
    "get_voice_manager",
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
    # Metacognitive Guardian
    "MetacognitiveGuardian",
    "GuardianConfig",
    "FailureType",
    "InterventionType",
    "FailurePrediction",
    "get_guardian",
    # NeuroDream
    "NeuroDreamEngine",
    "SleepPhase",
    "DreamTrigger",
    "DreamInsight",
    "SleepSession",
    "ConsolidatedPattern",
    "get_neurodream",
    "create_neurodream",
    # MirrorMind
    "MirrorMind",
    "CritiqueResult",
    # CognitiveTheater
    "CognitiveTheater",
    "Deliberation",
    "is_decision_question",
    # Reflexion - Learn From Mistakes
    "ReflexionEngine",
    "Reflection",
    "ReflexionResult",
    "code_syntax_evaluator",
    "function_evaluator",
    "json_evaluator",
    "answer_completeness_evaluator",
    # SynapseForge - Dynamic Tool Creation
    "SynapseForge",
    "SynthesizedTool",
    # Tool Contract
    "ToolResult",
    "ToolSpec",
    "ToolRegistry",
    "ToolSafety",
    "LatencyTier",
    "get_tool_registry",
    # WorldSim - Consequence Simulation
    "WorldSim",
    "RiskLevel",
    "SimulationResult",
    "quick_check",
    # A-MEM - Zettelkasten Agentic Memory
    "AMEMSystem",
    "MemoryNote",
    "get_amem",
    "AMEMTool",
    "get_amem_tool",
    # Hybrid A-MEM + KG Memory
    "HybridAMEMSystem",
    "HybridResult",
    "get_hybrid_memory",
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
    # Introspection Circuit
    "IntrospectionCircuit",
    "IntrospectionConfig",
    "IntrospectionResult",
    "IntrospectionAction",
    "ConfidenceLevel",
    "ConfidenceSignal",
    "QueryType",
    "create_introspection_circuit",
    "quick_confidence_check",
    "IntrospectionTool",
    "get_introspection_tool",
    # Calendar
    "CalendarTool",
    # Code Search & Edit
    "CodeSearchTool",
    "CodeEditTool",
    # Shell Executor
    "ShellExecutorTool",
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
    # Clipboard History
    "ClipboardHistoryTool",
    # Research
    "ResearchTool",
    # Tier 1 Quick Wins
    "ClipboardMemoryTool",
    "ObsidianTool",
    "GitHubTool",
    "LogAnalystTool",
    "DocumentGeneratorTool",
    # Tier 2 High-Impact
    "WindowsControlTool",
    "HomeAssistantTool",
    "TaskSchedulerTool",
    "DiscordTool",
    "SlackTool",
    "LocalImageGenTool",
    # Tier 3 Moonshot
    "AmbientAudioTool",
    "PredictiveTaskTool",
    "MeetingIntelTool",
    "VoiceSynthTool",
    "LifeLoggerTool",
]

# Filter out symbols that failed to import (are None) — but keep
# SESAME_AVAILABLE and FLUXMIND_AVAILABLE which are intentionally bool.
_ALWAYS_EXPORT = {"SESAME_AVAILABLE", "FLUXMIND_AVAILABLE"}
_ns = vars()
__all__ = [
    s for s in _ALL_SYMBOLS
    if s in _ALWAYS_EXPORT or _ns.get(s) is not None
]
