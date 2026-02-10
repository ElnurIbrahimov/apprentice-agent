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
# SesameTTS requires torch - make import optional
try:
    from .sesame_tts import SesameTTS
    SESAME_AVAILABLE = True
except ImportError:
    SesameTTS = None
    SESAME_AVAILABLE = False

from .voice_manager import VoiceManager, voice_manager
from .clawdbot import ClawdbotTool, clawdbot, send_message as clawdbot_send
from .evoemo import EvoEmoTool, evoemo, analyze_emotion, get_current_mood, get_mood_emoji
from .evoemo_prompts import get_tone_modifier, get_response_style, build_adaptive_system_prompt
from .inner_monologue import InnerMonologueTool, get_monologue, THOUGHT_TYPES, THOUGHT_ICONS
from .knowledge_graph import KnowledgeGraphTool, get_knowledge_graph, seed_initial_knowledge, Node, Edge, NODE_TYPES, EDGE_TYPES
from .kg_extractor import KnowledgeExtractor, create_extractor
from .hybrid_memory import HybridMemory, create_hybrid_memory, MemoryResult
from .metacog_guardian import (
    MetacognitiveGuardian,
    GuardianConfig,
    FailureType,
    InterventionType,
    FailurePrediction,
    get_guardian
)
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
from .mirrormind import MirrorMind, CritiqueResult
from .cognitive_theater import CognitiveTheater, Deliberation, is_decision_question
from .reflexion import (
    ReflexionEngine,
    Reflection,
    ReflexionResult,
    code_syntax_evaluator,
    function_evaluator,
    json_evaluator,
    answer_completeness_evaluator
)
from .synapseforge import SynapseForge, SynthesizedTool
from .worldsim import WorldSim, RiskLevel, SimulationResult, quick_check
from .amem import AMEMSystem, MemoryNote, get_amem
from .amem_tool import AMEMTool, get_amem_tool
from .hybrid_amem import HybridAMEMSystem, HybridResult, get_hybrid_memory
from .mcts_reasoning import (
    MCTSReasoning,
    MCTSConfig,
    MCTSResult,
    MCTSNode,
    ThoughtType,
    NodeState,
    mcts_reason
)
from .reasoning_tree_tool import ReasoningTreeTool, deep_reason
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
from .introspection_tool import IntrospectionTool, get_introspection_tool
from .calendar_tool import CalendarTool
from .shell_executor import ShellExecutorTool
from .screen_reader import ScreenReaderTool
from .email_tool import EmailTool
from .spaced_repetition import SpacedRepetitionTool

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
    "SESAME_AVAILABLE",
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
    # Shell Executor
    "ShellExecutorTool",
    # Screen Reader
    "ScreenReaderTool",
    # Email
    "EmailTool",
    # Spaced Repetition
    "SpacedRepetitionTool",
]
