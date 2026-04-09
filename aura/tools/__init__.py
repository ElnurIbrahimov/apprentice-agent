"""Tools available to the agent.

Lazy-loading module: tool classes and functions are NOT imported at package
load time.  Instead, ``__getattr__`` resolves them on first access so that
heavy dependencies (torch, networkx, sounddevice, numpy, etc.)
are only pulled in when actually needed.

Failed imports are logged as warnings and the symbol resolves to None.
"""

import importlib as _importlib
import logging as _logging

_log = _logging.getLogger(__name__)
_TOOLS_IMPORT_ERRORS: list[str] = []

# ---------------------------------------------------------------------------
#  Lazy import registry:  symbol_name -> (module_path, attribute_name)
#
#  module_path is relative to aura.tools unless it starts with ".."
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core / Filesystem
    "FileSystemTool":           (".filesystem",           "FileSystemTool"),
    # Web / Search
    "WebSearchTool":            (".web_search",           "WebSearchTool"),
    "BraveSearchTool":          (".brave_search",         "BraveSearchTool"),
    "TavilyTool":               (".tavily_tool",          "TavilyTool"),
    "FirecrawlTool":            (".firecrawl_tool",       "FirecrawlTool"),
    # Code / Execution
    "CodeExecutorTool":         (".code_executor",        "CodeExecutorTool"),
    # Vision / Media
    "ScreenshotTool":           (".screenshot",           "ScreenshotTool"),
    "VisionTool":               (".vision",               "VisionTool"),
    "PDFReaderTool":            (".pdf_reader",           "PDFReaderTool"),
    # Clipboard
    "ClipboardTool":            (".clipboard",            "ClipboardTool"),
    # Load Skill
    "LoadSkillTool":            (".load_skill",           "LoadSkillTool"),
    # Tool Search / Deferred Registry
    "ToolSearchTool":           (".tool_search",          "ToolSearchTool"),
    "DeferredToolRegistry":     (".deferred_registry",    "DeferredToolRegistry"),
    "deferred_registry":        (".deferred_registry",    "deferred_registry"),
    # Voice / Audio
    "VoiceTool":                (".voice",                "VoiceTool"),
    "VoiceConversation":        (".voice",                "VoiceConversation"),
    # Image Generation
    "ImageGenerationTool":      (".image_gen",            "ImageGenerationTool"),
    "generate_image":           (".image_gen",            "generate_image"),
    # Arxiv
    "ArxivSearchTool":          (".arxiv_search",         "ArxivSearchTool"),
    # Browser
    "BrowserTool":              (".browser",              "BrowserTool"),
    # System Control
    "SystemControlTool":        (".system_control",       "SystemControlTool"),
    # Notifications
    "NotificationTool":         (".notifications",        "NotificationTool"),
    # Tool Builder / Marketplace
    "ToolBuilderTool":          (".tool_builder",         "ToolBuilderTool"),
    "ToolUsageTracker":         (".tool_builder",         "ToolUsageTracker"),
    "get_usage_tracker":        (".tool_builder",         "get_usage_tracker"),
    "MarketplaceTool":          (".marketplace",          "MarketplaceTool"),
    # Git
    "GitTool":                  (".git_tool",             "GitTool"),
    # Tool Contract
    "ToolResult":               (".tool_contract",        "ToolResult"),
    "ToolSpec":                 (".tool_contract",        "ToolSpec"),
    "ToolRegistry":             (".tool_contract",        "ToolRegistry"),
    "ToolSafety":               (".tool_contract",        "ToolSafety"),
    "LatencyTier":              (".tool_contract",        "LatencyTier"),
    "get_tool_registry":        (".tool_contract",        "get_tool_registry"),
    # Voice Manager
    "VoiceManager":             (".voice_manager",        "VoiceManager"),
    "get_voice_manager":        (".voice_manager",        "get_voice_manager"),
    # EvoEmo
    "EvoEmoTool":               (".evoemo",               "EvoEmoTool"),
    "evoemo":                   (".evoemo",               "evoemo"),
    "analyze_emotion":          (".evoemo",               "analyze_emotion"),
    "get_current_mood":         (".evoemo",               "get_current_mood"),
    "get_mood_emoji":           (".evoemo",               "get_mood_emoji"),
    "get_tone_modifier":        (".evoemo_prompts",       "get_tone_modifier"),
    "get_response_style":       (".evoemo_prompts",       "get_response_style"),
    "build_adaptive_system_prompt": (".evoemo_prompts",   "build_adaptive_system_prompt"),
    # Inner Monologue
    "InnerMonologueTool":       (".inner_monologue",      "InnerMonologueTool"),
    "get_monologue":            (".inner_monologue",      "get_monologue"),
    "THOUGHT_TYPES":            (".inner_monologue",      "THOUGHT_TYPES"),
    "THOUGHT_ICONS":            (".inner_monologue",      "THOUGHT_ICONS"),
    # Knowledge Graph
    "KnowledgeGraphTool":       (".knowledge_graph",      "KnowledgeGraphTool"),
    "get_knowledge_graph":      (".knowledge_graph",      "get_knowledge_graph"),
    "seed_initial_knowledge":   (".knowledge_graph",      "seed_initial_knowledge"),
    "Node":                     (".knowledge_graph",      "Node"),
    "Edge":                     (".knowledge_graph",      "Edge"),
    "NODE_TYPES":               (".knowledge_graph",      "NODE_TYPES"),
    "EDGE_TYPES":               (".knowledge_graph",      "EDGE_TYPES"),
    # NeuroDream
    "NeuroDreamEngine":         (".neurodream",           "NeuroDreamEngine"),
    "SleepPhase":               (".neurodream",           "SleepPhase"),
    "DreamTrigger":             (".neurodream",           "DreamTrigger"),
    "DreamInsight":             (".neurodream",           "DreamInsight"),
    "SleepSession":             (".neurodream",           "SleepSession"),
    "ConsolidatedPattern":      (".neurodream",           "ConsolidatedPattern"),
    "get_neurodream":           (".neurodream",           "get_neurodream"),
    "create_neurodream":        (".neurodream",           "create_neurodream"),
    # MCTS Reasoning Tree
    "MCTSReasoning":            (".mcts_reasoning",       "MCTSReasoning"),
    "MCTSConfig":               (".mcts_reasoning",       "MCTSConfig"),
    "MCTSResult":               (".mcts_reasoning",       "MCTSResult"),
    "MCTSNode":                 (".mcts_reasoning",       "MCTSNode"),
    "ThoughtType":              (".mcts_reasoning",       "ThoughtType"),
    "NodeState":                (".mcts_reasoning",       "NodeState"),
    "mcts_reason":              (".mcts_reasoning",       "mcts_reason"),
    "ReasoningTreeTool":        (".reasoning_tree_tool",  "ReasoningTreeTool"),
    "deep_reason":              (".reasoning_tree_tool",  "deep_reason"),
    # Calendar
    "CalendarTool":             (".calendar_tool",        "CalendarTool"),
    # Code Search & Edit
    "CodeSearchTool":           (".code_search",          "CodeSearchTool"),
    "CodeEditTool":             (".code_edit",            "CodeEditTool"),
    # Codebase Index
    "CodebaseIndex":            (".codebase_index",       "CodebaseIndex"),
    # Shell Executor
    "ShellExecutorTool":        (".shell_executor",       "ShellExecutorTool"),
    # Sandbox Executor (from parent package)
    "SandboxExecutor":          ("..sandbox",             "SandboxExecutor"),
    "ExecutionResult":          ("..sandbox",             "ExecutionResult"),
    # Screen Reader
    "ScreenReaderTool":         (".screen_reader",        "ScreenReaderTool"),
    # Email
    "EmailTool":                (".email_tool",           "EmailTool"),
    # Spaced Repetition
    "SpacedRepetitionTool":     (".spaced_repetition",    "SpacedRepetitionTool"),
    # Task Manager
    "TaskManagerTool":          (".task_manager",         "TaskManagerTool"),
    # API Tester
    "APITesterTool":            (".api_tester",           "APITesterTool"),
    # Database
    "DatabaseTool":             (".database_tool",        "DatabaseTool"),
    # Audio Transcriber
    "AudioTranscriberTool":     (".audio_transcriber",    "AudioTranscriberTool"),
    # Research
    "ResearchTool":             (".research_tool",        "ResearchTool"),
    "DeepResearchTool":         (".deep_research",        "DeepResearchTool"),
    # Tier 1 Quick Wins
    "ObsidianTool":             (".obsidian_tool",        "ObsidianTool"),
    "GitHubTool":               (".github_tool",          "GitHubTool"),
    "LogAnalystTool":           (".log_analyst",          "LogAnalystTool"),
    "DocumentGeneratorTool":    (".document_generator",   "DocumentGeneratorTool"),
    # Tier 2 High-Impact
    "WindowsControlTool":       (".windows_control",      "WindowsControlTool"),
    "TaskSchedulerTool":        (".task_scheduler",       "TaskSchedulerTool"),
    # Tier 3 Moonshot
    "PredictiveTaskTool":       (".predictive_tasks",     "PredictiveTaskTool"),
    "MeetingIntelTool":         (".meeting_intel",        "MeetingIntelTool"),
    "VoiceSynthTool":           (".voice_synth",          "VoiceSynthTool"),
    "LifeLoggerTool":           (".life_logger",          "LifeLoggerTool"),
    # Deploy
    "DeployTool":               (".deploy_tool",          "DeployTool"),
    "get_deploy_tool":          (".deploy_tool",          "get_deploy_tool"),
    # Scaffold
    "ScaffoldTool":             (".scaffold",             "ScaffoldTool"),
    # Extension Feed
    "ExtensionFeedTool":        (".extension_feed",       "ExtensionFeedTool"),
    "get_feed_tool":            (".extension_feed",       "get_feed_tool"),
    # Visual Feedback Loop
    "VisualFeedbackLoop":       (".visual_feedback",      "VisualFeedbackLoop"),
    "get_visual_feedback":      (".visual_feedback",      "get_visual_feedback"),
    # Component Registry
    "ComponentRegistryTool":    (".component_registry",   "ComponentRegistryTool"),
}

# Cache for already-resolved symbols
_RESOLVED: dict[str, object] = {}


def __getattr__(name: str):
    """Lazy-import a tool symbol on first access."""
    # Return cached value if we already resolved it
    if name in _RESOLVED:
        return _RESOLVED[name]

    spec = _LAZY_IMPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'aura.tools' has no attribute {name!r}")

    rel_module, attr = spec

    try:
        mod = _importlib.import_module(rel_module, package="aura.tools")
        value = getattr(mod, attr)
        _RESOLVED[name] = value
        return value
    except Exception as exc:
        msg = f"[tools/__init__] Failed to import {name} from {rel_module}: {exc}"
        _log.warning(msg)
        _TOOLS_IMPORT_ERRORS.append(msg)
        _RESOLVED[name] = None
        return None


# ---------------------------------------------------------------------------
#  __all__ — list all exportable symbols (resolved lazily)
# ---------------------------------------------------------------------------
_ALL_SYMBOLS = list(_LAZY_IMPORTS.keys())

__all__ = _ALL_SYMBOLS
