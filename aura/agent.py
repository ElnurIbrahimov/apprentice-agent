"""Main agent implementation with ReAct loop (single LLM call per step)."""

import json
import os
import re
import time
import logging
import threading
import concurrent.futures
import ast
from collections import deque

# Shared executor for tool calls, memory queries, and observation context.
# Avoids creating+destroying a ThreadPoolExecutor on every message/tool call.
_AGENT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="agent_shared"
)
import atexit as _atexit
_atexit.register(_AGENT_EXECUTOR.shutdown, wait=False)

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, Callable, List, Dict

logger = logging.getLogger(__name__)


# Thinking system integration — safe import
def _record_thought(thought_type: str, content: str, intensity: float = 0.6, source: str = "agent"):
    """Record a real thought event. Safe to call even if thinking system isn't ready."""
    try:
        from api.routes.thinking import record_thought
        record_thought(thought_type, content, intensity, source)
    except Exception as e:
        logger.debug(f"[Agent] non-critical: {e}")
# ============================================================================
#                    SECURITY: Safe Custom Tool Validator
# ============================================================================

# Allowed imports for custom tools
ALLOWED_TOOL_IMPORTS = {
    "typing", "dataclasses", "json", "re", "datetime",
    "pathlib", "collections", "enum", "abc", "math",
    "itertools", "functools", "operator", "string",
    "urllib.parse",  # urllib.parse is safe (URL encoding/decoding)
}

# Forbidden patterns that indicate potentially malicious code
FORBIDDEN_PATTERNS = [
    "os.system", "subprocess", "eval(", "exec(", "__import__",
    "shutil.rmtree", "shutil.move", "socket", "requests.get",
    "urllib.request", "urllib.urlopen", "importlib", "ctypes", "pickle", "marshal",
    "compile(", "globals(", "locals(", "vars(",
    "__builtins__", "__code__", "__class__",
]

# Maximum history size to prevent memory bloat
MAX_HISTORY_SIZE = 100


def validate_custom_tool_code(code: str, tool_path: str) -> Tuple[bool, str]:
    """
    Validate custom tool code before dynamic import.

    SECURITY: Prevents arbitrary code execution via malicious custom tools.
    Checks:
    - No forbidden imports (os.system, subprocess, etc.)
    - No forbidden patterns (eval, exec, __import__)
    - Has required Tool class with execute method
    - Valid Python syntax

    Args:
        code: The tool source code
        tool_path: Path for error messages

    Returns:
        (is_valid, error_message_or_ok)
    """
    # 1. Check for forbidden patterns (fast string check first)
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            return False, f"Forbidden pattern '{pattern}' found in {tool_path}"

    # 2. Parse as AST to validate structure
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in {tool_path}: {e}"

    # 3. Check imports and dangerous AST patterns
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_base = alias.name.split('.')[0]
                if module_base not in ALLOWED_TOOL_IMPORTS:
                    return False, f"Forbidden import '{alias.name}' in {tool_path}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_base = node.module.split('.')[0]
                if module_base not in ALLOWED_TOOL_IMPORTS:
                    return False, f"Forbidden import 'from {node.module}' in {tool_path}"

        # Block dynamic import calls: importlib.import_module(), __import__(), etc.
        elif isinstance(node, ast.Call):
            func = node.func
            # Check for direct calls: __import__("os"), eval("code"), exec("code")
            if isinstance(func, ast.Name) and func.id in ("__import__", "eval", "exec", "compile", "getattr", "delattr"):
                return False, f"Forbidden call '{func.id}()' in {tool_path}"
            # Check for attribute calls: importlib.import_module(), builtins.__import__()
            if isinstance(func, ast.Attribute) and func.attr in ("import_module", "__import__", "system", "popen", "call", "run", "Popen"):
                return False, f"Forbidden call '*.{func.attr}()' in {tool_path}"

        # Block access to dunder attributes that enable sandbox escape
        elif isinstance(node, ast.Attribute):
            if node.attr in ("__subclasses__", "__bases__", "__mro__", "__globals__", "__code__", "__builtins__"):
                return False, f"Forbidden attribute access '.{node.attr}' in {tool_path}"

    # 4. Check for required class structure OR module-level execute() function
    has_tool_class = False
    has_execute = False
    has_module_execute = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Look for Tool class or any class ending with 'Tool'
            if node.name == "Tool" or node.name.endswith("Tool"):
                has_tool_class = True
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in ("execute", "run", "__call__"):
                            has_execute = True
        elif isinstance(node, ast.FunctionDef) and node.name == "execute":
            # Synthesized tools may use a module-level execute() function
            has_module_execute = True

    if has_tool_class and has_execute:
        return True, "Valid"

    if has_module_execute:
        return True, "Valid (module-level execute)"

    if not has_tool_class:
        return False, f"No Tool class found in {tool_path}"

    return False, f"Tool class missing execute/run method in {tool_path}"

# Timeout constants
AGENT_TIMEOUT = 120  # Overall agent loop timeout (2 minutes)
TOOL_TIMEOUT = 30    # Timeout for tool execution

from .brain import OllamaBrain, TaskType
from .identity import load_identity, get_identity_prompt, detect_name_change, detect_personality_change, update_name, update_personality
from .memory.unified_memory import get_unified_memory
from .metacognition import MetacognitionLogger
from .config import Config
from .tools import FileSystemTool, WebSearchTool, CodeExecutorTool, ScreenshotTool, VisionTool, PDFReaderTool, ClipboardTool, ArxivSearchTool, BrowserTool, SystemControlTool, NotificationTool, ToolBuilderTool, MarketplaceTool, GitTool, EvoEmoTool, get_tone_modifier, get_monologue, KnowledgeGraphTool, get_knowledge_graph, NeuroDreamEngine, SleepPhase, CalendarTool, SpacedRepetitionTool, TaskManagerTool, APITesterTool, DatabaseTool, AudioTranscriberTool, ResearchTool, BraveSearchTool, TavilyTool, FirecrawlTool, ObsidianTool, GitHubTool, LogAnalystTool, DocumentGeneratorTool, WindowsControlTool, TaskSchedulerTool, PredictiveTaskTool, MeetingIntelTool, VoiceSynthTool, LifeLoggerTool, CodeSearchTool, CodeEditTool
try:
    from .tools.crypto_price import CryptoPriceTool
except Exception:
    CryptoPriceTool = None

from .memory_retriever import MemoryRetriever
try:
    from .context_engine import AlwaysOnContextEngine
    CONTEXT_ENGINE_AVAILABLE = True
except ImportError:
    CONTEXT_ENGINE_AVAILABLE = False
    AlwaysOnContextEngine = None

# Code Agent Mode — LLM writes Python code as actions (5.1)
try:
    from aura.core.code_agent import CodeAgentMode, should_use_code_agent, CODE_AGENT_SYSTEM_PROMPT
    CODE_AGENT_AVAILABLE = True
except ImportError:
    CODE_AGENT_AVAILABLE = False
    CodeAgentMode = None
    should_use_code_agent = None

# Adaptive Planner — skip planning for simple tasks, re-plan for complex ones (5.4)
try:
    from aura.core.adaptive_planner import AdaptivePlanner
    ADAPTIVE_PLANNER_AVAILABLE = True
except ImportError:
    ADAPTIVE_PLANNER_AVAILABLE = False
    AdaptivePlanner = None

# AURA Fast Path - Instant responses for simple queries
try:
    from aura.fast_path import FastPathHandler
    FAST_PATH_AVAILABLE = True
except ImportError:
    FAST_PATH_AVAILABLE = False
    FastPathHandler = None

# DreamMode - Memory consolidation and pattern analysis
try:
    from .dream import DreamMode
    DREAM_MODE_AVAILABLE = True
except ImportError:
    DreamMode = None
    DREAM_MODE_AVAILABLE = False

# Thinker — MIRROR dual-process async background reasoning (roadmap 3.6)
try:
    from .thinker import get_thinker, ThinkerEngine
    THINKER_AVAILABLE = True
except ImportError:
    THINKER_AVAILABLE = False
    get_thinker = None
    ThinkerEngine = None

# Strategy Bandit - Adaptive reasoning strategy selection
try:
    from aura.consciousness.strategy_bandit import (
        get_strategy_bandit,
        ReasoningStrategy,
    )
    STRATEGY_BANDIT_AVAILABLE = True
except ImportError:
    STRATEGY_BANDIT_AVAILABLE = False
    ReasoningStrategy = None
    get_strategy_bandit = None

# Reasoning Template Library - Learn reusable reasoning patterns
try:
    from aura.consciousness.reasoning_templates import (
        get_template_library,
        build_trace_from_mcts,
    )
    TEMPLATE_LIBRARY_AVAILABLE = True
except ImportError:
    TEMPLATE_LIBRARY_AVAILABLE = False

# Prompt Evolution Engine - Self-modifying system prompts
try:
    from aura.consciousness.prompt_evolution import get_prompt_evolution_engine
    PROMPT_EVOLUTION_AVAILABLE = True
except ImportError:
    PROMPT_EVOLUTION_AVAILABLE = False

# Knowledge Graph Brain - Structured Long-Term Memory
try:
    from aura_knowledge_graph import (
        AURAKnowledgeGraph,
        TitansKGBridge,
        BridgeConfig,
        KGQueryEngine,
        QueryMode,
        KUZU_AVAILABLE
    )
    KG_BRAIN_AVAILABLE = KUZU_AVAILABLE
except ImportError:
    KG_BRAIN_AVAILABLE = False
    AURAKnowledgeGraph = None
    TitansKGBridge = None
    BridgeConfig = None
    KGQueryEngine = None
    QueryMode = None

# Episodic Time-Travel Memory - Autobiographical Memory System
try:
    from aura_episodic_memory import (
        EpisodicMemoryStore,
        Episode,
        EpisodeType,
        EpisodeQuery,
        TemporalContext,
        TitansEpisodicBridge,
        TitansEpisodicConfig,
        TimelineEngine,
        MemoryConsolidator,
        ConsolidationConfig,
        QDRANT_AVAILABLE
    )
    EPISODIC_MEMORY_AVAILABLE = QDRANT_AVAILABLE
except ImportError:
    EPISODIC_MEMORY_AVAILABLE = False
    EpisodicMemoryStore = None
    Episode = None
    EpisodeType = None
    EpisodeQuery = None
    TemporalContext = None
    TitansEpisodicBridge = None
    TitansEpisodicConfig = None
    TimelineEngine = None
    MemoryConsolidator = None
    ConsolidationConfig = None

# Skill Library - Procedural Knowledge Storage
try:
    from aura_skill_library import (
        SkillLibrary,
        Skill,
        SkillCategory,
        SkillStore,
        SkillLearner,
        SkillExecutor,
        TitansSkillBridge,
        EMBEDDINGS_AVAILABLE
    )
    SKILL_LIBRARY_AVAILABLE = True
except ImportError:
    SKILL_LIBRARY_AVAILABLE = False
    SkillLibrary = None
    Skill = None
    SkillCategory = None
    SkillStore = None
    SkillLearner = None
    SkillExecutor = None
    TitansSkillBridge = None
    EMBEDDINGS_AVAILABLE = False

# Predictive Life Modeling - World Simulator for Personal Decisions
try:
    from aura_life_modeling import (
        LifeState,
        LifeDomain,
        Scenario,
        ScenarioTemplates,
        DecisionType,
        SimulationConfig,
        run_monte_carlo,
        ReportGenerator,
        LifeModelingTools,
        MESA_AVAILABLE
    )
    LIFE_MODELING_AVAILABLE = True
except ImportError:
    LIFE_MODELING_AVAILABLE = False
    LifeState = None
    LifeDomain = None
    Scenario = None
    ScenarioTemplates = None
    DecisionType = None
    SimulationConfig = None
    run_monte_carlo = None
    ReportGenerator = None
    LifeModelingTools = None
    MESA_AVAILABLE = False


_TOOL_KEYWORDS = frozenset([
    # --- Tool 0a: code_search (grep/glob/definitions) ---
    'grep', 'search code', 'find in code', 'search in files', 'code search',
    'find definition', 'find class', 'find function', 'find method',
    'find references', 'where is', 'where does', 'which file',
    'project structure', 'show structure', 'codebase', 'repo map',
    'detect project', 'project type', 'what stack',
    'glob', 'find files', 'search files', 'file pattern',

    # --- Tool 0b: code_edit (surgical edits) ---
    'edit file', 'edit code', 'modify file', 'change code', 'update code',
    'replace in file', 'find and replace', 'search replace',
    'refactor', 'rename', 'fix bug', 'fix error', 'patch',
    'add import', 'add function', 'add class', 'add method',
    'remove line', 'delete line', 'insert line',
    'rollback edit', 'undo edit',

    # --- Tool 1: filesystem ---
    'list files', 'show files', 'what files', 'read file', 'write file',
    'open file', 'save file', 'delete file', 'create file', 'file contents',
    'list directory', 'show directory', 'folder contents', 'dir contents',
    'find file', 'search file', 'file system', 'filesystem',

    # --- Tool 2: web_search ---
    'search', 'google', 'look up', 'lookup', 'find online', 'search online',
    'search the web', 'web search', 'search for', 'find out',
    'stock price',
    'weather', 'current weather', 'weather in', 'forecast',
    'news', 'latest news', 'news about', 'headlines',
    'who is', 'what is the current', 'how much does',

    # --- Tool 2b: crypto_price (real-time crypto prices) ---
    'bitcoin price', 'btc price', 'ethereum price', 'eth price',
    'crypto price', 'cryptocurrency price', 'price of bitcoin',
    'price of ethereum', 'price of btc', 'price of eth', 'price of crypto',
    'how much is bitcoin', 'how much is ethereum', 'how much is btc',
    'current bitcoin', 'current ethereum', 'current btc', 'current eth',
    'solana price', 'sol price', 'dogecoin price', 'doge price',
    'cardano price', 'ada price', 'xrp price', 'ripple price',

    # --- Tool 3: code_executor ---
    'run code', 'execute code', 'run python', 'execute python', 'run this',
    'calculate', 'compute', 'factorial', 'fibonacci', 'prime number',
    'write code', 'code to', 'python code', 'script', 'program',

    # --- Tool 4: screenshot ---
    'screenshot', 'take screenshot', 'capture screen', 'capture my screen',
    'screen capture', 'grab screen', 'print screen', 'snapshot', 'screen shot',
    "what's on my screen", 'what is on my screen', 'show my screen',

    # --- Tool 5: vision ---
    'analyze image', 'analyze this image', 'analyze the image',
    'describe image', 'describe this image', 'describe the image',
    'look at image', 'look at this image', 'look at the image',
    "what's in this image", 'what is in this image', 'read image',
    'image analysis', 'picture analysis', 'photo analysis',
    'ocr', 'read text from image', 'extract text from image',

    # --- Tool 6: pdf_reader ---
    'read pdf', 'open pdf', 'pdf file', 'extract pdf', 'pdf contents',
    "what's in this pdf", 'summarize pdf', 'summarize the pdf',
    'search pdf', 'pdf document',

    # --- Tool 7: browser ---
    'browse', 'open website', 'go to website', 'visit website', 'open url',
    'go to url', 'visit url', 'navigate to', 'open page', 'web page',
    'click on', 'click the', 'scroll', 'browser',

    # --- Tool 8: git ---
    'git', 'commit', 'git commit', 'git push', 'git pull', 'git status',
    'git log', 'git diff', 'git stash', 'git branch', 'clone repo',
    'repository', 'repo', 'staged files', 'unstaged', 'untracked',
    'what branch', 'which branch', 'current branch', 'show commits',
    'recent commits', 'show changes', 'list branches',

    # --- Tool 9: arxiv_search ---
    'arxiv', 'research paper', 'academic paper', 'find papers',
    'search papers', 'download paper', 'summarize paper', 'compare papers',
    'scientific paper', 'journal article', 'academic research',

    # --- Tool 10: system_control ---
    'volume', 'set volume', 'get volume', 'brightness', 'set brightness',
    'system info', 'cpu usage', 'ram usage', 'memory usage', 'gpu usage',
    'disk usage', 'disk space', 'open app', 'launch app', 'start app',
    'open notepad', 'open calculator', 'open browser', 'open chrome',
    'open firefox', 'open vscode', 'open terminal', 'lock screen',

    # --- Tool 11: clipboard ---
    'clipboard', 'copy to clipboard', 'paste from clipboard', 'read clipboard',
    'write clipboard', 'clipboard contents', 'what is in clipboard',
    "what's in my clipboard", 'copy this', 'paste this',

    # --- Tool 12: notifications ---
    'remind me', 'reminder', 'set reminder', 'create reminder',
    'notification', 'notify me', 'alert me', 'schedule',
    'in 5 minutes', 'in 10 minutes', 'in 30 minutes', 'in an hour',
    'every day', 'every morning', 'every evening', 'daily at', 'weekly',
    'set alarm', 'timer',

    # --- Tool 13: knowledge_graph / KG Brain ---
    'remember this', 'store this', 'save this fact', 'add to knowledge',
    'what do you know about', 'recall', 'knowledge graph',
    'kg brain', 'kg stats', 'kg query', 'extract entities',
    'learn that', 'remember that', 'consolidate memory', 'graph stats',

    # --- Tool 14: tool_builder ---
    'create tool', 'build tool', 'make tool', 'new tool', 'custom tool',
    'generate tool', 'tool builder', 'design tool',

    # --- Tool 15: marketplace ---
    'marketplace', 'plugin', 'download tool', 'install plugin',
    'browse plugins', 'search plugins', 'uninstall plugin',
    'my plugins', 'installed plugins', 'share tool', 'publish tool',

    # --- Tool 16: regex_builder ---
    'regex', 'regular expression', 'build regex', 'test regex',
    'regex pattern', 'match pattern', 'validate regex', 'explain regex',

    # --- Cognitive System 23: NeuroDream (memory consolidation) ---
    'neurodream', 'go to sleep', 'sleep now', 'dream status',
    'dream journal', 'show dreams', 'sleep insights', 'dream insights',
    'sleep patterns', 'memory consolidation',

    # --- Cognitive System 24: EvoEmo (emotional) ---
    'evoemo', 'my mood', 'how am i feeling', 'mood history',
    'emotional state', 'analyze emotion', 'detect emotion',

    # --- Cognitive System 25: InnerMonologue ---
    'inner monologue', 'show thoughts', 'your thoughts', 'think aloud',
    'reasoning chain', 'what were you thinking', 'export thoughts',

    # --- Tool: deep_research ---
    'deep research', 'research thoroughly', 'thorough research',
    'in-depth research', 'comprehensive research', 'research topic',
    'investigate thoroughly', 'deep dive',

    # --- Tool: image_gen ---
    'generate image', 'create image', 'make image', 'draw image',
    'image generation', 'create picture', 'generate picture',


    # --- Tool: voice/tts ---
    'text to speech', 'speak this', 'say this', 'read aloud',
    'tts', 'sesame tts',
])

# Pre-compiled combined regex for _TOOL_KEYWORDS (avoids re-compiling ~150 patterns per message)
_TOOL_KEYWORDS_RE = re.compile(r'\b(?:' + '|'.join(re.escape(kw) for kw in _TOOL_KEYWORDS) + r')\b')


class AgentPhase(Enum):
    """Phases of the agent loop.

    The main loop uses REACT (single step).
    """
    REACT = "react"       # Single ReAct step (thought + action + deterministic eval)
    REMEMBER = "remember"


@dataclass
class AgentState:
    """Current state of the agent."""
    goal: str = ""
    phase: AgentPhase = AgentPhase.REACT
    observations: str = ""
    current_plan: str = ""
    last_action: Optional[dict] = None
    last_result: Optional[dict] = None
    evaluation: Optional[dict] = None
    iteration: int = 0
    completed: bool = False
    _history: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_SIZE))
    gathered_content: str = ""  # Store content gathered from searches for summarization

    @property
    def history(self) -> list:
        """Get history with size limit enforcement."""
        return list(self._history)

    @history.setter
    def history(self, value: list):
        """Set history with automatic truncation to MAX_HISTORY_SIZE."""
        self._history = deque(value, maxlen=MAX_HISTORY_SIZE)

    def add_to_history(self, item: dict):
        """Add an item to history, enforcing size limit."""
        self._history.append(item)


# Actions that require user confirmation before execution
class ApprenticeAgent:
    """An AI agent that learns and acts using a ReAct loop (single LLM call per step).

    The main loop (run()) uses brain.react_step() for combined thought+action,
    with deterministic tool result evaluation (no LLM call for eval).
    """

    def __init__(self, fast_init: bool = False):
        """Initialize the agent.

        Args:
            fast_init: If True, skip slow initialization (warmup, heavy tools).
                      Use for GUI where you want fast startup.
        """
        # Validate and select best available models before creating brain
        try:
            validated = Config.validate_models_on_startup()
            logger.info(f"[MODELS] Validated: {validated}")
        except Exception as e:
            logger.warning(f"[MODELS] Validation failed: {e}")

        # Skip Ollama warmup for fast init
        self.brain = OllamaBrain(warmup=not fast_init)
        self.memory = get_unified_memory()

        # Core lightweight tools (always load)
        # Each instantiation is guarded: if a tool class is None (failed import)
        # or its constructor throws, we skip it instead of crashing the agent.
        self.tools = {}
        _core_tools = [
            ("code_search", CodeSearchTool),
            ("code_edit", CodeEditTool),
            ("filesystem", FileSystemTool),
            ("web_search", WebSearchTool),
            ("brave_search", BraveSearchTool),
            ("tavily_search", TavilyTool),
            ("firecrawl", FirecrawlTool),
            ("crypto_price", CryptoPriceTool),
            ("code_executor", CodeExecutorTool),
            ("clipboard", ClipboardTool),
            ("notifications", NotificationTool),
            ("git", GitTool),
            ("evoemo", EvoEmoTool),
            ("calendar", CalendarTool),
            ("spaced_repetition", SpacedRepetitionTool),
            ("task_manager", TaskManagerTool),
            ("research", ResearchTool),
            ("obsidian", ObsidianTool),
            ("github", GitHubTool),
            ("log_analyst", LogAnalystTool),
            ("document_generator", DocumentGeneratorTool),
            # Tier 2 High-Impact Tools
            ("windows_control", WindowsControlTool),
            ("task_scheduler", TaskSchedulerTool),
            # Tier 3 Moonshot Tools
            ("predictive_tasks", PredictiveTaskTool),
            ("meeting_intel", MeetingIntelTool),
            ("voice_synth", VoiceSynthTool),
            ("life_logger", LifeLoggerTool),
        ]
        for _tool_name, _tool_cls in _core_tools:
            try:
                if _tool_cls is None:
                    logger.warning(f"[TOOLS] {_tool_name} skipped — import failed (class is None)")
                    continue
                self.tools[_tool_name] = _tool_cls()
                logger.debug(f"[TOOLS] {_tool_name} loaded")
            except Exception as _e:
                logger.warning(f"[TOOLS] {_tool_name} skipped — init failed: {_e}")

        # inner_monologue is a factory function, not a class
        try:
            if get_monologue is not None:
                self.tools["inner_monologue"] = get_monologue()
        except Exception as _e:
            logger.warning(f"[TOOLS] inner_monologue skipped — init failed: {_e}")

        # Wire LifeLogger to sibling tools for cross-source sync
        if "life_logger" in self.tools:
            try:
                self.tools["life_logger"].set_tools(self.tools)
            except Exception as _e:
                logger.warning(f"[TOOLS] life_logger.set_tools failed: {_e}")
        logger.info(f"[TOOLS] {len(self.tools)} core tools loaded")

        # Heavier tools - load lazily or skip for fast init
        if not fast_init:
            _heavy_tools = [
                ("screenshot", lambda: ScreenshotTool()),
                ("vision", lambda: VisionTool(brain=self.brain)),
                ("pdf_reader", lambda: PDFReaderTool()),
                ("arxiv_search", lambda: ArxivSearchTool()),
                ("browser", lambda: BrowserTool()),
                ("system_control", lambda: SystemControlTool()),
                ("tool_builder", lambda: ToolBuilderTool()),
                ("marketplace", lambda: MarketplaceTool()),
                ("knowledge_graph", lambda: get_knowledge_graph()),
            ]
            for _tool_name, _tool_factory in _heavy_tools:
                try:
                    _cls_or_fn = _tool_factory  # it's a lambda wrapping the call
                    inst = _cls_or_fn()
                    if inst is not None:
                        self.tools[_tool_name] = inst
                        logger.debug(f"[TOOLS] {_tool_name} loaded (heavy)")
                    else:
                        logger.warning(f"[TOOLS] {_tool_name} skipped — factory returned None")
                except Exception as _e:
                    logger.warning(f"[TOOLS] {_tool_name} skipped — init failed: {_e}")

            # === LOAD ADDITIONAL TOOLS ===
            # deep_research
            try:
                from .tools.deep_research import DeepResearchTool
                self.tools['deep_research'] = DeepResearchTool()
                logger.info("[LOADED] deep_research")
            except Exception as e:
                logger.warning(f"deep_research not loaded: {e}")

            # image_gen
            try:
                from .tools.image_gen import ImageGenerationTool
                self.tools['image_gen'] = ImageGenerationTool()
                logger.info("[LOADED] image_gen")
            except Exception as e:
                logger.warning(f"image_gen not loaded: {e}")


            # voice
            try:
                from .tools.voice import VoiceTool
                self.tools['voice'] = VoiceTool()
                logger.info("[LOADED] voice")
            except Exception as e:
                logger.warning(f"voice not loaded: {e}")

            # local_rag - Local document RAG system
            try:
                from .tools.local_rag import LocalRAGTool
                self.tools['local_rag'] = LocalRAGTool()
                logger.info("[LOADED] local_rag - Index and search local documents")
            except Exception as e:
                logger.warning(f"local_rag not loaded: {e}")

            # amem - A-MEM Zettelkasten-style agentic memory
            try:
                from .tools.amem_tool import get_amem_tool
                self.tools['amem'] = get_amem_tool()
                logger.info("[LOADED] amem - Zettelkasten agentic memory")
            except Exception as e:
                logger.warning(f"[WARNING] amem not loaded: {e}")

            # hybrid_amem - Combined A-MEM + Knowledge Graph memory
            try:
                from .tools.hybrid_amem import get_hybrid_memory
                kg = self.tools.get('knowledge_graph')
                self.tools['hybrid_amem'] = get_hybrid_memory(knowledge_graph=kg)
                logger.info("[LOADED] hybrid_amem - Hybrid A-MEM + KG memory")
            except Exception as e:
                logger.warning(f"[WARNING] hybrid_amem not loaded: {e}")

            # shell_executor
            try:
                from .tools.shell_executor import ShellExecutorTool
                self.tools['shell_executor'] = ShellExecutorTool()
                logger.info("[LOADED] shell_executor")
            except Exception as e:
                logger.warning(f"shell_executor not loaded: {e}")

            # screen_reader
            try:
                from .tools.screen_reader import ScreenReaderTool
                self.tools['screen_reader'] = ScreenReaderTool()
                logger.info("[LOADED] screen_reader")
            except Exception as e:
                logger.warning(f"screen_reader not loaded: {e}")

            # email
            try:
                from .tools.email_tool import EmailTool
                self.tools['email'] = EmailTool()
                logger.info("[LOADED] email")
            except Exception as e:
                logger.warning(f"email not loaded: {e}")

            # api_tester
            try:
                self.tools['api_tester'] = APITesterTool()
                logger.info("[LOADED] api_tester")
            except Exception as e:
                logger.warning(f"api_tester not loaded: {e}")

            # database
            try:
                self.tools['database'] = DatabaseTool()
                logger.info("[LOADED] database")
            except Exception as e:
                logger.warning(f"database not loaded: {e}")

            # audio_transcriber
            try:
                self.tools['audio_transcriber'] = AudioTranscriberTool()
                logger.info("[LOADED] audio_transcriber")
            except Exception as e:
                logger.warning(f"audio_transcriber not loaded: {e}")

            # Auto-load ALL synthesized tools (with security validation)
            try:
                import os
                synth_path = os.path.join(os.path.dirname(__file__), 'tools', 'synthesized')
                if os.path.exists(synth_path):
                    for file in os.listdir(synth_path):
                        if file.endswith('.py') and file != '__init__.py':
                            tool_name = file[:-3]
                            tool_file_path = os.path.join(synth_path, file)
                            try:
                                # SECURITY: Validate tool code before import
                                with open(tool_file_path, 'r', encoding='utf-8') as f:
                                    tool_code = f.read()

                                is_valid, validation_msg = validate_custom_tool_code(tool_code, tool_file_path)
                                if not is_valid:
                                    logger.warning(f"synthesized/{tool_name} BLOCKED: {validation_msg}")
                                    continue

                                module = __import__(f'aura.tools.synthesized.{tool_name}', fromlist=[tool_name])
                                # Try different class name patterns
                                class_name = ''.join(word.title() for word in tool_name.split('_')) + 'Tool'
                                tool_class = getattr(module, class_name, None)
                                if not tool_class:
                                    # Try simpler name
                                    tool_class = getattr(module, f'{tool_name}Tool', None)
                                if tool_class:
                                    self.tools[tool_name] = tool_class()
                                    logger.info(f"[LOADED] synthesized/{tool_name}")
                                elif hasattr(module, 'execute'):
                                    # Synthesized tools with module-level execute()
                                    # Wrap in a simple object so tool dispatch works uniformly
                                    exec_fn = module.execute
                                    wrapper = type(f'{tool_name}_tool', (), {
                                        'execute': staticmethod(exec_fn),
                                        'run': staticmethod(exec_fn),
                                        'name': tool_name,
                                        'description': getattr(module, '__doc__', '') or tool_name,
                                    })()
                                    self.tools[tool_name] = wrapper
                                    logger.info(f"[LOADED] synthesized/{tool_name}")
                            except Exception as e:
                                logger.warning(f"synthesized/{tool_name} not loaded: {e}")
            except Exception as e:
                logger.warning(f"Could not load synthesized tools: {e}")

            # Load custom and marketplace tools dynamically
            try:
                from .tools.custom_loader import load_custom_tools
                _custom_tools = load_custom_tools()
                for _tool_name, _tool_instance in _custom_tools.items():
                    if _tool_name not in self.tools:
                        self.tools[_tool_name] = _tool_instance
                        logger.info(f"[Agent] Registered custom tool: {_tool_name}")
            except Exception as _e:
                logger.warning(f"[Agent] Custom tool loading failed: {_e}")

        # Connect inner monologue to EvoEmo for emotional awareness
        self.monologue = self.tools.get("inner_monologue")
        if self.monologue and "evoemo" in self.tools:
            self.monologue.connect_evoemo(self.tools["evoemo"])
        self.state = AgentState()
        self.max_iterations = 10
        self.metacognition = MetacognitionLogger()
        self.use_fastpath = True  # Enable fast-path by default
        self.identity = load_identity()  # Load agent identity
        self.custom_tool_keywords = {}  # Map of keyword -> tool_name for custom tools
        self._load_custom_tools()  # Load active custom tools
        self._fast_init = fast_init

        self.guardian = None
        self._pending_prediction = None

        # Coherent Loop state — tracks previous exchange for reaction feedback
        self._prev_message = None   # User's previous message
        self._prev_response = None  # AURA's previous response

        # Initialize Thinker — MIRROR dual-process background reasoning (roadmap 3.6)
        self.thinker = None
        if THINKER_AVAILABLE:
            try:
                self.thinker = get_thinker(brain=self.brain)
                logger.info("[LOADED] Thinker — MIRROR dual-process background reasoning")
            except Exception as e:
                logger.warning(f"[Thinker] Init failed: {e}")

        # Initialize NeuroDream (Tool #24) - Sleep/Dream Memory Consolidation
        self.neurodream = NeuroDreamEngine(
            knowledge_graph=self.tools.get("knowledge_graph"),
            hybrid_memory=None,
            evoemo=self.tools.get("evoemo"),
            inner_monologue=self.monologue,
            chromadb=None,
            idle_threshold_minutes=30
        )
        logger.debug("[LOADED] NeuroDream - Sleep/dream memory consolidation")

        # ===== Phase 3 Fix 3B: NeuroDream true idle polling thread =====
        # Polls check_idle_trigger() every 60s so NeuroDream can sleep autonomously
        import threading as _threading
        # time is already imported at module level
        self._neurodream_stop_event = _threading.Event()
        def _neurodream_idle_poll():
            while not self._neurodream_stop_event.wait(timeout=60):
                try:
                    if self.neurodream and self.neurodream.current_phase.value == "awake":
                        self.neurodream.check_idle_trigger()
                except Exception as e:
                    logger.debug(f"[Agent] non-critical: {e}")
        _nd_poll_thread = _threading.Thread(
            target=_neurodream_idle_poll, daemon=True, name="NeuroDream-IdlePoll"
        )
        _nd_poll_thread.start()
        logger.debug("[NeuroDream] Idle polling thread started (60s interval)")

        # Dead modules — stubs to prevent AttributeError
        self.mirrormind = None
        self.mirrormind_enabled = False
        self.introspection = None
        self.theater = None
        self.theater_enabled = False
        self.reflexion = None
        self.reflexion_enabled = False
        self.forge = None
        self.synapseforge_enabled = False
        self.worldsim = None
        self.worldsim_enabled = False

        try:
            self.tools['neurodream'] = self.neurodream
        except Exception as e:
            logger.debug(f"[SKIP] NeuroDream registry: {e}")

        # AURA v3.0 ALIVE — AURAEngine removed; context/humanization via ALMA helpers.
        # self.aura_enabled controls whether _build_aura_context runs.
        self.aura_enabled = getattr(Config, 'AURA_ENABLED', True)

        # Soul — personality/identity config from soul files
        self._soul = None
        self._soul_loader = None
        try:
            from aura.soul.soul_loader import SoulLoader
            self._soul_loader = SoulLoader()
            self._soul = self._soul_loader.load("SOUL_PERSONAL")
            logger.info(f"[LOADED] Soul: {self._soul.get('name', 'AURA')}")
        except Exception as e:
            self._soul = None
            logger.debug(f"[SKIP] Soul: {e}")

        # VisibleThinking — transparent reasoning for ThoughtStream UI
        self._visible_thinking = None
        try:
            from aura.thinking.visible_thinking import VisibleThinking
            self._visible_thinking = VisibleThinking()
            logger.info("[LOADED] VisibleThinking")
        except Exception as e:
            logger.debug(f"[SKIP] VisibleThinking: {e}")

        self._humanizer = None

        # Initialize AURA Fast Path - Instant responses
        if FAST_PATH_AVAILABLE:
            try:
                self.fast_path_handler = FastPathHandler(
                    memory_store=self.memory if hasattr(self, 'memory') else None,
                    emotional_engine=None
                )
                self.fast_path_handler._agent = self
                logger.debug("[LOADED] AURA Fast Path - Instant emotional responses")
            except Exception as e:
                logger.warning(f"[WARNING] Fast Path initialization failed: {e}")
                self.fast_path_handler = None
        else:
            self.fast_path_handler = None

        # Initialize GatewayDaemon — proactive intelligence system
        # get_gateway_daemon() returns the singleton (creates it if first call).
        # The actual async start() is called by api/main.py; here we just ensure
        # the instance exists and wire a notification callback so the agent can
        # receive proactive messages even when running without the API server.
        self.gateway_daemon = None
        try:
            from aura.proactive.gateway_daemon import get_gateway_daemon
            self.gateway_daemon = get_gateway_daemon()

            def _on_proactive_agent(msg):
                """Delivery callback: log proactive messages; API wires WebSocket push separately."""
                try:
                    logger.info(f"[GatewayDaemon] Proactive: {getattr(msg, 'content', str(msg))[:120]}")
                except Exception as e:
                    logger.debug(f"[Agent] non-critical: {e}")
            # Only set fallback callback if none wired yet (api/main.py sets a richer one)
            if getattr(self.gateway_daemon, '_notification_callback', None) is None:
                self.gateway_daemon.set_notification_callback(_on_proactive_agent)
            logger.info("[LOADED] GatewayDaemon - Proactive intelligence system (singleton)")
        except Exception as e:
            logger.warning(f"[GatewayDaemon] Initialization failed: {e}")
            self.gateway_daemon = None

        self.proto_agi = None

        # Initialize Tool RAG for dynamic tool selection
        self.tool_rag = None
        try:
            from aura.tools.tool_rag import ToolRAG
            from aura.core.tool_schemas import AGENTIC_TOOLS
            self.tool_rag = ToolRAG()
            self.tool_rag.initialize(self.tools, AGENTIC_TOOLS)
        except Exception as e:
            logger.debug(f"[ToolRAG] Init failed (will use fallback): {e}")

        # Initialize Adaptive Planner (Roadmap 5.4)
        self.adaptive_planner = None
        if ADAPTIVE_PLANNER_AVAILABLE:
            try:
                self.adaptive_planner = AdaptivePlanner(
                    brain=self.brain, planning_interval=3
                )
                logger.debug("[LOADED] AdaptivePlanner — adaptive planning for complex tasks")
            except Exception as e:
                logger.debug(f"[AdaptivePlanner] Init failed: {e}")

        # Initialize ToolExecutor for ReAct loop (handles dev tools without sandbox)
        self._tool_executor = None
        try:
            from aura.core.agentic_loop import ToolExecutor
            _project_root = os.getcwd()
            try:
                from aura.core.context import find_project_root
                _project_root = find_project_root() or _project_root
            except Exception:
                pass
            self._tool_executor = ToolExecutor(project_root=_project_root)
            logger.debug(f"[ToolExecutor] Initialized at {_project_root}")
        except Exception as e:
            logger.debug(f"[ToolExecutor] Init failed: {e}")

        # CLI permission confirmation callback (set by main.py for interactive mode)
        self._cli_confirm_callback: Optional[Callable] = None
        self._approved_patterns: set = set()

        # Initialize Knowledge Graph Brain - Structured Long-Term Memory
        # KG Brain is lightweight and can initialize even with fast_init
        # The bridge (which needs LLM) is skipped for fast_init
        self.kg_brain = None
        self.kg_bridge = None
        self.kg_query_engine = None
        self.kg_brain_enabled = getattr(Config, 'KG_BRAIN_ENABLED', True)

        if KG_BRAIN_AVAILABLE and self.kg_brain_enabled:
            try:
                # Initialize Knowledge Graph database (lightweight, always init)
                kg_path = Path(__file__).parent.parent / "aura_data" / "knowledge_graph_brain"
                self.kg_brain = AURAKnowledgeGraph(str(kg_path))

                # Initialize Query Engine (lightweight, always init)
                self.kg_query_engine = KGQueryEngine(self.kg_brain)

                # Initialize Titans-KG Bridge for automatic extraction (needs LLM)
                # Skip for fast_init since it requires brain.think
                if not fast_init:
                    self.kg_bridge = TitansKGBridge(
                        knowledge_graph=self.kg_brain,
                        llm_func=self.brain.think,
                        config=BridgeConfig(
                            surprise_threshold=0.5,
                            batch_size=3,
                            auto_extract=True,
                            create_co_occurrence=True
                        )
                    )

                # Get statistics
                stats = self.kg_brain.get_statistics()
                bridge_status = "with bridge" if self.kg_bridge else "query-only"
                logger.debug(f"[LOADED] Knowledge Graph Brain - {stats['total_entities']} entities, {stats['total_relationships']} relationships ({bridge_status})")

                # Register KG Brain with UnifiedMemory so it's included in unified context queries
                if self.kg_bridge is not None:
                    try:
                        get_unified_memory().set_kg_brain(self.kg_bridge)
                    except Exception as e:
                        logger.debug(f"[Agent] non-critical: {e}")
            except Exception as e:
                logger.warning(f"[WARNING] Knowledge Graph Brain initialization failed: {e}")
                self.kg_brain = None
                self.kg_bridge = None
                self.kg_query_engine = None
        elif not KG_BRAIN_AVAILABLE:
            logger.debug("[INFO] Knowledge Graph Brain not available (install kuzu: pip install kuzu)")

        # Thread-safety locks (instance-level, not class-level)
        self._temporal_lock = threading.Lock()
        self._kg_queue_lock = threading.Lock()

        # Initialize Episodic Time-Travel Memory - Autobiographical Memory
        # Like KG Brain, this is lightweight and can initialize even with fast_init
        self.episodic_memory = None
        self.episodic_bridge = None
        self.episodic_timeline = None
        self.memory_retriever = MemoryRetriever()
        # Always-On Context Engine
        self.context_engine = None
        if CONTEXT_ENGINE_AVAILABLE:
            try:
                self.context_engine = AlwaysOnContextEngine(self)
                logger.debug("[ACE] Context engine initialized")
            except Exception as _e:
                logger.warning(f"[ACE] Failed to initialize: {_e}")
        self.parliament = None
        self.episodic_consolidator = None
        self.episodic_memory_enabled = getattr(Config, 'EPISODIC_MEMORY_ENABLED', True)

        if EPISODIC_MEMORY_AVAILABLE and self.episodic_memory_enabled:
            try:
                # Initialize Qdrant-based episodic memory store (lightweight)
                episodic_path = Path(__file__).parent.parent / "aura_data" / "episodic_memory"
                self.episodic_memory = EpisodicMemoryStore(str(episodic_path))

                # Initialize Timeline Engine (lightweight, always init)
                self.episodic_timeline = TimelineEngine(self.episodic_memory)

                # Initialize Titans-Episodic Bridge (needs LLM for significant episodes)
                # Skip full bridge for fast_init since it may record to memory
                if not fast_init:
                    self.episodic_bridge = TitansEpisodicBridge(
                        memory_store=self.episodic_memory,
                        config=TitansEpisodicConfig(
                            surprise_threshold=0.5,
                            turns_per_episode=3,
                            max_episodes_per_session=100
                        )
                    )

                    # Initialize Consolidator (for memory maintenance)
                    self.episodic_consolidator = MemoryConsolidator(
                        memory_store=self.episodic_memory,
                        config=ConsolidationConfig(
                            decay_rate=0.03,
                            gc_age_days=90
                        ),
                        llm_func=self.brain.think if not fast_init else None
                    )

                # Get statistics
                stats = self.episodic_memory.get_statistics()
                bridge_status = "with bridge" if self.episodic_bridge else "query-only"
                logger.debug(f"[LOADED] Episodic Memory - {stats['total_episodes']} episodes ({bridge_status})")
            except Exception as e:
                logger.warning(f"[WARNING] Episodic Memory initialization failed: {e}")
                self.episodic_memory = None
                self.episodic_bridge = None
                self.episodic_timeline = None
                self.episodic_consolidator = None
        elif not EPISODIC_MEMORY_AVAILABLE:
            logger.debug("[INFO] Episodic Memory not available (install qdrant-client: pip install qdrant-client)")

        # DreamMode — Memory consolidation and pattern analysis
        self.dream_mode = None
        if DREAM_MODE_AVAILABLE:
            try:
                self.dream_mode = DreamMode()
                logger.debug("[LOADED] DreamMode — Memory consolidation")
            except Exception as e:
                logger.warning(f"[WARNING] DreamMode init failed: {e}")

        # Initialize Skill Library - Procedural Knowledge Storage
        # Stores successful patterns, workflows, and techniques as reusable skills
        self.skill_library = None
        self.skill_bridge = None
        self.skill_library_enabled = getattr(Config, 'SKILL_LIBRARY_ENABLED', True)

        if SKILL_LIBRARY_AVAILABLE and self.skill_library_enabled:
            try:
                # Initialize Skill Library (lightweight, always init)
                skill_path = Path(__file__).parent.parent / "aura_data" / "skill_library"
                self.skill_library = SkillLibrary(
                    storage_path=str(skill_path),
                    llm_func=self.brain.think if not fast_init else None,
                    min_examples_to_learn=3
                )

                # Initialize Titans-Skill Bridge (connects to other memory systems)
                if not fast_init:
                    self.skill_bridge = self.skill_library.connect_bridge(
                        titans_memory=getattr(self, 'titans_memory', None),
                        episodic_memory=self.episodic_memory,
                        kg_brain=self.kg_brain
                    )

                # Get statistics
                stats = self.skill_library.get_stats()
                total_skills = stats['store']['total_skills']
                bridge_status = "with bridge" if self.skill_bridge else "library-only"
                logger.debug(f"[LOADED] Skill Library - {total_skills} skills ({bridge_status})")
            except Exception as e:
                logger.warning(f"[WARNING] Skill Library initialization failed: {e}")
                self.skill_library = None
                self.skill_bridge = None
        elif not SKILL_LIBRARY_AVAILABLE:
            logger.debug("[INFO] Skill Library not available (install sentence-transformers: pip install sentence-transformers)")

        # Initialize Predictive Life Modeling - World Simulator for Personal Decisions
        # NOTE: On-demand tool — accessed via life_update_state(), simulate_decision(),
        # compare_decisions(), what_if_analysis(), etc. Not auto-triggered.
        self.life_modeling = None
        self.life_modeling_enabled = getattr(Config, 'LIFE_MODELING_ENABLED', True)

        if LIFE_MODELING_AVAILABLE and self.life_modeling_enabled:
            try:
                self.life_modeling = LifeModelingTools(
                    knowledge_graph=self.kg_brain,
                    episodic_memory=self.episodic_memory,
                    llm_client=self.brain if not fast_init else None
                )
                logger.debug(f"[LOADED] Life Modeling - World Simulator for decisions (Mesa: {MESA_AVAILABLE})")
            except Exception as e:
                logger.warning(f"[WARNING] Life Modeling initialization failed: {e}")
                self.life_modeling = None
        elif not LIFE_MODELING_AVAILABLE:
            logger.debug("[INFO] Life Modeling not available (install mesa: pip install mesa)")

        # Initialize Hooks / Event System
        self.hooks = None
        try:
            from .hooks import HooksManager
            self.hooks = HooksManager(tools=self.tools)
            self.hooks.start_background(interval=15)
            hook_count = len(self.hooks.list_hooks())
            if hook_count:
                logger.debug(f"[LOADED] Hooks - {hook_count} active hooks")
            else:
                logger.debug("[LOADED] Hooks - event system ready")
        except Exception as e:
            logger.debug(f"[INFO] Hooks system not available: {e}")

        # Initialize Multi-Agent Orchestrator
        self.orchestrator = None
        try:
            from .multi_agent.orchestrator import MultiAgentOrchestrator

            def _orchestrator_llm(system_prompt, user_message):
                return self.brain.think(user_message, system_prompt=system_prompt, use_history=False)

            self.orchestrator = MultiAgentOrchestrator(
                tool_registry=self.tools,
                llm_func=_orchestrator_llm
            )
            specialists = list(self.orchestrator.specialists.keys())
            logger.debug(f"[LOADED] Multi-Agent Orchestrator - {', '.join(specialists)}")
        except Exception as e:
            logger.debug(f"[INFO] Multi-Agent Orchestrator not available: {e}")

    def _make_response(
        self,
        goal: str,
        response: str,
        *,
        completed: bool = True,
        iterations: int = 0,
        fast_path: bool = False,
        history: list | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Build a standardized response dict."""
        result = {
            "goal": goal,
            "completed": completed,
            "iterations": iterations,
            "fast_path": fast_path,
            "response": response,
            "history": history or [],
        }
        if metadata:
            result.update(metadata)
        # Default final_evaluation only if caller didn't provide one via metadata
        if "final_evaluation" not in result:
            result["final_evaluation"] = {
                "success": completed,
                "confidence": 100 if completed else 0,
                "progress": response,
            }
        return result

    def _ensure_tool(self, tool_name: str):
        """Lazily load a tool if not already loaded."""
        if tool_name in self.tools:
            return self.tools[tool_name]

        # Lazy load heavy tools
        if tool_name == "knowledge_graph":
            self.tools["knowledge_graph"] = get_knowledge_graph()
        elif tool_name == "screenshot":
            self.tools["screenshot"] = ScreenshotTool()
        elif tool_name == "vision":
            self.tools["vision"] = VisionTool(brain=self.brain)
        elif tool_name == "pdf_reader":
            self.tools["pdf_reader"] = PDFReaderTool()
        elif tool_name == "arxiv_search":
            self.tools["arxiv_search"] = ArxivSearchTool()
        elif tool_name == "browser":
            self.tools["browser"] = BrowserTool()
        elif tool_name == "system_control":
            self.tools["system_control"] = SystemControlTool()
        elif tool_name == "tool_builder":
            self.tools["tool_builder"] = ToolBuilderTool()
        elif tool_name == "marketplace":
            self.tools["marketplace"] = MarketplaceTool()
        elif tool_name == "shell_executor":
            from .tools.shell_executor import ShellExecutorTool
            self.tools["shell_executor"] = ShellExecutorTool()
        elif tool_name == "screen_reader":
            from .tools.screen_reader import ScreenReaderTool
            self.tools["screen_reader"] = ScreenReaderTool()
        elif tool_name == "email":
            from .tools.email_tool import EmailTool
            self.tools["email"] = EmailTool()
        elif tool_name == "calendar":
            self.tools["calendar"] = CalendarTool()
        elif tool_name == "spaced_repetition":
            self.tools["spaced_repetition"] = SpacedRepetitionTool()
        elif tool_name == "task_manager":
            self.tools["task_manager"] = TaskManagerTool()
        elif tool_name == "api_tester":
            self.tools["api_tester"] = APITesterTool()
        elif tool_name == "database":
            self.tools["database"] = DatabaseTool()
        elif tool_name == "audio_transcriber":
            self.tools["audio_transcriber"] = AudioTranscriberTool()
        elif tool_name == "research":
            self.tools["research"] = ResearchTool()

        return self.tools.get(tool_name)

    def _load_custom_tools(self) -> None:
        """Load active custom tools from registry."""
        import importlib.util

        registry_path = Path(__file__).parent.parent / "data" / "custom_tools.json"
        if not registry_path.exists():
            return

        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)

            for tool_entry in registry.get("tools", []):
                if tool_entry.get("status") != "active":
                    continue

                tool_name = tool_entry["name"]
                # SECURITY: Resolve path and verify it stays within the project tools directory
                tools_base = (Path(__file__).parent / "tools").resolve()
                tool_file = Path(tool_entry.get("file", ""))
                try:
                    tool_file_resolved = tool_file.resolve()
                except Exception:
                    continue
                if not str(tool_file_resolved).startswith(str(tools_base)):
                    logger.warning(f"[SECURITY] Custom tool path outside project directory: {tool_file}")
                    continue
                if not tool_file_resolved.exists():
                    continue
                tool_file = tool_file_resolved

                try:
                    # SECURITY: Validate tool code before dynamic import
                    tool_code = tool_file.read_text()
                    is_valid, validation_msg = validate_custom_tool_code(tool_code, str(tool_file))

                    if not is_valid:
                        logger.debug(f"[SECURITY] Rejected custom tool {tool_name}: {validation_msg}")
                        logger.warning(f"Custom tool {tool_name} failed security validation: {validation_msg}")
                        continue

                    # Dynamic import of validated custom tool
                    spec = importlib.util.spec_from_file_location(
                        tool_name,
                        tool_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Get the tool class
                        class_name = tool_entry.get("class_name")
                        if class_name and hasattr(module, class_name):
                            tool_class = getattr(module, class_name)
                            self.tools[tool_name] = tool_class()
                            logger.debug(f"[LOADED] Custom tool: {tool_name} (validated)")

                            # Load keywords for this tool
                            keywords = tool_entry.get("keywords", [])
                            if not keywords:
                                # Generate default keywords from name and description
                                keywords = self._generate_default_keywords(
                                    tool_name,
                                    tool_entry.get("description", ""),
                                    tool_entry.get("functions", [])
                                )
                            for kw in keywords:
                                self.custom_tool_keywords[kw.lower()] = tool_name
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load custom tool {tool_name}: {e}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"[ERROR] Failed to load custom tools registry: {e}")

    def _generate_default_keywords(self, name: str, description: str, functions: list) -> list[str]:
        """Generate default keywords for a custom tool if not provided.

        Args:
            name: Tool name
            description: Tool description
            functions: List of function names

        Returns:
            List of keywords for tool detection
        """
        keywords = set()

        # Add words from tool name
        for word in name.lower().split('_'):
            if len(word) > 2:
                keywords.add(word)
        keywords.add(name.lower().replace('_', ' '))

        # Add words from description
        stop_words = {'a', 'an', 'the', 'is', 'are', 'to', 'of', 'in', 'for', 'on',
                      'with', 'at', 'by', 'from', 'and', 'or', 'but', 'can', 'will'}
        desc_words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        for word in desc_words:
            if word not in stop_words and len(word) > 2:
                keywords.add(word)

        # Add function names
        for func in functions:
            for word in func.lower().split('_'):
                if len(word) > 2:
                    keywords.add(word)

        # Add common variations
        if 'bmi' in name.lower():
            keywords.update(['body mass index', 'height and weight'])
        if 'temperature' in name.lower():
            keywords.update(['celsius', 'fahrenheit', 'temp'])

        return list(keywords)

    def _is_simple_query(self, goal: str) -> bool:
        """Check if the goal is a simple conversational query.

        STRICT: Only pure greetings and identity questions go to fast-path.
        Everything else goes through the full agent loop with tools.
        """
        goal_lower = goal.lower().strip()
        words = goal_lower.split()

        # NEVER fast-path if message contains a file path or directory
        if any(c in goal for c in ('/', '\\', ':\\', '.py', '.js', '.ts', '.md')):
            return False

        # NEVER fast-path if more than 8 words — likely a real task
        if len(words) > 8:
            return False

        # NEVER fast-path if any tool keyword matches
        if _TOOL_KEYWORDS_RE.search(goal_lower):
            return False
        for kw in self.custom_tool_keywords:
            if kw in goal_lower:
                return False

        # Pure greetings only (1-4 words max)
        greetings = [
            'hello', 'hi', 'hey', 'yo', 'sup',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'thanks', 'thank you', 'thx', 'bye', 'goodbye', 'see you',
            'how are you', "what's up", 'whats up', "how's it going",
        ]
        for greeting in greetings:
            if goal_lower == greeting or goal_lower.rstrip('!?.') == greeting:
                return True

        # Identity questions only
        identity_patterns = [
            'who are you', 'what are you', 'what is your name',
            "what's your name", 'what can you do', 'introduce yourself',
        ]
        for pattern in identity_patterns:
            if goal_lower == pattern or goal_lower.rstrip('?') == pattern:
                return True

        # Conversational phrases that don't need tools (≤8 words)
        conversational_patterns = [
            'can you help', 'help me', 'i need help',
            'are you there', 'are you ready', 'are you busy',
            'what do you think', 'tell me about yourself',
            'nice to meet you', "let's chat", "let's talk",
            "i'm back", 'im back', 'i have a question',
            'can i ask you something', 'can you help me',
            'yes', 'no', 'ok', 'okay', 'sure', 'alright',
            'got it', 'understood', 'never mind', 'nevermind',
        ]
        for pattern in conversational_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', goal_lower):
                return True

        # Default: NOT simple — use full agent loop with tools
        return False

    def _fast_path_response(self, goal: str) -> dict:
        """Handle simple queries without the full agent loop."""
        logger.debug(f"\n{'='*60}")
        logger.debug(f"Agent responding (fast-path): {goal}")
        logger.debug(f"{'='*60}\n")

        # Build a context-aware system prompt with identity
        identity_prompt = get_identity_prompt()
        system_prompt = f"""{identity_prompt}

You are a helpful AI assistant running locally via Ollama.

About yourself:
- You are an AI agent that can search the web, take screenshots, read files, execute Python code, and more
- You run on Ollama cloud models (Pro subscription) — fast, reasoning, code, vision, and thinking specialists
- You were created to help users with various tasks through conversation and tool use

Guidelines:
- Be friendly, helpful, and concise
- For greetings, respond warmly but briefly
- For questions about yourself, be informative but humble
- For general knowledge questions, give accurate, helpful answers
- Keep responses short (1-3 sentences for simple queries)
- If asked to do something that requires tools (search, screenshot, files, code), say you can help with that

IMPORTANT: If the user asks about something you are not sure about, something recent, current events, news, real-time data (dates, prices, weather, scores, stock prices, exchange rates), or asks you to look something up or verify information — USE the web_search or tavily tool to search the internet FIRST. Do NOT guess or hallucinate. Always verify uncertain facts by searching."""

        # Check if this is an identity question - use reasoning model for better system prompt adherence
        goal_lower = goal.lower()
        identity_patterns = [
            'what is your name', "what's your name", 'who are you', 'your name',
            'are you called', 'what should i call you', 'introduce yourself',
            'tell me about yourself', 'what are you', 'are you an ai', 'are you a bot',
            'what model are you', 'are you qwen', 'are you llama', 'are you deepseek'
        ]
        is_identity_question = any(pattern in goal_lower for pattern in identity_patterns)
        task_type = TaskType.REASONING if is_identity_question else TaskType.SIMPLE

        # Use streaming to start receiving tokens immediately, then accumulate
        response_chunks = []
        for chunk in self.brain.think_stream(
            goal,
            system_prompt=system_prompt,
            use_history=True,  # Enable history for conversational context
            task_type=task_type
        ):
            response_chunks.append(chunk)
        response = "".join(response_chunks)

        model_used = self.brain.get_last_model_used()
        logger.debug(f"[FAST-PATH] Using {model_used}")
        logger.debug(f"Response: {response}\n")

        # Log to metacognition
        self.metacognition.start_goal(goal)
        self.metacognition.increment_iteration()
        self.metacognition.log_evaluation(
            tool="fast_path",
            action="direct_response",
            confidence=100,
            success=True,
            progress="Responded directly without tool execution",
            next_step="complete",
            result_summary=response[:500],
            model_used=model_used
        )

        return self._make_response(goal, response, fast_path=True)

    def _check_identity_update(self, goal: str) -> Optional[dict]:
        """Check if user is trying to update agent identity.

        Args:
            goal: The user's message/goal

        Returns:
            Response dict if identity was updated, None otherwise
        """
        try:
            # Check for name change
            new_name = detect_name_change(goal)
            if new_name:
                self.identity = update_name(new_name)
                response = f"Got it! I'll remember that. You can call me {new_name} from now on."
                logger.debug(f"\n[IDENTITY] Name updated to: {new_name}")
                return self._make_response(goal, response, fast_path=True, metadata={"identity_update": True})

            # Check for personality change
            new_personality = detect_personality_change(goal)
            if new_personality:
                # Append to existing personality or replace
                current = self.identity.get("personality", "") if self.identity else ""
                updated = f"{current}, {new_personality}" if current else new_personality
                self.identity = update_personality(updated)
                response = f"I'll try to be more {new_personality}. Thanks for the feedback!"
                logger.debug(f"\n[IDENTITY] Personality updated to: {updated}")
                return self._make_response(goal, response, fast_path=True, metadata={"identity_update": True})
        except Exception as e:
            logger.warning(f"[IDENTITY] Check failed (non-fatal): {e}")

        return None

    # ========== CLI Permission System ==========

    def set_cli_confirm_callback(self, callback: Callable):
        """Register a callback for CLI permission prompts."""
        self._cli_confirm_callback = callback

    def run(self, goal: str, context: Optional[dict] = None, use_fastpath: Optional[bool] = None, timeout_seconds: int = AGENT_TIMEOUT) -> dict:
        """Run the agent loop to achieve a goal.

        Args:
            goal: The goal to achieve
            context: Optional context dictionary
            use_fastpath: Override fast-path behavior (None uses self.use_fastpath)
            timeout_seconds: Maximum time for the entire agent loop
        """
        start_time = time.time()
        logger.info(f"[AGENT] Starting run() with goal: {goal[:100]}...")

        # Reload custom tools if marketplace installed something
        try:
            _sentinel = Path(__file__).parent / "tools" / "custom" / ".reload_needed"
            if _sentinel.exists():
                from .tools.custom_loader import load_custom_tools
                for _name, _inst in load_custom_tools().items():
                    if _name not in self.tools:
                        self.tools[_name] = _inst
                _sentinel.unlink(missing_ok=True)
                logger.info("[Agent] Reloaded custom tools after marketplace install")
        except Exception as _e:
            logger.warning(f"[Agent] Custom tool reload check failed: {_e}")

        # Record user activity for NeuroDream idle detection
        if hasattr(self, 'neurodream') and self.neurodream:
            self.neurodream.record_activity()

        # ===== Phase 5 Fix 5B: Intrinsic motivation → self-generated goal when idle =====
        if not goal or goal.strip() in ("", "__idle__", "__autonomous__"):
            try:
                from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
                _driven = get_intrinsic_motivation().generate_actions()
                if _driven:
                    _top = _driven[0]
                    goal = f"[Self-directed] {_top.description} ({_top.action})"
                    logger.info(f"[IntrinsicMotivation] Self-generated goal: {goal[:80]}")
                else:
                    return self._make_response(goal or "", "Nothing urgent on my mind right now.", fast_path=True)
            except Exception as _im_err:
                logger.debug(f"[IntrinsicMotivation] Goal gen failed: {_im_err}")
                if not goal:
                    return self._make_response("", "I'm idle.", fast_path=True)

        # ===== AURA FAST PATH - TRY FIRST =====
        # Handle simple queries instantly without agent loop
        fastpath_enabled = use_fastpath if use_fastpath is not None else self.use_fastpath
        if fastpath_enabled and hasattr(self, 'fast_path_handler') and self.fast_path_handler:
            fast_response = self.fast_path_handler.try_fast_path(goal)
            if fast_response:
                logger.debug(f"\n[FAST PATH] Handled instantly: {goal[:50]}...")
                logger.debug(f"[FAST PATH] Response: {fast_response[:100]}...")
                return self._make_response(goal, fast_response, fast_path=True, metadata={"aura_fast_path": True})

        # Check for identity updates first
        identity_response = self._check_identity_update(goal)
        if identity_response:
            return identity_response

        # Check for Git commands - handle directly without LLM hallucination
        git_response = self._handle_git_command(goal)
        if git_response:
            return self._make_response(goal, git_response, fast_path=True, metadata={"git_direct": True})

        # Check for Inner Monologue commands - handle directly
        monologue_response = self._handle_monologue_command(goal)
        if monologue_response:
            return self._make_response(goal, monologue_response, fast_path=True, metadata={"monologue_direct": True})

        # Check for Knowledge Graph commands - handle directly
        kg_response = self._handle_knowledge_graph_command(goal)
        if kg_response:
            return self._make_response(goal, kg_response, fast_path=True, metadata={"kg_direct": True})

        # Check for NeuroDream commands - handle directly
        neurodream_response = self._handle_neurodream_command(goal)
        if neurodream_response:
            return self._make_response(goal, neurodream_response, fast_path=True, metadata={"neurodream_direct": True})

        # Check for fast-path eligibility (reuse fastpath_enabled from above)
        if fastpath_enabled and self._is_simple_query(goal):
            return self._fast_path_response(goal)

        # Always-On Context
        ace_context = ""
        if self.context_engine is not None:
            try:
                _bundle = self.context_engine.gather(goal)
                ace_context = _bundle.to_system_prompt()
            except Exception as _ace_err:
                logger.warning(f"[Agent] ACE context gather failed: {_ace_err}")
        context = context or {}  # NOTE: reserved for future use, not yet wired into agent loop
        self.brain._last_screenshot_path = None
        self.metacognition.start_goal(goal)
        self.monologue.start_session()
        self.monologue.think("perceive", f"Received: '{goal[:80]}{'...' if len(goal) > 80 else ''}'")

        logger.info(f"[AGENT] Starting ReAct loop for: {goal[:80]}")

        # ===== Adaptive Planning (Roadmap 5.4) =====
        # Classify task complexity and generate plan for complex tasks
        task_plan = None
        task_is_complex = False
        if self.adaptive_planner:
            try:
                task_is_complex = self.adaptive_planner.classify(goal)
                if task_is_complex:
                    task_plan = self.adaptive_planner.generate_plan(goal)
                    if task_plan:
                        logger.info(f"[Planner] Plan generated: {len(task_plan.steps)} steps")
                    else:
                        logger.debug("[Planner] Complex task but plan generation failed, proceeding without plan")
            except Exception as _plan_err:
                logger.debug(f"[Planner] Classification/planning failed: {_plan_err}")

        # ===== ReAct Loop =====
        # Build system prompt with memory, identity, emotion, context
        system_prompt = self._build_react_system_prompt(goal, ace_context)

        # Decide: Code Agent Mode vs Standard Tool Mode (Roadmap 5.1)
        use_code_agent = self._should_use_code_agent(goal)
        _code_agent_inst = None
        _tool_namespace = None

        if use_code_agent:
            logger.info(f"[AGENT] Using CODE AGENT MODE for: {goal[:80]}")
            _code_agent_inst = CodeAgentMode(self)
            _tool_namespace = _code_agent_inst.build_tool_namespace()
            # Use code agent system prompt instead of tool schemas
            system_prompt += "\n\n" + CODE_AGENT_SYSTEM_PROMPT
            tool_schemas = []  # No Ollama tool schemas in code mode
        else:
            # Select relevant tools via Tool RAG
            tool_schemas = self._build_tool_schemas(goal)

        # Conversation messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

        # Inject plan context into conversation if we have a plan
        if task_plan:
            messages.append({
                "role": "assistant",
                "content": f"Let me work through this systematically.\n\n{task_plan.to_prompt_context()}",
            })

        # Loop state
        iteration = 0
        final_response = ""
        done = False
        state_hashes = set()
        consecutive_failures = 0
        tool_calls_total = 0

        while not done and iteration < self.max_iterations:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(f"[AGENT] Timeout after {elapsed:.1f}s")
                return self._format_timeout_response(goal, iteration, elapsed)

            iteration += 1
            self.metacognition.increment_iteration()
            logger.info(f"[AGENT] Iteration {iteration}/{self.max_iterations}")

            try:
                # Route to code agent or standard tool mode
                if use_code_agent and _code_agent_inst and _tool_namespace is not None:
                    step_result = self._react_step_code(
                        messages, _code_agent_inst, _tool_namespace,
                        iteration, consecutive_failures,
                    )
                else:
                    step_result = self._react_step(
                        messages, tool_schemas, iteration,
                        state_hashes, consecutive_failures,
                    )

                status = step_result["status"]
                if status == "error":
                    final_response = step_result.get("response", "An error occurred.")
                    break
                elif status == "done":
                    final_response = step_result["response"]
                    done = True
                    break
                elif status == "fallback_to_tools":
                    # Code agent failed repeatedly — switch to standard tool calling
                    use_code_agent = False
                    logger.warning("[AGENT] Code agent failed, falling back to standard tool mode")
                    consecutive_failures = 0
                elif status == "incomplete":
                    pass  # LLM gave too-short answer, nudge injected
                # else: "continue" -- tools executed, loop continues

                tool_calls_total += step_result.get("tool_calls_count", 0)
                consecutive_failures = step_result.get("consecutive_failures", consecutive_failures)

                # === Adaptive Re-planning (Roadmap 5.4) ===
                if task_plan and self.adaptive_planner and status == "continue":
                    try:
                        # Advance plan step on successful tool execution
                        if step_result.get("tool_calls_count", 0) > 0 and consecutive_failures == 0:
                            self.adaptive_planner.advance_step(
                                step_result.get("response", "")[:200]
                            )
                        # Tick the planner step counter, then check if re-planning is due
                        self.adaptive_planner.tick()
                        if self.adaptive_planner.should_replan():
                            recent_results = ""
                            for msg in messages[-4:]:
                                if msg.get("role") == "tool":
                                    recent_results += msg.get("content", "")[:200] + "\n"
                            updated_plan = self.adaptive_planner.replan(recent_results)
                            if updated_plan:
                                task_plan = updated_plan
                                messages.append({
                                    "role": "assistant",
                                    "content": f"[Re-planning after {iteration} steps]\n{task_plan.to_prompt_context()}",
                                })
                                logger.info(f"[Planner] Re-planned at iteration {iteration}: {len(task_plan.remaining_steps)} steps remaining")
                    except Exception as _replan_err:
                        logger.debug(f"[Planner] Re-plan check failed: {_replan_err}")

            except Exception as e:
                logger.error(f"[AGENT] Error in iteration {iteration}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    final_response = f"Encountered repeated errors: {e}"
                    break
                continue

        elapsed = time.time() - start_time
        logger.info(f"[AGENT] Completed in {elapsed:.1f}s, {iteration} iterations, {tool_calls_total} tool calls")

        # Store episode in memory (no LLM call — defer summarization to NeuroDream)
        self._store_episode(goal, final_response)

        # Clean up planner state
        plan_metadata = {}
        if self.adaptive_planner:
            if task_plan:
                plan_metadata = {
                    "planned": True,
                    "plan_steps": len(task_plan.steps),
                    "plan_completed": len(task_plan.completed_steps),
                    "replans": task_plan.replan_count,
                }
            else:
                plan_metadata = {"planned": False}
            self.adaptive_planner.reset()

        return self._make_response(
            goal,
            final_response,
            completed=done,
            iterations=iteration,
            metadata={
                "react_loop": True,
                "code_agent_mode": use_code_agent,
                "tool_calls": tool_calls_total,
                "adaptive_planning": plan_metadata,
                "final_evaluation": {
                    "success": done and bool(final_response),
                    "confidence": 90 if done else 30,
                    "progress": final_response[:200] if final_response else "No response generated",
                },
            },
        )

    def _format_timeout_response(self, goal: str, iteration: int, elapsed: float) -> dict:
        """Format a response when agent times out."""
        return self._make_response(
            goal,
            f"I ran out of time after {elapsed:.0f} seconds. Please try a simpler request.",
            completed=False,
            iterations=iteration,
            metadata={
                "timeout": True,
                "elapsed_seconds": elapsed,
                "final_evaluation": {
                    "success": False,
                    "confidence": 0,
                    "progress": f"Timeout after {iteration} iterations ({elapsed:.1f}s)",
                },
            },
        )

    # =================================================================
    # ReAct Loop Helper Methods
    # =================================================================

    def _build_react_system_prompt(self, goal: str, ace_context: str = "") -> str:
        """Build system prompt for ReAct loop with memory, identity, emotion.

        Keeps total prompt compact for tool-calling mode where context is precious.
        Identity + emotion are capped at 1500 chars combined.
        """
        parts = []

        # Identity (truncate for tool-calling mode)
        identity_text = get_identity_prompt()
        if len(identity_text) > 1200:
            identity_text = identity_text[:1200] + "..."
        parts.append(identity_text)

        # ALMA emotional tone (truncate to keep prompt lean)
        try:
            from aura.emotion.integration import get_emotional_tone_modifier
            tone = get_emotional_tone_modifier()
            if tone:
                parts.append(tone[:300])
        except Exception as _tone_err:
            logger.debug(f"[Agent] Emotional tone modifier unavailable: {_tone_err}")

        # ACE context
        if ace_context:
            parts.append(ace_context)

        # UserProfile injection
        try:
            from aura.memory.user_profile import load_profile
            _profile = load_profile()
            _profile_str = _profile.to_system_prompt()
            if _profile_str:
                parts.append(_profile_str)
        except Exception as _profile_err:
            logger.debug(f"[Agent] User profile load failed: {_profile_err}")

        # Memory recall (parallel) — unified store + KG
        try:
            _ctx_futures = {}
            _umem = get_unified_memory()
            _ctx_futures["memory"] = _AGENT_EXECUTOR.submit(_umem.query, goal, 3)
            if self.kg_bridge is not None:
                _ctx_futures["kg"] = _AGENT_EXECUTOR.submit(self.kg_bridge.get_context_for_query, goal, 5)

            for _key, _fut in _ctx_futures.items():
                try:
                    _result = _fut.result(timeout=2.0)
                    if _key == "memory" and _result:
                        mem_text = "\n".join(r.content[:150] for r in _result[:3])
                        parts.append(f"[Relevant memories]\n{mem_text}")
                    elif _key == "kg" and _result:
                        parts.append(f"[Knowledge context]\n{str(_result)[:300]}")
                except Exception as _ctx_err:
                    logger.debug(f"[Agent] {_key} context retrieval failed: {_ctx_err}")
        except Exception as _mem_err:
            logger.debug(f"[Agent] Memory/KG context setup failed: {_mem_err}")

        # ReAct instructions (plan-aware when a plan exists)
        react_instruction = (
            "You are an AI agent with access to tools. "
            "Use the provided tools to accomplish the user's goal. "
            "When done, respond with your final answer (no tool call). "
            "Be concise and direct.\n\n"
            "IMPORTANT: If the user asks about something you are not sure about, "
            "something recent, current events, news, real-time data (dates, prices, "
            "weather, scores, stock prices, exchange rates), or asks you to look "
            "something up or verify information — USE the web_search or tavily tool "
            "to search the internet FIRST. Do NOT guess or hallucinate. Always "
            "verify uncertain facts by searching."
        )
        if self.adaptive_planner and self.adaptive_planner.current_plan:
            react_instruction += (
                "\n\nYou have a plan for this task. Follow the plan steps in order. "
                "Reference the current step in your reasoning. "
                "If a step is blocked or unnecessary, skip it and move on."
            )
        parts.append(react_instruction)

        return "\n\n".join(parts)

    def _build_tool_schemas(self, goal: str) -> list:
        """Select relevant tool schemas for this goal.

        Always includes the 11 core dev tools (read_file, grep, list_dir, etc.)
        plus up to 4 additional agent-specific tools from Tool RAG.
        """
        try:
            from aura.core.tool_schemas import AGENTIC_TOOLS
            core_schemas = list(AGENTIC_TOOLS)
            core_names = {s["function"]["name"] for s in core_schemas}
        except ImportError:
            return []

        # Add a few agent-specific tools from RAG if relevant
        if self.tool_rag and self.tool_rag._schemas_loaded:
            extra = self.tool_rag.select_tools(goal, k=6)
            for schema in extra:
                name = schema["function"]["name"]
                if name not in core_names:
                    core_schemas.append(schema)
                    core_names.add(name)
                    if len(core_schemas) >= 15:  # Cap at 15 total tools
                        break

        return core_schemas

    # Tool name mapping: Ollama schema names → agent tool dispatch
    _TOOL_NAME_MAP = {
        "read_file": ("filesystem", "read"),
        "grep": ("code_search", "grep"),
        "glob": ("code_search", "glob"),
        "list_dir": ("filesystem", "list"),
        "edit_file": ("code_edit", "edit"),
        "write_file": ("filesystem", "write"),
        "shell": ("shell_executor", "execute"),
        "git": ("git", "execute"),
        "search_web": (None, None),  # Special: try tavily, brave, web_search
        "project_structure": ("code_search", "project_structure"),
        "spawn_agent": (None, None),  # Not dispatched via self.tools
    }

    def _execute_tool_call(self, tool_name: str, args: dict) -> str:
        """Execute a tool call and return result as string for the LLM.

        Priority:
        1. ToolExecutor (handles dev tools: read_file, grep, list_dir, shell, etc.)
           Also resolves aliases (filesystem→list_dir, code_search→grep, etc.)
        2. Agent tools (self.tools dict — screenshot, vision, search, etc.)
        3. Web search fallback chain (tavily → brave → web_search)
        """
        MAX_RESULT = 8000

        # 1. Try ToolExecutor first — handles dev tools + aliases + sandbox bypass
        if self._tool_executor:
            result = self._tool_executor.execute(tool_name, args)
            # Check if executor returned an "Unknown tool" error — if so, fall through
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and parsed.get("error", "").startswith("Unknown tool"):
                        pass  # Fall through to agent tools
                    else:
                        return result[:MAX_RESULT]
                except (json.JSONDecodeError, ValueError):
                    return result[:MAX_RESULT]

        # 2. Agent tools dispatch
        if tool_name in self.tools:
            try:
                tool = self.tools[tool_name]
                action = args.get("action", "") if isinstance(args, dict) else str(args)
                if not action:
                    # Build action string from structured args
                    action = " ".join(f"{v}" for v in args.values()) if isinstance(args, dict) else str(args)
                if hasattr(tool, 'execute'):
                    result = tool.execute(action)
                    return json.dumps(result, default=str)[:MAX_RESULT]
            except Exception as e:
                return json.dumps({"error": f"Tool '{tool_name}' failed: {e}"})

        # 3. Web search fallback chain
        if tool_name in ("search_web", "web_search", "search"):
            query = args.get("query", args.get("action", str(args)))
            for sn in ("tavily_search", "brave_search", "web_search"):
                if sn in self.tools:
                    try:
                        result = self.tools[sn].execute(f"search {query}")
                        return json.dumps(result, default=str)[:MAX_RESULT]
                    except Exception:
                        continue

        return json.dumps({"error": f"No handler for tool: {tool_name}"})

    # Cloud models known to support tool calling well, in preference order
    _REACT_TOOL_MODEL = "glm-5:cloud"         # Fast, reliable tool calling
    _REACT_CODE_MODEL = "deepseek-v3.2:cloud"  # Better for code tasks
    _REACT_REASON_MODEL = "kimi-k2.5:cloud"    # Best for complex reasoning

    def _pick_react_model(self, messages: list, iteration: int) -> str:
        """Pick model for this ReAct iteration.

        User's explicit model override always wins. Otherwise auto-route
        based on whether the conversation involves code edits.
        """
        # User explicitly selected a model — respect it
        if self.brain._model_override:
            return self.brain._model_override

        has_edits = False
        for msg in messages:
            for tc in msg.get("tool_calls", []) or []:
                fn_name = tc.get("function", {}).get("name", "")
                if fn_name in ("edit_file", "write_file"):
                    has_edits = True
                    break

        if has_edits:
            return self._REACT_CODE_MODEL
        return self._REACT_TOOL_MODEL

    def _store_episode(self, goal: str, response: str):
        """Store goal+response in memory after the loop ends. No LLM call."""
        if not response:
            return
        try:
            if self.memory:
                self.memory.store(
                    content=f"Goal: {goal[:200]}\nResult: {response[:300]}",
                    source="react_loop",
                    importance=0.5,
                )
        except Exception as e:
            logger.debug(f"[AGENT] Episode storage failed: {e}")

    # =================================================================
    # ReAct Step — single-step agent method (Roadmap 1.1)
    # =================================================================

    def _react_step(
        self,
        messages: list,
        tool_schemas: list,
        iteration: int,
        state_hashes: set,
        consecutive_failures: int,
    ) -> dict:
        """Execute one ReAct step: LLM call + tool execution + deterministic evaluation.

        This replaces the old 4-phase OPAE loop (observe/plan/act/evaluate) with a
        single LLM call that combines thought + action, followed by deterministic
        evaluation of the tool result (no LLM call for eval).

        Args:
            messages: Conversation history (mutated in-place with new messages)
            tool_schemas: Tool schemas for Ollama tool calling
            iteration: Current iteration number
            state_hashes: Set of seen action hashes (mutated in-place)
            consecutive_failures: Current failure count

        Returns:
            {
                "status": "done" | "continue" | "error" | "incomplete",
                "response": str (final answer when done, error message when error),
                "tool_calls_count": int,
                "consecutive_failures": int,
            }
        """
        # Pick model for this step
        step_model = self._pick_react_model(messages, iteration)

        # === Single LLM call (thought + action combined) ===
        step = self.brain.react_step(messages, tool_schemas, model_override=step_model)

        # Handle LLM error with fallback chain
        if "error" in step:
            logger.error(f"[REACT] LLM error: {step['error']}")
            _fallback_models = ["glm-5:cloud", "deepseek-v3.2:cloud", "kimi-k2.5:cloud"]
            _fallback_models = [m for m in _fallback_models if m != step_model]
            for _fb_model in _fallback_models:
                logger.info(f"[REACT] Trying fallback model: {_fb_model}")
                step = self.brain.react_step(messages, tool_schemas, model_override=_fb_model)
                if "error" not in step:
                    break
            if "error" in step:
                return {"status": "error", "response": f"I encountered an error: {step['error']}",
                        "tool_calls_count": 0, "consecutive_failures": consecutive_failures}

        thought = step.get("thought", "")
        tool_calls = step.get("tool_calls", [])

        # Record thought in inner monologue
        if thought:
            self.monologue.think("reason", thought[:200])

        # === No tool calls: LLM is done, content is the final answer ===
        if step.get("done"):
            _stripped = (thought or "").strip()
            _looks_incomplete = (
                len(_stripped) < 20
                and _stripped
                and _stripped[-1] not in '.!?'
                and iteration < self.max_iterations - 1
            )
            if _looks_incomplete:
                # Nudge the LLM to provide a real answer or use a tool
                messages.append({"role": "assistant", "content": thought})
                messages.append({
                    "role": "user",
                    "content": "Please provide your complete answer, or use a tool if you need more information.",
                })
                return {"status": "incomplete", "response": "",
                        "tool_calls_count": 0, "consecutive_failures": consecutive_failures}

            return {"status": "done", "response": step.get("final_answer", thought),
                    "tool_calls_count": 0, "consecutive_failures": consecutive_failures}

        # === Has tool calls: serialize message, execute tools, evaluate deterministically ===

        # Build assistant message for conversation history
        msg_dict = {
            "role": "assistant",
            "content": thought,
            "tool_calls": [
                {"function": {"name": tc["name"], "arguments": tc["args"]}}
                for tc in tool_calls
            ],
        }
        messages.append(msg_dict)

        # Execute each tool call
        calls_this_step = 0
        for tc in tool_calls:
            tool_name = tc["name"]
            args = tc["args"]

            # Loop guard: state dedup using raw string keys (hash() can collide)
            try:
                state_key = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
            except (TypeError, ValueError):
                state_key = f"{tool_name}:{str(args)}"
            if state_key in state_hashes:
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": "Duplicate action. Try a different approach."}),
                })
                continue
            state_hashes.add(state_key)

            # Execute tool
            tool_result = self._execute_tool_call(tool_name, args)
            calls_this_step += 1
            messages.append({"role": "tool", "content": tool_result})

            # === Deterministic evaluation (no LLM call) ===
            eval_result = self._evaluate_tool_result(tool_result)

            if not eval_result["success"]:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({
                            "warning": f"Multiple failures ({consecutive_failures}). Try a different tool or approach.",
                        }),
                    })
                    consecutive_failures = 0
            else:
                consecutive_failures = 0

            # Log deterministic eval to metacognition
            self.metacognition.log_evaluation(
                tool=tool_name,
                action=json.dumps(args, default=str)[:200],
                confidence=eval_result["confidence"],
                success=eval_result["success"],
                progress=eval_result["reason"],
                next_step="continue",
                result_summary=tool_result[:500],
                model_used=step.get("model", ""),
            )

        return {
            "status": "continue",
            "response": "",
            "tool_calls_count": calls_this_step,
            "consecutive_failures": consecutive_failures,
        }

    @staticmethod
    def _evaluate_tool_result(tool_result: str) -> dict:
        """Deterministic evaluation of a tool result — no LLM call.

        Tries to parse the result as JSON first and checks structured keys
        (``success``, ``error``). Falls back to string matching only when
        JSON parsing fails.

        Returns:
            {"success": bool, "confidence": int (0-100), "reason": str}
        """
        # --- Phase 1: Try structured JSON evaluation first ---
        try:
            parsed = json.loads(tool_result)
            if isinstance(parsed, dict):
                # Explicit "success" key is the strongest signal
                if "success" in parsed:
                    if parsed["success"]:
                        if "error" not in parsed:
                            return {"success": True, "confidence": 95, "reason": "Tool reported success"}
                        # success=True but also has error key — trust success flag
                        return {"success": True, "confidence": 80, "reason": "Tool reported success (with warning)"}
                    else:
                        reason = str(parsed.get("error", "Tool reported failure"))[:100]
                        return {"success": False, "confidence": 90, "reason": reason}

                # No "success" key, but has "error" key
                if "error" in parsed:
                    error_msg = str(parsed["error"])[:100]
                    return {"success": False, "confidence": 85, "reason": f"Error: {error_msg}"}

                # Structured dict with no success/error keys — assume OK
                if len(parsed) > 0:
                    return {"success": True, "confidence": 70, "reason": "Structured result (no error keys)"}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # --- Phase 2: Fallback to string matching for non-JSON output ---
        result_lower = tool_result.lower()

        has_traceback = 'traceback' in result_lower or 'exception' in result_lower
        has_not_found = 'not found' in result_lower or 'no such file' in result_lower
        has_permission = 'permission denied' in result_lower or 'access denied' in result_lower
        has_timeout = 'timeout' in result_lower or 'timed out' in result_lower

        if has_traceback:
            return {"success": False, "confidence": 90, "reason": "Exception traceback in output"}
        if has_permission:
            return {"success": False, "confidence": 95, "reason": "Permission denied"}
        if has_timeout:
            return {"success": False, "confidence": 90, "reason": "Operation timed out"}
        if has_not_found:
            return {"success": False, "confidence": 80, "reason": "Resource not found"}

        # No explicit indicators — assume success if we got non-empty output
        if len(tool_result.strip()) > 2:
            return {"success": True, "confidence": 70, "reason": "Non-empty result (no error indicators)"}

        # Empty or minimal output
        return {"success": False, "confidence": 50, "reason": "Empty or minimal output"}

    # =================================================================
    # Code Agent Mode — LLM writes Python code as actions (5.1)
    # =================================================================

    def _should_use_code_agent(self, goal: str) -> bool:
        """Decide whether to use code agent mode for this goal.

        Code agent mode is better for complex multi-step tasks where the LLM
        can write Python loops, conditionals, and compose tool calls in code.
        Standard tool mode remains the default for simple queries.

        Returns True if code agent mode should be used.
        """
        if not CODE_AGENT_AVAILABLE:
            return False

        # Check config flag (allow disabling globally)
        if not getattr(Config, 'CODE_AGENT_ENABLED', True):
            return False

        return should_use_code_agent(goal)

    def _react_step_code(
        self,
        messages: list,
        code_agent: "CodeAgentMode",
        tool_namespace: dict,
        iteration: int,
        consecutive_failures: int,
    ) -> dict:
        """Execute one ReAct code step: LLM writes Python code, we execute it.

        This is the code-agent-mode equivalent of _react_step(). Instead of
        Ollama structured tool calling, the LLM writes a Python code block
        that calls tool functions directly.

        Args:
            messages: Conversation history (mutated in-place)
            code_agent: CodeAgentMode instance
            tool_namespace: Dict of tool functions for the code sandbox
            iteration: Current iteration number
            consecutive_failures: Current failure count

        Returns:
            Same shape as _react_step():
            {"status": "done"|"continue"|"error", "response": str, ...}
        """
        step_model = self._pick_react_model(messages, iteration)

        # LLM call — produces thought + Python code block (no tools= param)
        step = self.brain.react_step_code(messages, model_override=step_model)

        if "error" in step:
            logger.error(f"[REACT-CODE] LLM error: {step['error']}")
            return {
                "status": "error",
                "response": f"Code agent error: {step['error']}",
                "tool_calls_count": 0,
                "consecutive_failures": consecutive_failures,
            }

        thought = step.get("thought", "")
        code = step.get("code")

        # Record thought
        if thought:
            self.monologue.think("reason", f"[code-mode] {thought[:200]}")

        # No code block = LLM is done, final answer
        if step.get("done"):
            final = step.get("final_answer", thought)
            if len(final.strip()) < 20 and iteration < self.max_iterations - 1:
                messages.append({"role": "assistant", "content": final})
                messages.append({
                    "role": "user",
                    "content": "Please provide your complete answer, or write more code if needed.",
                })
                return {
                    "status": "incomplete",
                    "response": "",
                    "tool_calls_count": 0,
                    "consecutive_failures": consecutive_failures,
                }
            return {
                "status": "done",
                "response": final,
                "tool_calls_count": 0,
                "consecutive_failures": consecutive_failures,
            }

        # Has code block — execute it
        messages.append({
            "role": "assistant",
            "content": step.get("final_answer", "") or f"Thought: {thought}\n\n```python\n{code}\n```",
        })

        logger.info(f"[REACT-CODE] Executing code block ({len(code)} chars)")
        exec_result = code_agent.execute_code_safely(code, tool_namespace)
        formatted = code_agent.format_execution_result(exec_result)

        # Add execution result as a "user" message (simulates tool result)
        messages.append({"role": "user", "content": f"Code execution result:\n{formatted}"})

        # Deterministic eval of execution result
        has_error = bool(exec_result.get("error")) or exec_result.get("timed_out", False)
        if has_error:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.warning("[REACT-CODE] 3 consecutive failures, falling back to standard tools")
                return {
                    "status": "fallback_to_tools",
                    "response": "",
                    "tool_calls_count": 0,
                    "consecutive_failures": 0,
                }
        else:
            consecutive_failures = 0

        # Log to metacognition
        self.metacognition.log_evaluation(
            tool="code_agent",
            action=code[:200],
            confidence=70 if not has_error else 30,
            success=not has_error,
            progress="Code executed" if not has_error else "Code failed",
            next_step="continue",
            result_summary=formatted[:500],
            model_used=step.get("model", ""),
        )

        return {
            "status": "continue",
            "response": "",
            "tool_calls_count": 1,
            "consecutive_failures": consecutive_failures,
        }

    def _handle_monologue_command(self, message: str) -> Optional[str]:
        """Handle inner monologue commands directly, bypassing the LLM.

        Args:
            message: The user's message

        Returns:
            Formatted result string if monologue command, None otherwise
        """
        msg_lower = message.lower()

        monologue_keywords = [
            'show thoughts', 'your thoughts', 'recent thoughts',
            'think aloud', 'verbosity', 'why did you do that',
            'explain your reasoning', 'reasoning chain', 'export thoughts'
        ]

        if not any(kw in msg_lower for kw in monologue_keywords):
            return None

        if "inner_monologue" not in self.tools:
            return "Inner monologue not available."

        monologue = self.tools["inner_monologue"]
        result = monologue.execute(message)

        if result.get("success"):
            if "thoughts" in result:
                return result["thoughts"]
            if "reasoning_chain" in result:
                return result["reasoning_chain"]
            if "message" in result:
                return result["message"]
            return str(result)

        return result.get("error", "Unknown error")

    def _handle_knowledge_graph_command(self, message: str) -> Optional[str]:
        """Handle knowledge graph commands directly, bypassing the LLM.

        Supports both the legacy KnowledgeGraphTool and the new KG Brain.

        Args:
            message: The user's message

        Returns:
            Formatted result string if KG command, None otherwise
        """
        msg_lower = message.lower()

        kg_keywords = [
            'what do you know about', 'knowledge graph', 'show graph',
            'how is', 'related to', 'connected to', 'find path between',
            'what have you learned', 'consolidate memory', 'graph stats',
            'kg brain', 'kg stats', 'add to knowledge', 'remember that',
            'learn that', 'extract entities', 'kg query'
        ]

        if not any(kw in msg_lower for kw in kg_keywords):
            return None

        # Try KG Brain first (new system)
        if self.kg_brain is not None and self.kg_query_engine is not None:
            # Handle KG Brain specific commands
            if "kg brain" in msg_lower or "kg stats" in msg_lower:
                stats = self.kg_brain.get_statistics()
                bridge_stats = self.kg_bridge.get_statistics() if self.kg_bridge else {}
                return (
                    f"**Knowledge Graph Brain Statistics**\n"
                    f"- Total Entities: {stats.get('total_entities', 0)}\n"
                    f"- Total Relationships: {stats.get('total_relationships', 0)}\n"
                    f"- Average Importance: {stats.get('average_importance', 0):.2f}\n"
                    f"- Entity Types: {stats.get('entity_type_distribution', {})}\n"
                    f"- Entities Extracted: {bridge_stats.get('total_entities_extracted', 0)}\n"
                    f"- Extractions Triggered: {bridge_stats.get('total_extractions_triggered', 0)}"
                )

            if "what do you know about" in msg_lower:
                topic = msg_lower.split("what do you know about")[-1].strip().rstrip("?")
                # Query KG Brain
                result = self.kg_query_engine.query(topic, mode=QueryMode.HYBRID, max_entities=10)
                if result.entities:
                    return result.context_string
                # Fall through to legacy KG if no results

            if "how is" in msg_lower and "related to" in msg_lower:
                parts = msg_lower.replace("?", "").split("related to")
                if len(parts) == 2:
                    source = parts[0].replace("how is", "").strip()
                    target = parts[1].strip()
                    # Try to find path in KG Brain
                    path = self.kg_query_engine.find_path(source, target)
                    if path:
                        return f"Connection found: {path}"

            if "extract entities" in msg_lower or "learn that" in msg_lower or "remember that" in msg_lower:
                # Force extraction from message
                text_to_extract = message.split("that", 1)[-1].strip() if "that" in message else message
                if self.kg_bridge:
                    entity_ids = self.kg_bridge.force_extract(text_to_extract, context="user command")
                    if entity_ids:
                        return f"Extracted and stored {len(entity_ids)} entities in knowledge graph."
                    return "No entities could be extracted from that text."

            if "consolidate memory" in msg_lower:
                # Apply decay and prune
                self.kg_brain.decay_importance(decay_rate=0.05)
                self.kg_brain.prune_low_importance(threshold=0.03)
                if self.kg_bridge:
                    self.kg_bridge.flush()
                stats = self.kg_brain.get_statistics()
                return f"Memory consolidated. Current state: {stats['total_entities']} entities, {stats['total_relationships']} relationships."

            # Generic KG Brain query
            if "kg query" in msg_lower:
                query = msg_lower.replace("kg query", "").strip()
                result = self.kg_query_engine.query(query, mode=QueryMode.HYBRID)
                return result.context_string if result.entities else "No matching entities found."

        # Fall back to legacy knowledge_graph tool
        if "knowledge_graph" not in self.tools:
            if self.kg_brain is None:
                return "Knowledge graph not available. Install kuzu: pip install kuzu"
            return "No results found in knowledge graph."

        kg = self.tools["knowledge_graph"]

        # Handle specific patterns with legacy tool
        if "what do you know about" in msg_lower:
            topic = msg_lower.split("what do you know about")[-1].strip().rstrip("?")
            result = kg.execute(f"query {topic}")
            if result.get("success") and result.get("results"):
                return "Here's what I know:\n" + "\n".join(result["results"])
            return f"I don't have much knowledge about '{topic}' yet."

        if "how is" in msg_lower and "related to" in msg_lower:
            parts = msg_lower.replace("?", "").split("related to")
            if len(parts) == 2:
                source = parts[0].replace("how is", "").strip()
                target = parts[1].strip()
                result = kg.execute(f"path {source} to {target}")
                if result.get("success") and result.get("path"):
                    return f"Connection: {result['path']}"
                return f"No direct connection found between '{source}' and '{target}'."

        if "consolidate memory" in msg_lower:
            result = kg.execute("consolidate")
            return f"Memory consolidated: {result.get('merged_nodes', 0)} nodes merged, {result.get('pruned_edges', 0)} edges pruned."

        if "graph stats" in msg_lower or "knowledge graph" in msg_lower:
            result = kg.execute("stats")
            if result.get("success"):
                return f"Knowledge Graph: {result['total_nodes']} nodes, {result['total_edges']} edges, {result['clusters']} clusters"

        # Generic query
        result = kg.execute(message)
        if result.get("success"):
            if result.get("results"):
                return "\n".join(result["results"])
            return str(result)

        return result.get("error", "Unknown error")

    def _handle_neurodream_command(self, message: str) -> Optional[str]:
        """Handle NeuroDream sleep/dream commands directly.

        Args:
            message: The user's message

        Returns:
            Formatted result string if NeuroDream command, None otherwise
        """
        msg_lower = message.lower()

        neurodream_keywords = [
            'go to sleep', 'sleep now', 'start sleeping', 'enter sleep',
            'dream status', 'sleep status', 'neurodream status',
            'wake up', 'stop sleeping',
            'dream journal', 'show dreams', 'recent dreams',
            'dream insights', 'show insights',
            'sleep patterns', 'consolidated patterns'
        ]

        if not any(kw in msg_lower for kw in neurodream_keywords):
            return None

        if not hasattr(self, 'neurodream') or self.neurodream is None:
            return "NeuroDream not available."

        # Handle specific patterns
        if any(kw in msg_lower for kw in ['go to sleep', 'sleep now', 'start sleeping', 'enter sleep']):
            if self.neurodream.current_phase != SleepPhase.AWAKE:
                return f"Already in {self.neurodream.current_phase.value} phase."
            result = self.neurodream.enter_sleep(trigger="manual")
            if result.get("success"):
                return "Entering sleep mode... Beginning memory consolidation cycle."
            return f"Could not enter sleep: {result.get('error', 'Unknown error')}"

        if any(kw in msg_lower for kw in ['dream status', 'sleep status', 'neurodream status']):
            status = self.neurodream.get_status()
            phase_emoji = {
                "awake": "Awake",
                "light": "Light Sleep",
                "deep": "Deep Sleep",
                "rem": "REM Sleep",
                "waking": "Waking Up"
            }
            return (f"**NeuroDream Status**\n"
                   f"- Phase: {phase_emoji.get(status['phase'], status['phase'])}\n"
                   f"- Total Sessions: {status['total_sessions']}\n"
                   f"- Total Insights: {status['total_insights']}\n"
                   f"- Idle Minutes: {status['idle_minutes']:.1f}\n"
                   f"- Last Sleep: {status['last_sleep'] or 'Never'}")

        if any(kw in msg_lower for kw in ['wake up', 'stop sleeping']):
            if self.neurodream.current_phase == SleepPhase.AWAKE:
                return "Already awake."
            result = self.neurodream.wake_up(reason="manual")
            summary = result.get("summary", {})
            return (f"Waking up...\n"
                   f"- Phases completed: {', '.join(summary.get('phases_completed', []))}\n"
                   f"- Insights generated: {summary.get('insights_generated', 0)}\n"
                   f"- Patterns found: {summary.get('patterns_found', 0)}")

        if any(kw in msg_lower for kw in ['dream journal', 'show dreams', 'recent dreams']):
            entries = self.neurodream.get_dream_journal(n=5)
            if not entries:
                return "No dream journal entries yet."
            lines = ["**Recent Dream Sessions:**"]
            for entry in entries[-5:]:
                phases = ', '.join(entry.get('phases_completed', []))
                insights = entry.get('insights_generated', 0)
                lines.append(f"- {entry.get('start_time', 'Unknown')[:16]}: {phases} ({insights} insights)")
            return '\n'.join(lines)

        if any(kw in msg_lower for kw in ['dream insights', 'show insights']):
            insights = self.neurodream.get_insights(n=5)
            if not insights:
                return "No dream insights generated yet."
            lines = ["**Recent Dream Insights:**"]
            for insight in insights[-5:]:
                lines.append(f"- [{insight.get('insight_type', 'unknown')}] {insight.get('content', '')[:100]}...")
            return '\n'.join(lines)

        if any(kw in msg_lower for kw in ['sleep patterns', 'consolidated patterns']):
            patterns = self.neurodream.get_patterns(n=5)
            if not patterns:
                return "No patterns consolidated yet."
            lines = ["**Consolidated Patterns:**"]
            for pattern in patterns[-5:]:
                lines.append(f"- [{pattern.get('pattern_type', 'unknown')}] {pattern.get('description', '')[:80]}...")
            return '\n'.join(lines)

        return None

    def _handle_git_command(self, message: str) -> Optional[str]:
        """Handle Git commands directly, bypassing the LLM.

        Args:
            message: The user's message

        Returns:
            Formatted result string if Git command, None otherwise
        """
        message_lower = message.lower()

        # Check if this is a Git command - expanded natural language patterns
        git_keywords = [
            # Explicit git commands
            'git status', 'git log', 'git diff', 'git branch', 'git stash',
            # Branch queries
            'what branch', 'which branch', 'current branch', 'show branches', 'list branches',
            # Commit queries
            'show commits', 'recent commits', 'commit history', 'last commit',
            # Status queries
            'staged files', 'unstaged', 'untracked', 'show changes',
            'what changed', 'pending changes', 'working tree'
        ]
        if not any(kw in message_lower for kw in git_keywords):
            return None  # Not a Git command

        # Get the git tool
        git_tool = self.tools.get('git')
        if not git_tool:
            return "Git tool is not available."

        # Map natural language to specific git actions
        if any(kw in message_lower for kw in ['what branch', 'which branch', 'current branch', 'show branches', 'list branches']):
            result = git_tool.branch('.')
        elif any(kw in message_lower for kw in ['show commits', 'recent commits', 'commit history', 'last commit', 'git log']):
            result = git_tool.log('.', count=5)
        elif any(kw in message_lower for kw in ['staged files', 'unstaged', 'untracked', 'git status', 'what changed', 'pending changes', 'working tree']):
            result = git_tool.status('.')
        elif any(kw in message_lower for kw in ['show changes', 'git diff']):
            result = git_tool.diff('.')
        elif 'git stash' in message_lower:
            if 'list' in message_lower:
                result = git_tool.stash('.', 'list')
            elif 'pop' in message_lower:
                result = git_tool.stash('.', 'pop')
            else:
                result = git_tool.stash('.', 'push')
        else:
            # Fallback to execute() for other commands
            result = git_tool.execute(message)

        if result.get('success'):
            return result.get('output', str(result))
        else:
            return f"Git error: {result.get('error', 'Unknown error')}"

    def _handle_direct_search(self, message: str, synthesize: bool = True) -> Optional[str]:
        """Handle explicit search requests directly, bypassing agent loop.

        This prevents the LLM's planning phase from hallucinating different queries.
        User says "search for AI news" -> searches for "AI news" exactly.
        Results are then synthesized by the LLM for better presentation.

        Args:
            message: The user's message
            synthesize: Whether to use LLM to synthesize results

        Returns:
            Formatted search results if search request, None otherwise
        """
        message_lower = message.lower().strip()

        # If user wants comprehensive/detailed research, let it go through full agent loop
        comprehensive_keywords = ['comprehensive', 'detailed', 'in-depth', 'thorough', 'deep dive', 'extensive', 'full analysis']
        if any(kw in message_lower for kw in comprehensive_keywords):
            return None  # Let full agent loop handle this

        # Strip common greeting/name prefixes to allow "hey aura search for X"
        prefix_patterns = [
            r'^(?:hey\s+)?(?:aura|assistant|ai|bot)[,!.]?\s*',
            r'^(?:hi|hello|hey)[,!.]?\s*',
            r'^(?:okay|ok|yo)[,!.]?\s*',
            r'^(?:alright|sure|yeah|yep|yes)[,!.]?\s*',
            r'^(?:let\'?s|lets|can\s+you|could\s+you|please|pls)[,!.]?\s*',
            r'^(?:i\s+want\s+(?:you\s+)?to|i\s+need\s+(?:you\s+)?to)[,!.]?\s*',
            r'^(?:go\s+ahead\s+and|now)[,!.]?\s*',
        ]
        for prefix in prefix_patterns:
            message_lower = re.sub(prefix, '', message_lower, flags=re.IGNORECASE).strip()

        # Keywords that indicate "search online" intent
        online_keywords = ['online', 'web', 'internet', 'google', 'latest', 'current', 'recent', 'news', 'today']

        # Check if this is an ambiguous research request (no topic or unclear intent)
        ambiguous_patterns = [
            r'^(?:do\s+)?(?:a\s+)?research$',
            r'^(?:do\s+)?(?:a\s+)?(?:deep\s+)?search$',
            r'^(?:can\s+you\s+)?research$',
            r'^(?:please\s+)?research$',
            r'^look\s+(?:something\s+)?up$',
            r'^find\s+(?:something|info|information)$',
        ]

        for pattern in ambiguous_patterns:
            if re.match(pattern, message_lower, re.IGNORECASE):
                return ("I'd be happy to help with research! 🔍\n\n"
                        "**What would you like me to do?**\n"
                        "1. **Search online** - Get the latest info from the web\n"
                        "2. **Use my knowledge** - Answer from what I already know\n\n"
                        "Just tell me the topic! For example:\n"
                        "- \"Search online for quantum computing\"\n"
                        "- \"Tell me about quantum computing\"\n"
                        "- \"Research latest AI news online\"")

        # Patterns for EXPLICIT ONLINE search requests
        search_patterns = [
            # Direct search commands
            r'^search\s+(?:online\s+)?(?:the\s+web\s+)?(?:for\s+)?["\']?(.+?)["\']?$',
            r'^(?:web\s+)?search[:\s]+["\']?(.+?)["\']?$',
            r'^look\s+up\s+["\']?(.+?)["\']?$',
            r'^google\s+["\']?(.+?)["\']?$',
            r'^find\s+(?:online|on the web)\s+["\']?(.+?)["\']?$',
            r'^search\s+for\s+["\']?(.+?)["\']?[.,!?]?$',
            # Flexible patterns
            r'^do\s+(?:a\s+)?(?:deep\s+)?search\s+(?:online\s+)?(?:for\s+|about\s+|on\s+)?["\']?(.+?)["\']?$',
            r'^(?:please\s+)?(?:can\s+you\s+)?search\s+(?:online\s+)?(?:for\s+|about\s+)?["\']?(.+?)["\']?$',
            r'^(?:deep\s+)?search\s+(?:online\s+)?(?:for\s+|about\s+|on\s+)?["\']?(.+?)["\']?$',
            # Research with online intent
            r'^research\s+(?:online\s+)?(?:about\s+|on\s+)?["\']?(.+?)["\']?\s+online$',
            r'^research\s+online\s+(?:about\s+|on\s+|for\s+)?["\']?(.+?)["\']?$',
            # News/latest patterns (always online)
            r'^(?:get|find|show)\s+(?:me\s+)?(?:the\s+)?(?:latest|recent|current)\s+(?:news\s+)?(?:on|about|for)\s+["\']?(.+?)["\']?$',
            r'^what(?:\'s|\s+is)\s+(?:the\s+)?(?:latest|recent|current)\s+(?:news\s+)?(?:on|about)\s+["\']?(.+?)["\']?$',
            # Lookup patterns
            r'^look\s+(?:this\s+)?up[:\s]+["\']?(.+?)["\']?$',
            r'^(?:can\s+you\s+)?(?:please\s+)?look\s+up\s+["\']?(.+?)["\']?$',
            # Find info patterns
            r'^find\s+(?:me\s+)?(?:info|information)\s+(?:on|about)\s+["\']?(.+?)["\']?$',
            r'^get\s+(?:me\s+)?(?:info|information)\s+(?:on|about)\s+["\']?(.+?)["\']?$',
        ]

        # Extract the search query
        query = None
        for pattern in search_patterns:
            match = re.match(pattern, message_lower, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                # Remove trailing punctuation
                query = re.sub(r'[.,!?]+$', '', query).strip()
                break

        # If no explicit pattern matched, check for "research X" with online keywords
        if not query:
            research_match = re.match(r'^(?:do\s+)?(?:a\s+)?research\s+(?:about\s+|on\s+)?["\']?(.+?)["\']?$', message_lower)
            if research_match:
                potential_query = research_match.group(1).strip()
                # Check if any online keyword is present
                if any(kw in message_lower for kw in online_keywords):
                    query = potential_query
                else:
                    # Ambiguous - has topic but unclear if online or knowledge
                    return (f"I can help you research **{potential_query}**! 🔍\n\n"
                            f"Would you like me to:\n"
                            f"1. **Search online** - \"search online for {potential_query}\"\n"
                            f"2. **Use my knowledge** - \"tell me about {potential_query}\"\n\n"
                            f"Which would you prefer?")

        if not query:
            return None  # Not an explicit search request

        # Check if web_search tool is available
        if 'web_search' not in self.tools:
            return "Web search tool not available."

        logger.debug(f"[DIRECT SEARCH] User query: '{query}'")

        try:
            # Call web search directly with the exact user query
            tool = self.tools['web_search']
            result = tool.search(query, num_results=5)

            if not result.get("success"):
                return f"Search failed: {result.get('error', 'Unknown error')}"

            results = result.get("results", [])
            if not results:
                return f"No results found for '{query}'."

            # Format raw results
            raw_results = ""
            for i, r in enumerate(results[:5], 1):
                title = r.get("title", "No title")
                snippet = r.get("snippet", "No description")
                url = r.get("url", "")
                raw_results += f"{i}. {title}\n   {snippet}\n   URL: {url}\n\n"

            # Synthesize with LLM if available and requested
            if synthesize and hasattr(self, 'brain'):
                try:
                    synthesis_prompt = f"""Based on these web search results for '{query}', provide a helpful summary:

{raw_results}

Instructions:
- Summarize the key information from these results
- Include relevant URLs as references
- Keep it concise but informative
- Format nicely with markdown"""

                    synthesized = self.brain.think(synthesis_prompt)
                    if synthesized and len(synthesized) > 50:
                        return synthesized
                except Exception as e:
                    logger.debug(f"[DIRECT SEARCH] Synthesis failed, returning raw: {e}")

            # Fallback to formatted raw results
            formatted = f"Here's what I found for '{query}':\n\n"
            for i, r in enumerate(results[:5], 1):
                title = r.get("title", "No title")
                snippet = r.get("snippet", "No description")
                url = r.get("url", "")
                formatted += f"{i}. **{title}**\n   {snippet}\n   {url}\n\n"

            return formatted.strip()

        except Exception as e:
            logger.debug(f"[DIRECT SEARCH] Error: {e}")
            return f"Search error: {e}"

    def _handle_direct_crypto(self, message: str) -> Optional[str]:
        """Handle crypto price requests directly, bypassing agent loop.

        This prevents the LLM from hallucinating crypto prices.
        User says "BTC price" -> fetches real BTC price from API.

        Args:
            message: The user's message

        Returns:
            Formatted price info if crypto request, None otherwise
        """
        message_lower = message.lower().strip()

        # Crypto symbols and names mapping
        crypto_map = {
            'btc': 'bitcoin', 'bitcoin': 'bitcoin',
            'eth': 'ethereum', 'ethereum': 'ethereum',
            'sol': 'solana', 'solana': 'solana',
            'ada': 'cardano', 'cardano': 'cardano',
            'doge': 'dogecoin', 'dogecoin': 'dogecoin',
            'xrp': 'ripple', 'ripple': 'ripple',
            'dot': 'polkadot', 'polkadot': 'polkadot',
            'bnb': 'binancecoin', 'binance': 'binancecoin',
            'avax': 'avalanche-2', 'avalanche': 'avalanche-2',
            'matic': 'matic-network', 'polygon': 'matic-network',
        }

        # Patterns for crypto price requests
        crypto_patterns = [
            r'(?:what(?:\'s| is) )?(?:the )?(?:current )?(?:price (?:of )?)?(\w+)\s*price',
            r'(?:what(?:\'s| is) )?(?:the )?(?:current )?price (?:of )?(\w+)',
            r'how much (?:is|does) (\w+)(?: cost)?',
            r'^(\w+)\s*price$',
            r'^price\s*(?:of\s+)?(\w+)$',
            r'(\w+) (?:price|value|cost)',
        ]

        # Try to extract crypto name
        crypto_id = None
        for pattern in crypto_patterns:
            match = re.search(pattern, message_lower)
            if match:
                potential_crypto = match.group(1).strip()
                if potential_crypto in crypto_map:
                    crypto_id = crypto_map[potential_crypto]
                    break

        if not crypto_id:
            return None  # Not a crypto price request

        # Check if crypto_price tool is available
        if 'crypto_price' not in self.tools:
            return "Crypto price tool not available."

        logger.debug(f"[DIRECT CRYPTO] Fetching price for: {crypto_id}")

        try:
            tool = self.tools['crypto_price']
            result = tool.get_price(crypto_id)

            if not result.get("success"):
                return f"Failed to get price: {result.get('error', 'Unknown error')}"

            # Format the response
            price = result.get("price", 0)
            change_24h = result.get("change_24h", 0)
            name = result.get("name", crypto_id.title())
            symbol = result.get("symbol", "").upper()

            change_emoji = "📈" if change_24h >= 0 else "📉"
            change_sign = "+" if change_24h >= 0 else ""

            formatted = f"**{name} ({symbol})** {change_emoji}\n"
            formatted += f"💰 Current Price: **${price:,.2f}**\n"
            formatted += f"📊 24h Change: {change_sign}{change_24h:.2f}%"

            return formatted

        except Exception as e:
            logger.debug(f"[DIRECT CRYPTO] Error: {e}")
            return f"Crypto price error: {e}"

    def _handle_direct_code(self, message: str) -> Optional[str]:
        """Handle code execution requests directly, bypassing agent loop.

        This ensures code is actually executed when the user asks for it.
        Handles: "calculate X", "run python for X", "execute code for X", "what is X!" (factorial)

        Args:
            message: The user's message

        Returns:
            Formatted execution result if code request, None otherwise
        """
        message_lower = message.lower().strip()

        # Check if code_executor tool is available
        if 'code_executor' not in self.tools:
            return None

        # Patterns that indicate code execution intent
        execute_patterns = [
            r'^(?:please\s+)?(?:run|execute)\s+(?:python\s+)?(?:code\s+)?(?:for|to)\s+(.+)$',
            r'^(?:please\s+)?(?:write\s+and\s+)?(?:run|execute)\s+(?:python\s+)?(?:code\s+)?(?:for|to)\s+(.+)$',
            r'^(?:please\s+)?calculate\s+(.+)$',
            r'^(?:please\s+)?compute\s+(.+)$',
            r'^what\s+is\s+(\d+)\s*[!]$',  # "what is 20!" -> factorial
            r'^(\d+)\s*[!]$',  # "20!" -> factorial
            r'^(?:please\s+)?(?:find|generate|show)\s+(?:the\s+)?(?:first\s+)?(\d+)\s+(?:fibonacci|fib)\s*(?:numbers?)?$',
            r'^(?:please\s+)?(?:fibonacci|fib)\s+(?:sequence\s+)?(?:of\s+)?(\d+)$',
        ]

        task_description = None
        code_to_run = None

        for pattern in execute_patterns:
            match = re.match(pattern, message_lower, re.IGNORECASE)
            if match:
                task_description = match.group(1).strip()
                break

        # Also check for explicit code in the message (```python ... ```)
        # When attachment context is present, only scan the user's actual request — not document content
        scan_target = message
        if '[FILE_ATTACHMENT_CONTEXT]' in message and 'User request:' in message:
            scan_target = message.split('User request:', 1)[-1]
        code_block_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', scan_target, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            code_to_run = code_block_match.group(1).strip()
            task_description = "provided code"

        if not task_description and not code_to_run:
            return None  # Not a code execution request

        logger.debug(f"[DIRECT CODE] Task: '{task_description}'")

        # Generate code if not provided
        if not code_to_run:
            # Handle common patterns directly without LLM
            if re.match(r'^\d+\s*!?$', task_description) or 'factorial' in task_description:
                # Factorial
                num_match = re.search(r'(\d+)', task_description)
                if num_match:
                    n = num_match.group(1)
                    code_to_run = f"import math\nresult = math.factorial({n})\nprint(f'{n}! = {{result}}')"

            elif 'fibonacci' in task_description or 'fib' in task_description:
                # Fibonacci
                num_match = re.search(r'(\d+)', task_description)
                n = num_match.group(1) if num_match else "10"
                code_to_run = f"""def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib[:n]

result = fibonacci({n})
print(f"First {n} Fibonacci numbers: {{result}}")"""

            elif 'prime' in task_description:
                # Prime numbers or prime check
                num_match = re.search(r'(\d+)', task_description)
                if num_match:
                    n = num_match.group(1)
                    if 'first' in task_description or 'generate' in task_description:
                        code_to_run = f"""def sieve_of_eratosthenes(limit):
    primes = []
    is_prime = [True] * (limit + 1)
    for num in range(2, limit + 1):
        if is_prime[num]:
            primes.append(num)
            for multiple in range(num * num, limit + 1, num):
                is_prime[multiple] = False
    return primes

# Generate enough primes
primes = sieve_of_eratosthenes({int(n) * 15})[:{int(n)}]
print(f"First {int(n)} prime numbers: " + str(primes))"""
                    else:
                        code_to_run = f"""def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

result = is_prime({n})
print(f"{n} is{'' if result else ' not'} a prime number")"""

            else:
                # Use LLM to generate code for complex requests
                code_prompt = f"""Write Python code to: {task_description}

Requirements:
- Include print statements to show the output
- Keep it simple and readable
- Only output the Python code, nothing else

Python code:"""

                generated = self.brain.think(code_prompt, use_history=False, task_type=TaskType.CODE)

                # Extract code from response
                code_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', generated, re.DOTALL)
                if code_match:
                    code_to_run = code_match.group(1).strip()
                else:
                    # Try to use the whole response if it looks like code
                    lines = generated.strip().split('\n')
                    code_lines = [l for l in lines if any(c in l for c in ['print', 'def ', 'import ', '=', 'for ', 'if ', 'return'])]
                    if code_lines:
                        code_to_run = '\n'.join(code_lines)
                    else:
                        code_to_run = generated.strip()

        if not code_to_run:
            return None

        # Execute the code
        try:
            tool = self.tools['code_executor']
            result = tool.execute(code_to_run)

            # Format the response
            formatted = f"**Code Execution Result**\n\n"
            formatted += f"```python\n{code_to_run}\n```\n\n"

            if result.get("success"):
                output = result.get("output", "").strip()
                if output:
                    formatted += f"**Output:**\n```\n{output}\n```"
                else:
                    formatted += "**Output:** (no output)"
            else:
                error = result.get("errors", result.get("error", "Unknown error"))
                formatted += f"**Error:**\n```\n{error}\n```"

            return formatted

        except Exception as e:
            logger.debug(f"[DIRECT CODE] Error: {e}")
            return f"Code execution error: {e}"

    # ------------------------------------------------------------------
    # Shared pre/post processing for chat() and chat_stream()
    # ------------------------------------------------------------------

    def _prepare_chat(self, message: str, speak: bool = False) -> dict:
        """Shared pre-processing for chat() and chat_stream().

        Runs monologue start, context tracking, feedback loops, fast path,
        AURA context, emotion analysis, command/handler detection, task type
        classification, memory query, tone modifier, and system prompt building.

        Returns a context dict with all gathered state.  If the dict contains
        an ``early_return`` key, the caller should yield/return that value
        immediately without calling the LLM.
        """
        ctx: dict = {}

        # Start inner monologue session
        if hasattr(self, 'monologue') and self.monologue:
            self.monologue.start_session()
            self.monologue.think("perceive", f"Received: '{message[:80]}{'...' if len(message) > 80 else ''}'")

        # Track context for UI heatmap
        try:
            from api.routes.context import track_context_from_message
            track_context_from_message(message, is_user=True)
        except Exception as e:
            logger.debug(f"[Agent] non-critical: {e}")

        # NeuroDream: check idle trigger FIRST (before resetting the timer), then record activity
        if hasattr(self, 'neurodream') and self.neurodream:
            try:
                if (self.neurodream.check_idle_trigger()
                        and self.neurodream.current_phase == SleepPhase.AWAKE):
                    self.neurodream.enter_sleep(trigger="idle")
                self.neurodream.record_activity()
            except Exception as e:
                logger.debug(f"[Agent] non-critical: {e}")

        # ===== COHERENT LOOP: Post-response feedback (Phase 3.1) =====
        self._post_response_feedback(message)

        # Handle /init-project command
        if message.strip().lower().startswith("/init-project"):
            parts = message.strip().split(None, 1)
            target_path = parts[1].strip() if len(parts) > 1 else "."
            try:
                from aura.tools.project_context import init_project
                ctx["early_return"] = init_project(target_path)
            except Exception as e:
                ctx["early_return"] = f"Failed to initialize project: {e}"
            return ctx

        # ===== AURA FAST PATH - TRY FIRST =====
        if self.use_fastpath and hasattr(self, 'fast_path_handler') and self.fast_path_handler:
            fast_response = self.fast_path_handler.try_fast_path(message)
            if fast_response:
                _record_thought("observing", f"fast path: {message[:40]}", 0.3, "agent")
                logger.debug(f"[FAST PATH] {message[:30]}... -> {fast_response[:50]}...")
                if hasattr(self, 'monologue') and self.monologue:
                    self.monologue.think("reason", "Using fast path for simple query")
                    self.monologue.think("respond", f"Fast path response ({len(fast_response)} chars)")
                if speak:
                    self._speak(fast_response)
                ctx["early_return"] = fast_response
                return ctx

        # AURA v3.0 ALIVE - Build context using ALMA/unified memory
        aura_context = None
        if self.aura_enabled:
            try:
                aura_context = self._build_aura_context(message)
            except Exception as e:
                logger.debug(f"[AURA] Input processing error: {e}")
        ctx["aura_context"] = aura_context

        # ===== COHERENT LOOP: Pre-response appraisal (Phase 3.2) =====
        self._pre_response_appraisal(message)

        # Analyze emotional state (EvoEmo - Tool #20)
        emotion_reading = self._analyze_emotion(message)
        ctx["emotion_reading"] = emotion_reading

        # Track emotional context for UI heatmap
        if emotion_reading and emotion_reading.emotion:
            try:
                from api.routes.context import track_context_from_emotion
                track_context_from_emotion(emotion_reading.emotion, emotion_reading.confidence / 100.0)
            except Exception as e:
                logger.debug(f"[Agent] non-critical: {e}")

        # Check for EvoEmo commands
        evoemo_result = self._handle_evoemo_command(message)
        if evoemo_result:
            if speak:
                self._speak(evoemo_result)
            ctx["early_return"] = evoemo_result
            return ctx

        # Check for AURA-specific commands
        if self.aura_enabled:
            aura_result = self._handle_aura_command(message)
            if aura_result:
                if speak:
                    self._speak(aura_result)
                ctx["early_return"] = aura_result
                return ctx

        # ===== DIRECT HANDLERS — bypass agent loop =====
        search_response = self._handle_direct_search(message)
        if search_response:
            if speak:
                self._speak(search_response, emotion=emotion_reading.emotion if emotion_reading else None)
            ctx["early_return"] = search_response
            return ctx

        crypto_response = self._handle_direct_crypto(message)
        if crypto_response:
            if speak:
                self._speak(crypto_response, emotion=emotion_reading.emotion if emotion_reading else None)
            ctx["early_return"] = crypto_response
            return ctx

        code_response = self._handle_direct_code(message)
        if code_response:
            if hasattr(self, 'monologue') and self.monologue:
                self.monologue.think("execute", "Running code via direct handler")
            if speak:
                self._speak(code_response, emotion=emotion_reading.emotion if emotion_reading else None)
            ctx["early_return"] = code_response
            return ctx

        # ===== TASK TYPE CLASSIFICATION =====
        is_simple = self._is_simple_query(message)
        message_lower = message.lower()

        code_patterns = [
            'calculate', 'compute', 'factorial', 'fibonacci', 'prime',
            'run code', 'execute code', 'run python', 'execute python',
            'write code', 'write a function', 'write a script', 'implement',
            'algorithm', 'sort', 'binary search', 'recursion',
            'what is', 'what\'s'
        ]
        math_patterns = ['!', '+', '-', '*', '/', '^', '**', 'squared', 'cubed', 'power of']
        is_code_task = any(p in message_lower for p in code_patterns)
        is_math_task = any(p in message for p in math_patterns) and any(c.isdigit() for c in message)

        if is_simple:
            task_type = TaskType.SIMPLE
        elif is_code_task or is_math_task:
            task_type = TaskType.CODE
        else:
            task_type = None

        ctx["is_simple"] = is_simple
        ctx["task_type"] = task_type

        # ===== UNIFIED MEMORY QUERY =====
        unified_context = ""
        if not is_simple:
            _record_thought("recalling", f"searching all memory backends for: {message[:40]}", 0.5, "memory")
            try:
                from aura.emotion.integration import get_current_pad_dict
                _umem = get_unified_memory()
                _current_pad = get_current_pad_dict()
                _mem_future = _AGENT_EXECUTOR.submit(_umem.query, message, 10, None, 0.0, _current_pad)
                try:
                    unified_results = _mem_future.result(timeout=1.5)
                except concurrent.futures.TimeoutError:
                    unified_results = []
                    logger.warning("[UnifiedMemory] Query timed out after 1.5s, proceeding without memory context")
                if unified_results:
                    from .memory.context_budget import ContextBudget
                    _ctx_budget = ContextBudget(total_tokens=3000)
                    _budget = _ctx_budget.allocate("unified", requested=_ctx_budget.remaining)
                    _per = max(200, (_budget * 4) // max(1, len(unified_results)))
                    texts = [f"- [{r.source.upper()}] {r.content[:_per]}"
                             for r in unified_results if r.content]
                    if texts:
                        unified_context = "MEMORY CONTEXT:\n" + "\n".join(texts)
                    _srcs = set(r.source for r in unified_results)
                    _record_thought("recalling", f"recalled {len(unified_results)} memories from {_srcs}", 0.7, "memory")
                    logger.debug(f"[UnifiedMemory] {len(unified_results)} results from {_srcs}")
                    try:
                        from api.routes.memory import record_memory_recall
                        record_memory_recall("unified", len(unified_results), message,
                                             [r.content[:100] for r in unified_results[:5]])
                    except Exception as e:
                        logger.debug(f"[Agent] non-critical: {e}")
                    try:
                        from api.routes.context import track_context_from_memory
                        track_context_from_memory([r.content[:100] for r in unified_results[:5]])
                    except Exception as e:
                        logger.debug(f"[Agent] non-critical: {e}")
            except Exception as e:
                logger.debug(f"[UnifiedMemory] Query error: {e}")

        # ===== TONE MODIFIER =====
        tone_modifier = None
        if aura_context and aura_context.get("tone"):
            tone_modifier = f"Respond in a {aura_context['tone']} manner."
        elif emotion_reading and emotion_reading.confidence >= 50:
            tone_modifier = get_tone_modifier(emotion_reading.emotion)
        ctx["tone_modifier"] = tone_modifier

        # AURA thinking prefix
        thinking_prefix = ""
        if aura_context and aura_context.get("thinking_prefix"):
            thinking_prefix = aura_context["thinking_prefix"] + "\n\n"
        ctx["thinking_prefix"] = thinking_prefix

        # ===== BUILD SYSTEM PROMPT ADDON =====
        context_parts = []

        # Temporal grounding
        try:
            _grounding = self._temporal_grounding()
            if _grounding:
                context_parts.append(_grounding)
        except Exception as _tg_err:
            logger.debug(f"[Agent] Temporal grounding failed: {_tg_err}")

        # Soul personality
        soul_prompt = self._get_soul_prompt()
        if soul_prompt:
            context_parts.append(f"PERSONALITY:\n{soul_prompt}")

        # User profile
        try:
            from aura.memory.user_profile import load_profile
            _profile = load_profile()
            _profile_str = _profile.to_system_prompt()
            if _profile_str:
                context_parts.append(_profile_str)
            else:
                profile_path = Path("data/memory/user_profile.md")
                if profile_path.exists():
                    profile_text = profile_path.read_text(encoding='utf-8').strip()
                    if profile_text:
                        context_parts.append(f"USER PROFILE:\n{profile_text}")
        except Exception as e:
            logger.debug(f"[Agent] non-critical: {e}")

        if unified_context:
            context_parts.append(unified_context)

        # NeuroDream learned context
        try:
            if hasattr(self, 'neurodream') and self.neurodream:
                nd_context = self.neurodream.get_learned_context_prompt()
                if nd_context:
                    context_parts.append(f"LEARNED CONTEXT (from memory consolidation):\n{nd_context}")
        except Exception as e:
            logger.debug(f"[Agent] non-critical: {e}")

        # Skill Library context
        try:
            if hasattr(self, 'skill_library') and self.skill_library:
                _skill_context = self.skill_library.get_skill_context(message)
                if _skill_context:
                    context_parts.append(f"SKILL CONTEXT:\n{_skill_context}")
                    logger.debug("[SkillLibrary] Injected skill context for: %s", message[:40])
        except Exception as e:
            logger.debug("[SkillLibrary] Skill lookup error: %s", e)

        # Thinker context — MIRROR dual-process private reflection (roadmap 3.6)
        try:
            _thinker_ctx = None
            if hasattr(self, 'thinker') and self.thinker:
                _thinker_ctx = self.thinker.get_talker_context()
            if not _thinker_ctx and hasattr(self, 'monologue') and self.monologue:
                _thinker_ctx = self.monologue.generate_thinking_context(brain=self.brain)
            if _thinker_ctx:
                context_parts.append(_thinker_ctx)
        except Exception as _thinker_err:
            logger.debug(f"[Agent] Thinker context failed: {_thinker_err}")

        system_prompt_addon = None
        if context_parts:
            system_prompt_addon = "\n\n".join(context_parts) + "\n\nUse this knowledge and memories when relevant to the conversation. Remember personal details about the user. Always address the user by their name if known."
        ctx["system_prompt_addon"] = system_prompt_addon

        # Record reasoning in monologue
        if hasattr(self, 'monologue') and self.monologue:
            self.monologue.think("reason", f"Processing query with task_type={task_type}")

        _record_thought("formulating", f"reasoning about: {message[:50]}...", 0.7, "agent")

        return ctx

    def _finalize_chat(self, message: str, response: str, ctx: dict, speak: bool = False) -> None:
        """Shared post-processing for chat() and chat_stream().

        Handles ALMA emotional update, narrative self update, TTS, KG extraction,
        memory writes, fact extraction, skill library recording, monologue end,
        thinker kickoff, and prev_message/prev_response tracking.
        """
        is_simple = ctx.get("is_simple", False)
        emotion_reading = ctx.get("emotion_reading")

        # Close the coherent loop — feed outcome back to ALMA
        try:
            self.brain.update_emotional_state(success=bool(response and len(response) > 10))
        except Exception as _alma_err:
            logger.debug(f"[Agent] ALMA emotional update failed: {_alma_err}")

        # Update narrative self-model for significant interactions (background)
        if len(response) > 200:
            try:
                from aura.narrative_self import get_narrative_self
                _AGENT_EXECUTOR.submit(get_narrative_self().update_from_interaction, message, response, self.brain)
            except Exception as _narr_err:
                logger.debug(f"[Agent] Narrative self update failed: {_narr_err}")

        # TTS
        if speak:
            self._speak(response, emotion=emotion_reading.emotion if emotion_reading else None)

        # KG entity extraction (background)
        if not is_simple and self.kg_bridge is not None:
            try:
                if len(response) > 20:
                    extraction_text = f"User: {message}\nAssistant: {response[:500]}"
                    with self._kg_queue_lock:
                        self.kg_bridge.extraction_queue.append({
                            "trace_id": f"chat_{time.time()}",
                            "content": extraction_text,
                            "surprise": 0.6,
                            "timestamp": time.time()
                        })
                        if len(self.kg_bridge.extraction_queue) >= self.kg_bridge.config.batch_size:
                            self.kg_bridge.flush()
            except Exception as e:
                logger.debug(f"[KG BRAIN] Chat entity extraction error: {e}")

        # Extract and persist user facts (name, location, role, etc.)
        if hasattr(self, 'memory_retriever') and self.memory_retriever is not None:
            try:
                self.memory_retriever._extract_facts(message)
            except Exception as e:
                logger.debug(f"[Agent] non-critical: {e}")

        # ===== Unified memory write — gated store =====
        if not is_simple and len(response) > 20:
            try:
                _get_umem = get_unified_memory
                _clean_message = message.split("\n[Screen context:")[0].strip()
                _clean_response = response.split("\n\n---\n")[0].strip() if "\n\n---\n" in response else response
                _mem_content = f"User: {_clean_message[:200]}\nAURA: {_clean_response[:400]}"
                _pad = None
                try:
                    from aura.emotion.alma_engine import get_alma_engine
                    _alma = get_alma_engine()
                    if _alma:
                        _s = _alma.get_emotional_state()
                        _pad = {"pleasure": _s.get("pleasure", 0.0),
                                "arousal": _s.get("arousal", 0.0),
                                "dominance": _s.get("dominance", 0.0)}
                except Exception as e:
                    logger.debug(f"[Agent] non-critical: {e}")
                _umem_ref = _get_umem()
                import threading as _threading
                _store_fn = getattr(_umem_ref, "store_gated", _umem_ref.store)
                def _safe_store(_fn=_store_fn, _c=_mem_content, _p=_pad):
                    try:
                        _fn(content=_c, source="conversation", importance=0.5, emotional_pad=_p)
                    except Exception as _e:
                        logger.debug("[UnifiedMemory] Background store error: %s", _e)
                _AGENT_EXECUTOR.submit(_safe_store)
            except Exception as e:
                logger.debug(f"[UnifiedMemory] Conversation store error: {e}")

        # Record interaction for skill learning (background, non-blocking)
        try:
            if hasattr(self, 'skill_library') and self.skill_library:
                _sl_ref = self.skill_library
                _sl_msg = message[:500]
                _sl_resp = response[:500]
                _AGENT_EXECUTOR.submit(
                    _sl_ref.record_interaction,
                    user_input=_sl_msg, output=_sl_resp,
                    success=True, context={}
                )
        except Exception as e:
            logger.debug("[SkillLibrary] Record interaction error: %s", e)

        # End monologue session
        if hasattr(self, 'monologue') and self.monologue:
            self.monologue.think("reflect", "Chat response completed")

        # ===== THINKER: Kick off async background reasoning (roadmap 3.6) =====
        if hasattr(self, 'thinker') and self.thinker:
            try:
                _conv_hist = self.brain.conversation_history if hasattr(self.brain, 'conversation_history') else None
                self.thinker.run_async(message, response, _conv_hist)
            except Exception as e:
                logger.debug(f"[Thinker] Async kickoff error: {e}")

        # ===== COHERENT LOOP: Track exchange for next-turn feedback =====
        self._prev_message = message
        self._prev_response = response

    # ------------------------------------------------------------------

    def chat(self, message: str, speak: bool = False) -> str:
        """Simple chat interface for one-off interactions.

        Args:
            message: User message
            speak: If True, speak the response using TTS

        Returns:
            Agent response text
        """
        # ===== SHARED PRE-PROCESSING =====
        ctx = self._prepare_chat(message, speak=speak)

        # Early return for commands, fast path, direct handlers
        if "early_return" in ctx:
            return ctx["early_return"]

        task_type = ctx["task_type"]
        tone_modifier = ctx["tone_modifier"]
        thinking_prefix = ctx["thinking_prefix"]
        system_prompt_addon = ctx["system_prompt_addon"]
        is_simple = ctx["is_simple"]

        # ===== Strategy Bandit — Adaptive Reasoning Strategy Selection =====
        bandit_selection = None
        _strategy_start = time.time()

        if STRATEGY_BANDIT_AVAILABLE and getattr(Config, 'STRATEGY_BANDIT_ENABLED', False):
            try:
                if self._is_simple_query(message):
                    selected_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT
                else:
                    bandit = get_strategy_bandit()
                    bandit_selection = bandit.select_strategy(message)
                    selected_strategy = bandit_selection.strategy
                    logger.debug(f"[StrategyBandit] selected: {selected_strategy.value} for {bandit_selection.category.value}")
            except Exception as e:
                logger.debug(f"[StrategyBandit] Selection error, falling back to CoT: {e}")
                selected_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT if ReasoningStrategy else "chain_of_thought"
        else:
            selected_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT if ReasoningStrategy else "chain_of_thought"

        # ===== Prompt Evolution Engine — Inject evolved prompt =====
        if PROMPT_EVOLUTION_AVAILABLE and getattr(Config, 'PROMPT_EVOLUTION_ENABLED', False):
            try:
                evo_engine = get_prompt_evolution_engine()
                evolved_prompt = evo_engine.get_active_prompt("reasoner")
                if evolved_prompt is None:
                    from aura.consciousness.prompt_evolution import DEFAULT_REASONER_PROMPT
                    evo_engine.seed_prompt("reasoner", DEFAULT_REASONER_PROMPT)
                    evolved_prompt = DEFAULT_REASONER_PROMPT
                if system_prompt_addon:
                    system_prompt_addon = evolved_prompt + "\n\n" + system_prompt_addon
                else:
                    system_prompt_addon = evolved_prompt
            except Exception as e:
                logger.debug(f"[PromptEvolution] Injection error: {e}")

        # ===== Reasoning Template Library — Retrieve template guidance (top-K) =====
        template_match = None       # backward compat: best match
        template_matches = []       # all top-K matches
        if TEMPLATE_LIBRARY_AVAILABLE and getattr(Config, 'REASONING_TEMPLATES_ENABLED', False):
            try:
                template_lib = get_template_library()
                category_str = bandit_selection.category.value if bandit_selection else None
                template_matches = template_lib.retrieve_templates(message, category=category_str, top_k=3)
                if template_matches:
                    template_match = template_matches[0]
                    # Inject multi-template guidance into system prompt
                    guidance = template_lib._format_guidance_multi(template_matches)
                    if guidance:
                        if system_prompt_addon:
                            system_prompt_addon = system_prompt_addon + "\n\n" + guidance
                        else:
                            system_prompt_addon = guidance
                    logger.debug(f"[TemplateLib] Injected {len(template_matches)} template(s), best: {template_match.template.name}")
            except Exception as e:
                logger.debug(f"[TemplateLib] Retrieval error: {e}")

        # Raw strategy results for rich trace capture
        _mcts_raw_result = None
        _reflexion_raw_result = None

        # Execute the selected strategy
        try:
            if selected_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
                response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

            elif selected_strategy == ReasoningStrategy.COGNITIVE_THEATER:
                if hasattr(self, 'theater') and self.theater:
                    response = self.theater.quick_debate(message)
                else:
                    response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

            elif selected_strategy == ReasoningStrategy.DEBATE:
                if hasattr(self, 'theater') and self.theater:
                    response = self.theater.quick_debate(message)
                else:
                    response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

            elif selected_strategy == ReasoningStrategy.REFLEXION:
                if hasattr(self, 'reflexion') and self.reflexion and self.reflexion_enabled:
                    try:
                        reflexion_result = self.reflexion.solve(message)
                        _reflexion_raw_result = reflexion_result
                        response = reflexion_result if isinstance(reflexion_result, str) else getattr(reflexion_result, 'final_output', str(reflexion_result))
                    except Exception as e:
                        logger.debug(f"[StrategyBandit] Reflexion error, falling back to CoT: {e}")
                        response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)
                else:
                    response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

            elif selected_strategy == ReasoningStrategy.MCTS:
                if hasattr(self, 'reasoning_tree') and self.reasoning_tree:
                    try:
                        mcts_result = self.reasoning_tree.execute("solve", problem=message)
                        _mcts_raw_result = mcts_result
                        response = mcts_result.get("answer", "") or mcts_result.get("result", "")
                        if not response:
                            response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)
                    except Exception as e:
                        logger.debug(f"[StrategyBandit] MCTS error, falling back to CoT: {e}")
                        response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)
                else:
                    response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

            else:
                # Unknown strategy — safe fallback
                response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

        except Exception as e:
            logger.debug(f"[StrategyBandit] Strategy execution error, falling back to CoT: {e}")
            response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

        # Record response generation in monologue
        if hasattr(self, 'monologue') and self.monologue:
            self.monologue.think("respond", f"Generated response ({len(response)} chars)")

        if thinking_prefix:
            response = thinking_prefix + response

        # ===== Strategy Bandit — Record Outcome =====
        composite_reward = 0.5  # Default if bandit is skipped
        if bandit_selection is not None and STRATEGY_BANDIT_AVAILABLE:
            try:
                _strategy_latency = (time.time() - _strategy_start) * 1000  # ms
                bandit = get_strategy_bandit()
                metrics = {}

                # Async LLM-based evaluation if enabled
                if getattr(Config, 'STRATEGY_BANDIT_EVAL_ENABLED', False):
                    try:
                        from aura.consciousness.reward_signals import RewardSignalCollector
                        collector = RewardSignalCollector()
                        eval_future = collector.collect_async(
                            message, response,
                            lambda prompt: self.brain.think(prompt, task_type=None),
                        )
                        # Fire-and-forget: update outcome when eval completes
                        def _on_eval_done(fut):
                            try:
                                eval_metrics = fut.result(timeout=30)
                                bandit.record_outcome(
                                    request_id=bandit_selection.request_id + "_eval",
                                    strategy=bandit_selection.strategy,
                                    category=bandit_selection.category,
                                    latency_ms=_strategy_latency,
                                    response_length=len(response),
                                    metrics=eval_metrics,
                                )
                            except Exception as ex:
                                logger.debug(f"[StrategyBandit] Async eval error: {ex}")
                        eval_future.add_done_callback(_on_eval_done)
                    except Exception as e:
                        logger.debug(f"[StrategyBandit] Eval setup error: {e}")

                # Always record basic outcome with latency
                composite_reward = bandit.record_outcome(
                    request_id=bandit_selection.request_id,
                    strategy=bandit_selection.strategy,
                    category=bandit_selection.category,
                    latency_ms=_strategy_latency,
                    response_length=len(response),
                    metrics=metrics,
                )
            except Exception as e:
                composite_reward = 0.5
                logger.debug(f"[StrategyBandit] Outcome recording error: {e}")

        # ===== Prompt Evolution Engine — Record invocation =====
        if PROMPT_EVOLUTION_AVAILABLE and getattr(Config, 'PROMPT_EVOLUTION_ENABLED', False):
            try:
                evo_engine = get_prompt_evolution_engine()
                _failure = "low_quality" if composite_reward < 0.4 else None
                evo_engine.record_invocation("reasoner", composite_reward, failure_type=_failure)
            except Exception as e:
                logger.debug(f"[PromptEvolution] Record error: {e}")

        # ===== Reasoning Template Library — Collect trace + record usage =====
        if TEMPLATE_LIBRARY_AVAILABLE and getattr(Config, 'REASONING_TEMPLATES_ENABLED', False):
            try:
                template_lib = get_template_library()
                _cr = composite_reward if bandit_selection is not None else 0.5

                # Collect high-reward traces (strategy-aware)
                if _cr > 0.8 and bandit_selection is not None:
                    strategy_name = bandit_selection.strategy.value
                    # Build rich trace for MCTS / Reflexion; simple trace for others
                    try:
                        if strategy_name == "mcts" and _mcts_raw_result is not None:
                            full_trace = build_trace_from_mcts(_mcts_raw_result)
                        else:
                            full_trace = json.dumps([
                                {"step": "problem_understanding", "content": message[:500]},
                                {"step": "reasoning", "content": response[:1000]},
                            ])
                    except Exception:
                        full_trace = json.dumps([
                            {"step": "problem_understanding", "content": message[:500]},
                            {"step": "reasoning", "content": response[:1000]},
                        ])
                    template_lib.collect_trace(
                        request_id=bandit_selection.request_id,
                        problem=message,
                        category=bandit_selection.category.value,
                        strategy=strategy_name,
                        full_trace=full_trace,
                        reward=_cr,
                    )

                # Record template usage for all injected templates
                for _tm in template_matches:
                    template_lib.record_template_usage(
                        _tm.template.template_id,
                        _cr,
                    )
            except Exception as e:
                logger.debug(f"[TemplateLib] Trace/usage recording error: {e}")

        # ===== SHARED POST-PROCESSING =====
        self._finalize_chat(message, response, ctx, speak=speak)

        return response

    def chat_stream(self, message: str, speak: bool = False):
        """Streaming chat interface that yields response chunks in real-time.

        Pre-processes (fast path, emotion, context) via _prepare_chat(),
        then streams LLM output via brain.think_stream(), and runs
        post-processing via _finalize_chat().

        Args:
            message: User message
            speak: If True, speak the final response using TTS

        Yields:
            str: Response text chunks as they arrive
        """
        # Reload custom tools if marketplace installed something
        try:
            _sentinel = Path(__file__).parent / "tools" / "custom" / ".reload_needed"
            if _sentinel.exists():
                from .tools.custom_loader import load_custom_tools
                for _name, _inst in load_custom_tools().items():
                    if _name not in self.tools:
                        self.tools[_name] = _inst
                _sentinel.unlink(missing_ok=True)
                logger.info("[Agent] Reloaded custom tools after marketplace install")
        except Exception as _e:
            logger.warning(f"[Agent] Custom tool reload check failed: {_e}")

        # ===== SHARED PRE-PROCESSING =====
        ctx = self._prepare_chat(message, speak=speak)

        # Early return for commands, fast path, direct handlers — yield as single chunk
        if "early_return" in ctx:
            yield ctx["early_return"]
            return

        task_type = ctx["task_type"]
        tone_modifier = ctx["tone_modifier"]
        system_prompt_addon = ctx["system_prompt_addon"]

        # ===== STREAMING LLM RESPONSE =====
        full_response = ""
        for chunk in self.brain.think_stream(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon):
            full_response += chunk
            yield chunk

        # ===== SHARED POST-PROCESSING =====
        self._finalize_chat(message, full_response, ctx, speak=speak)

    def create_plan(self, task: str) -> dict:
        """Create an execution plan without acting.

        Asks the LLM to plan using the available tools list.

        Args:
            task: The task description to plan for

        Returns:
            dict with keys: task, steps (list), tools (list), complexity (str)
        """
        # Build tools list for the planner
        tool_names = list(self.tools.keys())
        tools_desc = ", ".join(tool_names[:30])  # Limit for prompt size

        plan_prompt = (
            f"You are a planning assistant. Create a step-by-step execution plan for this task. "
            f"Do NOT execute anything — only plan.\n\n"
            f"Available tools: {tools_desc}\n\n"
            f"Task: {task}\n\n"
            f"Respond in this exact format:\n"
            f"COMPLEXITY: simple|medium|complex\n"
            f"TOOLS: tool1, tool2, tool3\n"
            f"STEPS:\n"
            f"1. First step\n"
            f"2. Second step\n"
            f"3. Third step\n"
        )

        raw = self.brain._quick_generate(plan_prompt)

        # Parse response
        steps = []
        tools = []
        complexity = "medium"

        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("COMPLEXITY:"):
                complexity = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("TOOLS:"):
                tools = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
            elif line and line[0].isdigit() and "." in line[:4]:
                # Numbered step like "1. Do something"
                step_text = line.split(".", 1)[1].strip() if "." in line else line
                steps.append(step_text)

        # Fallback if parsing failed — use entire response as single step
        if not steps:
            steps = [line.strip() for line in raw.split("\n") if line.strip() and not line.startswith(("COMPLEXITY", "TOOLS"))]

        return {
            "task": task,
            "steps": steps,
            "tools": tools,
            "complexity": complexity,
        }

    def _analyze_emotion(self, message: str):
        """Analyze emotional state from user message."""
        try:
            if "evoemo" in self.tools and self.tools["evoemo"].is_enabled():
                return self.tools["evoemo"].analyze_text(message)
        except Exception as e:
            logger.debug(f"[EvoEmo] Analysis error: {e}")
        return None

    def _get_soul_prompt(self) -> str:
        """Get soul's system prompt addition for personality injection."""
        if self._soul is None:
            return ""
        try:
            return self._soul.get_system_prompt_addition()
        except Exception:
            return ""

    def _temporal_grounding(self) -> Optional[str]:
        """Build temporal grounding context if this is a new session.

        Detects session start (>5 min since last interaction), loads narrative
        self-model, calculates time elapsed, returns grounding context.
        """
        with self._temporal_lock:
            now = time.time()
            last = getattr(self, '_last_interaction_ts', 0)
            self._last_interaction_ts = now

        if last == 0:
            # First call ever — skip grounding
            return None

        elapsed_minutes = (now - last) / 60
        if elapsed_minutes < 5:
            # Same session — no grounding needed
            return None

        # This is a session start
        elapsed_hours = elapsed_minutes / 60
        parts = []

        # Time awareness
        if elapsed_hours < 1:
            parts.append(f"It's been about {int(elapsed_minutes)} minutes since we last talked.")
        elif elapsed_hours < 24:
            parts.append(f"It's been about {elapsed_hours:.0f} hours since we last talked.")
        else:
            days = elapsed_hours / 24
            parts.append(f"It's been about {days:.0f} days since we last talked.")

        # Load narrative relationship state
        try:
            from aura.narrative_self import get_narrative_self
            narrative = get_narrative_self()
            if narrative.relationship_state:
                parts.append(narrative.relationship_state)
        except Exception:
            pass

        # Load dream insights from last sleep cycle (Phase 4)
        try:
            _project_root = Path(__file__).resolve().parent.parent
            dream_queue = _project_root / "data" / "neurodream" / "dream_proactive_queue.json"
            if dream_queue.exists():
                import json as _json
                queue_data = _json.loads(dream_queue.read_text(encoding='utf-8'))
                dream_msgs = queue_data.get("messages", [])
                if dream_msgs:
                    dream_text = "\n".join(
                        f"- [{m['type']}] {m['content']}" for m in dream_msgs[:3]
                    )
                    parts.append(
                        "Thoughts from my last sleep cycle:\n" + dream_text
                    )
                # Clear the queue so it doesn't repeat
                dream_queue.unlink(missing_ok=True)
        except Exception:
            pass

        if not parts:
            return None

        return "SESSION CONTEXT:\n" + " ".join(parts)

    def _build_aura_context(self, message: str) -> dict:
        """Build AURA context using ALMA and unified memory."""
        context = {"mood": "neutral", "tone": None, "memory_context": "", "thinking_prefix": ""}
        try:
            # Get mood from ALMA if available — drives both mood label and tone.
            # tone=None lets _build_full_system_prompt use get_emotional_style_prompt()
            # which reads live ALMA state instead of a hardcoded "warm".
            from aura.emotion.alma_engine import get_alma_engine
            alma = get_alma_engine()
            if alma:
                state = alma.get_emotional_state()
                context["mood"] = state.get("dominant_emotion", "neutral")
        except Exception as e:
            logger.debug(f"[Agent] non-critical: {e}")
        # Generate thinking prefix via VisibleThinking
        if self._visible_thinking:
            try:
                prefix = self._visible_thinking.generate_thinking_prefix(message)
                if prefix:
                    context["thinking_prefix"] = prefix
            except Exception as e:
                logger.debug(f"[Agent] non-critical: {e}")
        return context

    # =================================================================
    # Coherent Loop — Phase 3.1: Pre-response appraisal & post-response feedback
    # =================================================================

    def _pre_response_appraisal(self, message: str) -> None:
        """Run chain-of-emotion appraisal BEFORE generating a response.

        Calls the fast model to ask "how would I naturally feel about this
        message?" and feeds the result into ALMA so the mood is updated
        before the response style prompt is generated.

        Must be synchronous (with a short timeout) so the mood is ready
        by the time the response generation starts.
        """
        try:
            from aura.emotion.integration import appraise_message
            result = appraise_message(message, self.brain)
            if result:
                _record_thought(
                    "observing",
                    f"emotional appraisal: {result.get('emotion', '?')} "
                    f"(intensity={result.get('intensity', 0):.1f})",
                    0.4, "emotion",
                )
        except Exception as e:
            logger.debug("[ALMA] Pre-response appraisal error: %s", e)

    def _post_response_feedback(self, current_message: str) -> None:
        """Analyze the user's new message as a reaction to our previous response.

        Closes the coherent loop: response outcome feeds back into ALMA so
        the mood drifts based on how the user actually reacted.  Runs only
        when there is a previous exchange to compare against.
        """
        if not self._prev_message or not self._prev_response:
            return
        try:
            from aura.emotion.integration import analyze_user_reaction
            result = analyze_user_reaction(
                current_message, self._prev_response, self.brain,
            )
            if result:
                _record_thought(
                    "reflecting",
                    f"user reaction: {result.get('emotion', '?')} "
                    f"(sat={result.get('satisfaction', 0):.1f} "
                    f"eng={result.get('engagement', 0):.1f})",
                    0.4, "emotion",
                )
        except Exception as e:
            logger.debug("[ALMA] Post-response feedback error: %s", e)

    def _handle_aura_command(self, message: str) -> Optional[str]:
        """Handle AURA system commands (legacy AURAEngine removed; uses ALMA/unified memory)."""
        message_lower = message.lower()

        aura_commands = [
            "aura status", "aura mood", "aura soul", "aura memory",
            "aura patterns", "aura insights", "remember this",
            "aura remember", "what do you remember"
        ]

        if not any(cmd in message_lower for cmd in aura_commands):
            return None

        try:
            if "status" in message_lower:
                tool_count = len(self.tools)
                name = self.identity.get('name', 'AURA')
                return f"AURA Status:\n- Name: {name}\n- Tools: {tool_count} loaded\n- Status: Online"

            elif "mood" in message_lower:
                evoemo = self.tools.get("evoemo")
                if evoemo and hasattr(evoemo, 'get_state'):
                    state = evoemo.get_state()
                    return f"Current mood: {state.get('dominant_emotion', 'neutral')}"
                return "Mood system active. Feeling ready!"

            elif "soul" in message_lower:
                name = self.identity.get('name', 'AURA')
                personality = self.identity.get('personality', 'friendly and helpful')
                return f"My Identity:\n- Name: {name}\n- Personality: {personality}"

            elif "remember this" in message_lower or "aura remember" in message_lower:
                fact = message.replace("remember this:", "").replace("aura remember:", "").strip()
                fact = fact.replace("remember this", "").replace("aura remember", "").strip()
                if fact:
                    try:
                        self.memory.store(content=fact, source="user_fact", importance=0.7)
                        return f"Got it, I'll remember: '{fact[:50]}...'"
                    except Exception:
                        return "I couldn't store that memory."
                return "What would you like me to remember?"

            elif "memory" in message_lower or "what do you remember" in message_lower:
                return "Memory system active. I store and recall conversations automatically."

            elif "patterns" in message_lower or "insights" in message_lower:
                return "Pattern detection is handled by ALMA and EvoEmo subsystems."

        except Exception as e:
            logger.debug(f"[AURA] Command error: {e}")
            return f"AURA command error: {e}"

        return None

    def _handle_evoemo_command(self, message: str) -> Optional[str]:
        """Handle EvoEmo-specific commands."""
        message_lower = message.lower()

        evoemo_commands = [
            "my mood", "how am i feeling", "current mood", "mood status",
            "mood history", "emotion history", "clear mood", "disable mood",
            "enable mood", "mood patterns"
        ]

        if not any(cmd in message_lower for cmd in evoemo_commands):
            return None

        try:
            evoemo = self.tools.get("evoemo")
            if not evoemo:
                return None

            if "clear" in message_lower:
                result = evoemo.clear_history()
                return "Mood history cleared." if result.get("success") else "Failed to clear history."

            elif "disable" in message_lower:
                evoemo.set_enabled(False)
                return "Mood tracking disabled."

            elif "enable" in message_lower:
                evoemo.set_enabled(True)
                return "Mood tracking enabled."

            elif "history" in message_lower:
                history = evoemo.get_history(days=7)
                if not history:
                    return "No mood history yet."
                # Summarize
                from collections import Counter
                emotions = [h["emotion"] for h in history]
                dist = Counter(emotions)
                summary = ", ".join(f"{e}: {c}" for e, c in dist.most_common())
                return f"Mood history (7 days, {len(history)} readings): {summary}"

            elif "pattern" in message_lower:
                patterns = evoemo.get_patterns()
                if patterns.get("status") == "insufficient_data":
                    return f"Not enough data for patterns yet ({patterns.get('readings', 0)} readings)."
                dominant = patterns.get("dominant_emotion", "calm")
                stress_hours = patterns.get("stress_hours", [])
                stress_info = f" Stress tends to peak around: {stress_hours}" if stress_hours else ""
                return f"Your dominant mood: {dominant}.{stress_info}"

            else:
                # Current mood
                mood = evoemo.get_current_mood()
                if mood:
                    emoji = evoemo.get_mood_emoji()
                    return f"Current mood: {emoji} {mood.emotion} ({mood.confidence}% confidence)"
                return "No mood data yet. Keep chatting and I'll pick up on how you're feeling."

        except Exception as e:
            logger.debug(f"[EvoEmo] Command error: {e}")
            return None

    def get_current_mood(self):
        """Get current emotional state (for external use)."""
        try:
            if "evoemo" in self.tools:
                return self.tools["evoemo"].get_current_mood()
        except (AttributeError, KeyError, TypeError):
            pass  # EvoEmo tool not properly initialized
        return None

    def get_mood_emoji(self) -> str:
        """Get emoji for current mood (for GUI)."""
        try:
            if "evoemo" in self.tools:
                return self.tools["evoemo"].get_mood_emoji()
        except (AttributeError, KeyError, TypeError):
            pass  # EvoEmo tool not properly initialized
        return "😐"

    # =========================================================================
    #                    KNOWLEDGE GRAPH BRAIN METHODS
    # =========================================================================

    def get_kg_brain_stats(self) -> dict:
        """Get Knowledge Graph Brain statistics.

        Returns:
            dict with KG Brain stats, or empty dict if not available
        """
        if self.kg_brain is None:
            return {"available": False, "reason": "KG Brain not initialized"}

        try:
            kg_stats = self.kg_brain.get_statistics()
            bridge_stats = self.kg_bridge.get_statistics() if self.kg_bridge else {}

            return {
                "available": True,
                "total_entities": kg_stats.get("total_entities", 0),
                "total_relationships": kg_stats.get("total_relationships", 0),
                "entity_types": kg_stats.get("entity_type_distribution", {}),
                "average_importance": kg_stats.get("average_importance", 0),
                "entities_extracted": bridge_stats.get("total_entities_extracted", 0),
                "extractions_triggered": bridge_stats.get("total_extractions_triggered", 0),
                "queue_size": bridge_stats.get("queue_size", 0)
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def kg_brain_query(self, query: str, max_entities: int = 10) -> str:
        """Query the Knowledge Graph Brain.

        Args:
            query: Search query
            max_entities: Maximum entities to return

        Returns:
            Formatted context string with matching entities
        """
        if self.kg_query_engine is None:
            return "Knowledge Graph Brain not available."

        try:
            result = self.kg_query_engine.query(query, mode=QueryMode.HYBRID, max_entities=max_entities)
            return result.context_string if result.entities else "No matching entities found."
        except Exception as e:
            return f"Query error: {e}"

    def kg_brain_add_knowledge(self, text: str, context: str = "manual") -> dict:
        """Manually add knowledge to the KG Brain.

        Args:
            text: Text to extract entities from
            context: Context for the extraction

        Returns:
            dict with extraction results
        """
        if self.kg_bridge is None:
            return {"success": False, "error": "KG Brain not available"}

        try:
            entity_ids = self.kg_bridge.force_extract(text, context=context)
            return {
                "success": True,
                "entities_extracted": len(entity_ids),
                "entity_ids": entity_ids
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def kg_brain_consolidate(self, decay_rate: float = 0.01, prune_threshold: float = 0.05) -> dict:
        """Consolidate KG Brain memory (decay importance, prune low entities).

        Args:
            decay_rate: Rate of importance decay
            prune_threshold: Threshold below which to prune entities

        Returns:
            dict with consolidation results
        """
        if self.kg_brain is None:
            return {"success": False, "error": "KG Brain not available"}

        try:
            # Flush any pending extractions
            if self.kg_bridge:
                self.kg_bridge.flush()

            # Get stats before
            stats_before = self.kg_brain.get_statistics()

            # Apply decay
            self.kg_brain.decay_importance(decay_rate)

            # Prune low importance
            self.kg_brain.prune_low_importance(prune_threshold)

            # Get stats after
            stats_after = self.kg_brain.get_statistics()

            return {
                "success": True,
                "entities_before": stats_before.get("total_entities", 0),
                "entities_after": stats_after.get("total_entities", 0),
                "entities_pruned": stats_before.get("total_entities", 0) - stats_after.get("total_entities", 0),
                "decay_rate": decay_rate,
                "prune_threshold": prune_threshold
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    #                    EPISODIC TIME-TRAVEL MEMORY METHODS
    # =========================================================================

    def get_episodic_memory_stats(self) -> dict:
        """Get Episodic Memory statistics.

        Returns:
            dict with episodic memory stats, or empty dict if not available
        """
        if self.episodic_memory is None:
            return {"available": False, "reason": "Episodic Memory not initialized"}

        try:
            store_stats = self.episodic_memory.get_statistics()
            bridge_stats = self.episodic_bridge.get_statistics() if self.episodic_bridge else {}

            return {
                "available": True,
                "total_episodes": store_stats.get("total_episodes", 0),
                "vector_dimension": store_stats.get("vector_dimension", 0),
                "episodes_formed": bridge_stats.get("episodes_formed", 0),
                "session_id": bridge_stats.get("session_id", None),
                "context_retrievals": bridge_stats.get("context_retrievals", 0)
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def episodic_recall(self, query: str, limit: int = 5, time_filter: str = None) -> list:
        """Recall episodic memories matching a query.

        Args:
            query: Search query text
            limit: Maximum results to return
            time_filter: Optional natural language time filter (e.g., "yesterday", "last week")

        Returns:
            List of matching episodes with metadata
        """
        if self.episodic_memory is None:
            return []

        try:
            from aura_episodic_memory import TemporalParser

            # Parse time filter if provided
            start_time = None
            end_time = None
            if time_filter:
                parser = TemporalParser()
                time_range = parser.parse(time_filter)
                if time_range:
                    start_time = time_range.start
                    end_time = time_range.end

            search_query = EpisodeQuery(
                query_text=query,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )

            results = self.episodic_memory.search(search_query)

            return [
                {
                    "id": r.episode.id,
                    "content": r.episode.content[:300],
                    "type": r.episode.episode_type.value,
                    "timestamp": r.episode.temporal_context.timestamp.isoformat(),
                    "importance": r.episode.importance,
                    "score": r.score,
                    "entities": r.episode.entities_involved[:5]
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Episodic recall error: {e}")
            return []

    def episodic_time_travel(self, time_reference: str) -> dict:
        """Time travel to a point in memory.

        Args:
            time_reference: Natural language time reference (e.g., "yesterday afternoon", "last week")

        Returns:
            dict with episodes and narrative from that time
        """
        if self.episodic_timeline is None:
            return {"success": False, "error": "Episodic Memory not available"}

        try:
            episodes, narrative = self.episodic_timeline.time_travel(time_reference)

            return {
                "success": True,
                "time_reference": time_reference,
                "episode_count": len(episodes),
                "narrative": narrative,
                "episodes": [
                    {
                        "id": ep.id,
                        "title": ep.title or ep.content[:50],
                        "type": ep.episode_type.value,
                        "timestamp": ep.temporal_context.timestamp.isoformat()
                    }
                    for ep in episodes[:10]
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def episodic_record(self, content: str, episode_type: str = "conversation",
                        importance: float = 0.5, entities: list = None,
                        tools_used: list = None) -> dict:
        """Manually record an episode to memory.

        Args:
            content: Episode content
            episode_type: Type of episode (conversation, task_execution, learning, etc.)
            importance: Importance score (0-1)
            entities: List of entities involved
            tools_used: List of tools used

        Returns:
            dict with result of recording
        """
        if self.episodic_memory is None:
            return {"success": False, "error": "Episodic Memory not available"}

        try:
            # Map string to EpisodeType
            type_map = {
                "conversation": EpisodeType.CONVERSATION,
                "task_execution": EpisodeType.TASK_EXECUTION,
                "learning": EpisodeType.LEARNING,
                "error": EpisodeType.ERROR,
                "milestone": EpisodeType.MILESTONE,
                "insight": EpisodeType.INSIGHT,
                "user_preference": EpisodeType.USER_PREFERENCE,
                "system_event": EpisodeType.SYSTEM_EVENT
            }
            ep_type = type_map.get(episode_type, EpisodeType.CONVERSATION)

            episode = Episode(
                content=content,
                episode_type=ep_type,
                temporal_context=TemporalContext(timestamp=datetime.now()),
                importance=importance,
                entities_involved=entities or [],
                tools_used=tools_used or []
            )

            episode_id = self.episodic_memory.store_episode(episode)

            return {
                "success": True,
                "episode_id": episode_id,
                "type": ep_type.value
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def episodic_get_context(self, query: str, include_timeline: bool = False) -> str:
        """Get episodic context for a query (for LLM prompting).

        Args:
            query: Query to get context for
            include_timeline: Include recent activity summary

        Returns:
            Formatted context string
        """
        if self.episodic_bridge is None:
            return ""

        try:
            return self.episodic_bridge.get_context_for_query(query, include_timeline=include_timeline)
        except Exception as e:
            logger.error(f"Episodic context error: {e}")
            return ""

    def episodic_consolidate(self) -> dict:
        """Run memory consolidation (decay, merge, garbage collect).

        Returns:
            dict with consolidation results
        """
        if self.episodic_consolidator is None:
            return {"success": False, "error": "Episodic consolidator not available"}

        try:
            results = self.episodic_consolidator.run_full_consolidation()

            return {
                "success": True,
                "operations": [
                    {
                        "operation": r.operation,
                        "episodes_affected": r.episodes_affected,
                        "duration_seconds": r.duration_seconds
                    }
                    for r in results
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def episodic_get_health(self) -> dict:
        """Get memory health report with recommendations.

        Returns:
            dict with health metrics and recommendations
        """
        if self.episodic_consolidator is None:
            return {"status": "unavailable"}

        try:
            return self.episodic_consolidator.get_health_report()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ==================== Skill Library Methods ====================

    def get_skill_library_stats(self) -> dict:
        """Get Skill Library statistics.

        Returns:
            dict with skill library stats, or empty dict if not available
        """
        if self.skill_library is None:
            return {"available": False, "reason": "Skill Library not initialized"}

        try:
            return self.skill_library.get_stats()
        except Exception as e:
            return {"available": False, "error": str(e)}

    def skill_search(self, query: str, limit: int = 5, category: str = None) -> list:
        """Search for relevant skills.

        Args:
            query: Search query
            limit: Maximum results
            category: Optional category filter

        Returns:
            list of (skill_id, score) tuples
        """
        if self.skill_library is None:
            return []

        try:
            return self.skill_library.search(query, limit=limit, category=category)
        except Exception as e:
            logger.error(f"Skill search error: {e}")
            return []

    def skill_get(self, skill_id: str) -> dict:
        """Get a skill by ID.

        Args:
            skill_id: ID of the skill to retrieve

        Returns:
            dict with skill data, or error dict
        """
        if self.skill_library is None:
            return {"error": "Skill Library not available"}

        try:
            skill = self.skill_library.get_skill(skill_id)
            if skill:
                return skill.to_dict()
            return {"error": f"Skill not found: {skill_id}"}
        except Exception as e:
            return {"error": str(e)}

    def skill_create(
        self,
        name: str,
        description: str,
        category: str,
        trigger_patterns: list,
        procedure: str,
        tags: list = None
    ) -> dict:
        """Create a new skill.

        Args:
            name: Skill name (2-4 words)
            description: What the skill does
            category: coding, writing, research, automation, analysis, communication, learning, custom
            trigger_patterns: Phrases that trigger this skill
            procedure: Step-by-step procedure
            tags: Optional tags

        Returns:
            dict with created skill ID
        """
        if self.skill_library is None:
            return {"success": False, "error": "Skill Library not available"}

        try:
            skill_id = self.skill_library.create_skill(
                name=name,
                description=description,
                category=category,
                trigger_patterns=trigger_patterns,
                procedure=procedure,
                tags=tags
            )
            return {"success": True, "skill_id": skill_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def skill_record_use(
        self,
        skill_id: str,
        input_context: str,
        output: str,
        success: bool,
        feedback: str = None
    ) -> dict:
        """Record usage of a skill for learning.

        Args:
            skill_id: ID of the skill used
            input_context: What triggered the skill
            output: What the skill produced
            success: Whether it worked
            feedback: Optional user feedback

        Returns:
            dict with success status
        """
        if self.skill_library is None:
            return {"success": False, "error": "Skill Library not available"}

        try:
            result = self.skill_library.record_use(
                skill_id=skill_id,
                input_context=input_context,
                output=output,
                success=success,
                feedback=feedback
            )
            return {"success": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def skill_find_applicable(self, user_input: str, max_skills: int = 3) -> list:
        """Find skills applicable to a user request.

        Args:
            user_input: User's request
            max_skills: Maximum skills to return

        Returns:
            list of (skill, score) tuples
        """
        if self.skill_library is None:
            return []

        try:
            return self.skill_library.find_applicable(user_input, max_skills=max_skills)
        except Exception as e:
            logger.error(f"Skill find error: {e}")
            return []

    def skill_get_context(self, user_input: str) -> str:
        """Get skill context for LLM prompting.

        Args:
            user_input: User's request

        Returns:
            Formatted context string for LLM injection
        """
        if self.skill_library is None:
            return ""

        try:
            return self.skill_library.get_skill_context(user_input)
        except Exception as e:
            logger.error(f"Skill context error: {e}")
            return ""

    def skill_record_interaction(
        self,
        user_input: str,
        output: str,
        success: bool,
        context: dict = None,
        feedback: str = None
    ) -> str:
        """Record an interaction for potential skill learning.

        Args:
            user_input: What the user asked
            output: What was produced
            success: Whether successful
            context: Optional context
            feedback: Optional feedback

        Returns:
            skill_id if a skill was learned/updated, None otherwise
        """
        if self.skill_library is None:
            return None

        try:
            return self.skill_library.record_interaction(
                user_input=user_input,
                output=output,
                success=success,
                context=context,
                feedback=feedback
            )
        except Exception as e:
            logger.error(f"Skill record error: {e}")
            return None

    def skill_list(self, category: str = None, sort_by: str = "success_rate") -> list:
        """List all skills.

        Args:
            category: Optional category filter
            sort_by: Sort order (success_rate, uses, name, updated)

        Returns:
            list of skill info dicts
        """
        if self.skill_library is None:
            return []

        try:
            return self.skill_library.list_skills(category=category, sort_by=sort_by)
        except Exception as e:
            logger.error(f"Skill list error: {e}")
            return []

    def skill_improve(self, skill_id: str, apply: bool = False) -> dict:
        """Analyze and optionally improve a skill.

        Args:
            skill_id: Skill to improve
            apply: Whether to apply improvements

        Returns:
            Improvement suggestions
        """
        if self.skill_library is None:
            return {"error": "Skill Library not available"}

        try:
            return self.skill_library.improve_skill(skill_id, apply=apply)
        except Exception as e:
            return {"error": str(e)}

    # ==================== Life Modeling Methods ====================

    def get_life_modeling_stats(self) -> dict:
        """Get Life Modeling availability status.

        Returns:
            dict with status info
        """
        if self.life_modeling is None:
            return {"available": False, "reason": "Life Modeling not initialized"}

        return {
            "available": True,
            "mesa_available": MESA_AVAILABLE if LIFE_MODELING_AVAILABLE else False,
            "tools": [t["name"] for t in self.life_modeling.get_tools()]
        }

    def life_update_state(
        self,
        financial: dict = None,
        career: dict = None,
        health: dict = None,
        personal: dict = None
    ) -> dict:
        """Update life state model for simulations.

        Args:
            financial: dict with monthly_income, monthly_expenses, savings, etc.
            career: dict with current_role, satisfaction (0-1), is_employed
            health: dict with stress_level (0-1), age
            personal: dict with life_satisfaction (0-1), location

        Returns:
            dict with success status and current wellbeing score
        """
        if self.life_modeling is None:
            return {"success": False, "error": "Life Modeling not available"}

        try:
            params = {}
            if financial:
                params["financial"] = financial
            if career:
                params["career"] = career
            if health:
                params["health"] = health
            if personal:
                params["personal"] = personal

            return self.life_modeling.handle_tool_call("life_state_update", params)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def life_get_state(self) -> dict:
        """Get current life state model.

        Returns:
            dict with life state and wellbeing score
        """
        if self.life_modeling is None:
            return {"error": "Life Modeling not available"}

        try:
            return self.life_modeling.handle_tool_call("get_life_state", {})
        except Exception as e:
            return {"error": str(e)}

    def life_simulate_decision(
        self,
        decision_type: str,
        parameters: dict = None,
        time_horizon_years: int = 5,
        num_simulations: int = 100
    ) -> dict:
        """Simulate a life decision and see projected outcomes.

        Args:
            decision_type: career_change, quit_job, start_business, major_purchase,
                          relocation, education, have_child, retirement, lifestyle_change
            parameters: Decision-specific parameters (e.g., salary_change_pct, startup_cost)
            time_horizon_years: How many years to simulate (default 5)
            num_simulations: Number of Monte Carlo runs (default 100)

        Returns:
            dict with outcomes and risk metrics
        """
        if self.life_modeling is None:
            return {"error": "Life Modeling not available"}

        try:
            return self.life_modeling.handle_tool_call("simulate_decision", {
                "decision_type": decision_type,
                "parameters": parameters or {},
                "time_horizon_years": time_horizon_years,
                "num_simulations": num_simulations
            })
        except Exception as e:
            return {"error": str(e)}

    def life_compare_decisions(
        self,
        decisions: list,
        time_horizon_years: int = 5
    ) -> dict:
        """Compare multiple decision scenarios.

        Args:
            decisions: List of dicts with decision_type and optional parameters
            time_horizon_years: How many years to simulate

        Returns:
            dict with ranking and summaries
        """
        if self.life_modeling is None:
            return {"error": "Life Modeling not available"}

        try:
            return self.life_modeling.handle_tool_call("compare_decisions", {
                "decisions": decisions,
                "time_horizon_years": time_horizon_years
            })
        except Exception as e:
            return {"error": str(e)}

    def life_what_if(self, question: str, variables: dict = None) -> dict:
        """Answer what-if questions about life changes.

        Args:
            question: Natural language what-if question
            variables: Specific variable changes to analyze

        Returns:
            dict with analysis and interpretation
        """
        if self.life_modeling is None:
            return {"error": "Life Modeling not available"}

        try:
            return self.life_modeling.handle_tool_call("what_if_analysis", {
                "question": question,
                "variables": variables or {}
            })
        except Exception as e:
            return {"error": str(e)}

    def life_generate_report(
        self,
        decision_type: str,
        parameters: dict = None,
        format: str = "markdown"
    ) -> dict:
        """Generate a detailed decision analysis report.

        Args:
            decision_type: Type of decision to analyze
            parameters: Decision-specific parameters
            format: "markdown" or "json"

        Returns:
            dict with report in requested format
        """
        if self.life_modeling is None:
            return {"error": "Life Modeling not available"}

        try:
            return self.life_modeling.handle_tool_call("generate_decision_report", {
                "decision_type": decision_type,
                "parameters": parameters or {},
                "format": format
            })
        except Exception as e:
            return {"error": str(e)}

    def _speak(self, text: str, emotion: Optional[str] = None):
        """Speak text using TTS with optional emotional adaptation."""
        try:
            from .services.voice_presence import get_voice_presence
            vps = get_voice_presence()
            if vps._enabled:
                vps.speak(text, emotion=emotion, block=False)
        except Exception as e:
            logger.debug(f"TTS error: {e}")

    def recall_memories(self, query: str, n: int = 5) -> list:
        """Recall relevant memories."""
        results = self.memory.query(query, k=n)
        return [{"content": r.content, "score": r.score, "metadata": r.metadata} for r in results]

    def run_dream_consolidation(self) -> dict:
        """Run DreamMode memory consolidation. Call after long conversations."""
        if self.dream_mode:
            try:
                result = self.dream_mode.dream()
                logger.info(f"[DREAM] Consolidation: {result}")
                return result
            except Exception as e:
                logger.warning(f"[DREAM] Failed: {e}")
        return {"success": False, "error": "DreamMode not available"}

    def shutdown(self) -> dict:
        """Gracefully shutdown the agent and free all resources.

        This method should be called when the agent is being terminated
        to ensure proper cleanup of VRAM, file handles, and background threads.

        Returns:
            dict with shutdown status and details
        """
        results = {
            "success": True,
            "freed_resources": [],
            "errors": []
        }

        # 1. Shutdown NeuroDream if sleeping
        try:
            if hasattr(self, '_neurodream_stop_event'):
                self._neurodream_stop_event.set()
                results["freed_resources"].append("neurodream_idle_poll_thread")
            if hasattr(self, 'neurodream') and self.neurodream:
                if self.neurodream.current_phase != SleepPhase.AWAKE:
                    self.neurodream.wake_up(reason="shutdown")
                    results["freed_resources"].append("neurodream_sleep_thread")
                if hasattr(self.neurodream, 'shutdown'):
                    self.neurodream.shutdown()
        except Exception as e:
            results["errors"].append(f"NeuroDream shutdown: {e}")

        # 1.5. AURA v3.0 ALIVE system — removed (migrated to ALMA helpers)

        # 2. Unload Ollama models to free VRAM
        try:
            if hasattr(self, 'brain') and self.brain:
                unload_result = self.brain.unload_all_models()
                for model, success in unload_result.items():
                    if success:
                        results["freed_resources"].append(f"ollama:{model}")
        except Exception as e:
            results["errors"].append(f"Ollama unload: {e}")

        # 3. Close browser if open
        try:
            if "browser" in self.tools and hasattr(self.tools["browser"], 'close'):
                self.tools["browser"].close()
                results["freed_resources"].append("browser")
        except Exception as e:
            results["errors"].append(f"Browser close: {e}")

        # 4. Unload voice/TTS models
        try:
            for tool_name in ["voice", "voice_manager"]:
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    if hasattr(tool, 'unload'):
                        tool.unload()
                        results["freed_resources"].append(tool_name)
                    elif hasattr(tool, 'unload_whisper'):
                        tool.unload_whisper()
                        results["freed_resources"].append(f"{tool_name}:whisper")
        except Exception as e:
            results["errors"].append(f"Voice unload: {e}")

        # 5. Save knowledge graph
        try:
            if "knowledge_graph" in self.tools:
                self.tools["knowledge_graph"].save()
                results["freed_resources"].append("knowledge_graph:saved")
        except Exception as e:
            results["errors"].append(f"KG save: {e}")

        # 7. Save conversation history (preserve on disk, only clear in-memory)
        try:
            if hasattr(self, 'brain') and self.brain:
                self.brain._save_history()
                results["freed_resources"].append("conversation_history:saved")
        except Exception as e:
            results["errors"].append(f"History save: {e}")

        # 9. Close Knowledge Graph Brain
        try:
            if hasattr(self, 'kg_brain') and self.kg_brain:
                # Flush pending extractions first
                if self.kg_bridge:
                    self.kg_bridge.flush()
                self.kg_brain.close()
                results["freed_resources"].append("kg_brain")
        except Exception as e:
            results["errors"].append(f"KG Brain close: {e}")

        # 10. Close Episodic Memory
        try:
            if hasattr(self, 'episodic_memory') and self.episodic_memory:
                # Flush pending episodes first
                if self.episodic_bridge:
                    self.episodic_bridge.flush_pending()
                self.episodic_memory.close()
                results["freed_resources"].append("episodic_memory")
        except Exception as e:
            results["errors"].append(f"Episodic Memory close: {e}")

        # 11. Close Skill Library
        try:
            if hasattr(self, 'skill_library') and self.skill_library:
                self.skill_library.shutdown()
                results["freed_resources"].append("skill_library")
        except Exception as e:
            results["errors"].append(f"Skill Library close: {e}")

        # 12. Save ALMA emotional state for cross-session continuity
        try:
            from aura.emotion.alma_engine import save_state as alma_save_state
            alma_save_state()
            results["freed_resources"].append("alma_emotional_state:saved")
        except Exception as e:
            results["errors"].append(f"ALMA state save: {e}")

        results["success"] = len(results["errors"]) == 0
        return results

