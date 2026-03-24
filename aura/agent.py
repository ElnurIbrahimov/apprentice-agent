"""Main agent implementation with ReAct loop (single LLM call per step)."""

import json
import os
import re
import time
import logging
import threading
import concurrent.futures
from collections import deque

# Shared pool — centralized in aura.pools (2026-03-22)
from aura.pools import bg_pool as _bg_pool_fn
_AGENT_EXECUTOR = _bg_pool_fn()

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, Callable, List, Dict

logger = logging.getLogger(__name__)

# ============================================================================
# Re-exports for backward compatibility — other files import these from here
# ============================================================================
from aura.security.tool_validator import (          # noqa: F401
    validate_custom_tool_code,
    validate_script_code,
    ALLOWED_TOOL_IMPORTS,
    FORBIDDEN_PATTERNS,
)
from aura.tools.loader import (                     # noqa: F401
    _TOOL_KEYWORDS,
    _TOOL_KEYWORDS_RE,
    load_core_tools,
    load_heavy_tools,
    load_synthesized_tools,
    ensure_tool as _ensure_tool_fn,
)
from aura.tools.custom_registry import (             # noqa: F401
    load_custom_tools_from_registry,
    generate_default_keywords,
)

# ============================================================================
# Mixins — extracted domain-specific method groups (2026-03-23)
# ============================================================================
from aura.core.kg_brain import KGBrainMixin
from aura.core.skill_manager import SkillManagerMixin
from aura.core.narrative import NarrativeMixin
from aura.core.direct_handlers import DirectHandlersMixin


# Thinking system integration — re-export from shared module for backward compat
from aura.core.thought_recorder import record_thought as _record_thought

# Maximum history size to prevent memory bloat
MAX_HISTORY_SIZE = 100

# Timeout constants
AGENT_TIMEOUT = 120  # Overall agent loop timeout (2 minutes)
TOOL_TIMEOUT = 30    # Timeout for tool execution

from .brain import OllamaBrain, TaskType
from .identity import load_identity, get_identity_prompt, detect_name_change, detect_personality_change, update_name, update_personality
from .memory.unified_memory import get_unified_memory
from .metacognition import MetacognitionLogger
from .config import Config
from .tools import get_tone_modifier, get_knowledge_graph, NeuroDreamEngine, SleepPhase

# MemoryRetriever removed — consolidated into UnifiedMemory (2026-03-22)
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
class ApprenticeAgent(KGBrainMixin, SkillManagerMixin, NarrativeMixin, DirectHandlersMixin):
    """An AI agent that learns and acts using a ReAct loop (single LLM call per step).

    The main loop (run()) uses brain.react_step() for combined thought+action,
    with deterministic tool result evaluation (no LLM call for eval).

    Domain-specific methods are organized into mixins:
    - KGBrainMixin: Knowledge graph commands, stats, queries, episodic stubs
    - SkillManagerMixin: Skill CRUD, search, context, learning
    - NarrativeMixin: Emotion, soul, AURA context, coherent loop, temporal grounding
    - DirectHandlersMixin: Monologue, neurodream, git, search, crypto, code handlers
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
        except (AttributeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"[MODELS] Validation failed: {e}")

        # Skip Ollama warmup for fast init
        self.brain = OllamaBrain(warmup=not fast_init)
        self.memory = get_unified_memory()

        # Core lightweight tools (always load) — delegated to tools.loader
        self.tools = load_core_tools(brain=self.brain)

        # Heavy tools + synthesized + custom — delegated to tools.loader
        if not fast_init:
            load_heavy_tools(self.tools, brain=self.brain)
            try:
                load_synthesized_tools(self.tools)
            except (ImportError, AttributeError, OSError, ValueError) as e:
                logger.warning(f"Could not load synthesized tools: {e}")
            try:
                from .tools.custom_loader import load_custom_tools
                _custom_tools = load_custom_tools()
                for _tool_name, _tool_instance in _custom_tools.items():
                    if _tool_name not in self.tools:
                        self.tools[_tool_name] = _tool_instance
                        logger.info(f"[Agent] Registered custom tool: {_tool_name}")
            except (ImportError, AttributeError, OSError, ValueError) as _e:
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
            except (ImportError, AttributeError, TypeError, ValueError, OSError) as e:
                logger.warning(f"[Thinker] Init failed: {e}")

        # Initialize MCTS Reasoning Tree — used by Strategy Bandit for MCTS strategy
        self.reasoning_tree = None
        try:
            from aura.tools.reasoning_tree_tool import ReasoningTreeTool
            # Create LLM function adapter: MCTSReasoning expects (prompt, system_prompt) -> str
            def _mcts_llm_func(prompt: str, system_prompt: str = None) -> str:
                return self.brain.think(prompt, system_prompt=system_prompt, use_history=False)

            # Create tool executor adapter for LATS pattern:
            # MCTS nodes can invoke tools (search, code exec, file read) during expansion
            def _mcts_tool_executor(tool_name: str, tool_args: dict):
                return self._execute_tool_call(tool_name, tool_args)

            self.reasoning_tree = ReasoningTreeTool(
                llm_func=_mcts_llm_func,
                tool_executor=_mcts_tool_executor,
            )
            logger.info("[LOADED] ReasoningTreeTool — MCTS + LATS tool integration for Strategy Bandit")
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            logger.warning(f"[ReasoningTreeTool] Init failed: {e}")

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
                except Exception as e:  # Catch-all: protects daemon polling thread
                    logger.debug(f"[NeuroDream] Idle poll error: {e}")
        _nd_poll_thread = _threading.Thread(
            target=_neurodream_idle_poll, daemon=True, name="NeuroDream-IdlePoll"
        )
        _nd_poll_thread.start()
        logger.debug("[NeuroDream] Idle polling thread started (60s interval)")

        self.tools['neurodream'] = self.neurodream

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
        except (ImportError, AttributeError, OSError, ValueError) as e:
            self._soul = None
            logger.debug(f"[SKIP] Soul: {e}")

        # VisibleThinking — transparent reasoning for ThoughtStream UI
        self._visible_thinking = None
        try:
            from aura.thinking.visible_thinking import VisibleThinking
            self._visible_thinking = VisibleThinking()
            logger.info("[LOADED] VisibleThinking")
        except (ImportError, AttributeError, OSError, ValueError) as e:
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
            except (ImportError, AttributeError, TypeError, ValueError, OSError) as e:
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
                except (AttributeError, TypeError) as e:
                    logger.debug(f"[GatewayDaemon] Callback format error: {e}")
            # Only set fallback callback if none wired yet (api/main.py sets a richer one)
            if getattr(self.gateway_daemon, '_notification_callback', None) is None:
                self.gateway_daemon.set_notification_callback(_on_proactive_agent)
            logger.info("[LOADED] GatewayDaemon - Proactive intelligence system (singleton)")
        except (ImportError, AttributeError, TypeError, OSError) as e:
            logger.warning(f"[GatewayDaemon] Initialization failed: {e}")
            self.gateway_daemon = None

        # Initialize Tool RAG for dynamic tool selection
        self.tool_rag = None
        try:
            from aura.tools.tool_rag import ToolRAG
            from aura.core.tool_schemas import AGENTIC_TOOLS
            self.tool_rag = ToolRAG()
            self.tool_rag.initialize(self.tools, AGENTIC_TOOLS)
        except (ImportError, AttributeError, TypeError, ValueError, OSError) as e:
            logger.debug(f"[ToolRAG] Init failed (will use fallback): {e}")

        # Initialize Adaptive Planner (Roadmap 5.4)
        self.adaptive_planner = None
        if ADAPTIVE_PLANNER_AVAILABLE:
            try:
                self.adaptive_planner = AdaptivePlanner(
                    brain=self.brain, planning_interval=3
                )
                logger.debug("[LOADED] AdaptivePlanner — adaptive planning for complex tasks")
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"[AdaptivePlanner] Init failed: {e}")

        # Initialize ToolExecutor for ReAct loop (handles dev tools without sandbox)
        self._tool_executor = None
        try:
            from aura.core.agentic_loop import ToolExecutor
            _project_root = os.getcwd()
            try:
                from aura.core.context import find_project_root
                _project_root = find_project_root() or _project_root
            except (ImportError, OSError):
                pass
            self._tool_executor = ToolExecutor(project_root=_project_root)
            logger.debug(f"[ToolExecutor] Initialized at {_project_root}")
        except (ImportError, AttributeError, TypeError, OSError) as e:
            logger.debug(f"[ToolExecutor] Init failed: {e}")

        # CLI permission confirmation callback (set by main.py for interactive mode)
        self._cli_confirm_callback: Optional[Callable] = None
        self._approved_patterns: set = set()

        # Permission manager — set by CLI layer when permission manager is available
        self.permissions = None

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
                # NOTE: Kuzu only allows one process to hold the DB lock at a time.
                # If another process (e.g., main aura service) holds it, this will
                # fail gracefully and the agent runs without KG.
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

                # Wire KG Sync Bridge: keep NetworkX runtime KG and Kuzu persistent KG in sync
                nx_kg = self.tools.get("knowledge_graph")
                if nx_kg is not None:
                    try:
                        from aura.core.kg_sync import KGSyncBridge
                        _kg_sync = KGSyncBridge(nx_kg, self.kg_brain)
                        nx_kg.set_sync_bridge(_kg_sync)
                        _kg_sync.sync_from_kuzu()
                        logger.debug("[KGSync] Bridge wired — NetworkX <-> Kuzu sync active")
                    except Exception as _sync_err:
                        logger.debug(f"[KGSync] Bridge wiring failed (non-fatal): {_sync_err}")

                # Register KG Brain with UnifiedMemory so it's included in unified context queries
                if self.kg_bridge is not None:
                    try:
                        get_unified_memory().set_kg_brain(self.kg_bridge)
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"[KG Brain] UnifiedMemory bridge wiring failed: {e}")
            except (ImportError, AttributeError, TypeError, ValueError, OSError, RuntimeError) as e:
                logger.warning(f"[WARNING] Knowledge Graph Brain initialization failed: {e}")
                self.kg_brain = None
                self.kg_bridge = None
                self.kg_query_engine = None
        elif not KG_BRAIN_AVAILABLE:
            logger.debug("[INFO] Knowledge Graph Brain not available (install kuzu: pip install kuzu)")

        # Thread-safety locks (instance-level, not class-level)
        self._temporal_lock = threading.Lock()
        self._kg_queue_lock = threading.Lock()

        # Episodic memory consolidated into UnifiedMemory (2026-03-22)
        self.episodic_memory = None
        self.episodic_bridge = None
        self.episodic_timeline = None
        self.episodic_consolidator = None
        self.memory_retriever = None

        # Always-On Context Engine
        self.context_engine = None
        if CONTEXT_ENGINE_AVAILABLE:
            try:
                self.context_engine = AlwaysOnContextEngine(self)
                logger.debug("[ACE] Context engine initialized")
            except (AttributeError, TypeError, ValueError, OSError) as _e:
                logger.warning(f"[ACE] Failed to initialize: {_e}")

        # DreamMode — Memory consolidation and pattern analysis
        self.dream_mode = None
        if DREAM_MODE_AVAILABLE:
            try:
                self.dream_mode = DreamMode()
                logger.debug("[LOADED] DreamMode — Memory consolidation")
            except (ImportError, AttributeError, TypeError, ValueError, OSError) as e:
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
                        episodic_memory=None,  # Consolidated into UnifiedMemory
                        kg_brain=self.kg_brain
                    )

                # Get statistics
                stats = self.skill_library.get_stats()
                total_skills = stats['store']['total_skills']
                bridge_status = "with bridge" if self.skill_bridge else "library-only"
                logger.debug(f"[LOADED] Skill Library - {total_skills} skills ({bridge_status})")
            except (ImportError, AttributeError, TypeError, ValueError, OSError) as e:
                logger.warning(f"[WARNING] Skill Library initialization failed: {e}")
                self.skill_library = None
                self.skill_bridge = None
        elif not SKILL_LIBRARY_AVAILABLE:
            logger.debug("[INFO] Skill Library not available (install sentence-transformers: pip install sentence-transformers)")

        # Wire skill library into the load_skill tool for progressive loading
        if self.skill_library and "load_skill" in self.tools:
            try:
                from aura.tools.load_skill import set_skill_library
                set_skill_library(self.skill_library)
                logger.debug("[TOOLS] load_skill wired to skill library")
            except Exception as e:
                logger.warning(f"[TOOLS] Failed to wire load_skill: {e}")

        # Life modeling removed (dead code, 2026-03-22)

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
        except (ImportError, AttributeError, TypeError, OSError) as e:
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
        except (ImportError, AttributeError, TypeError) as e:
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
        return _ensure_tool_fn(self.tools, tool_name, brain=self.brain)

    def _load_custom_tools(self) -> None:
        """Load active custom tools from registry — delegated to tools.custom."""
        load_custom_tools_from_registry(self.tools, self.custom_tool_keywords)

    def _generate_default_keywords(self, name: str, description: str, functions: list) -> list[str]:
        """Generate default keywords for a custom tool if not provided."""
        return generate_default_keywords(name, description, functions)

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
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
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
        except (ImportError, AttributeError, OSError) as _e:
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
            except (ImportError, AttributeError, TypeError, ValueError) as _im_err:
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

        # ===== Multi-Agent Orchestrator — route complex multi-agent queries =====
        # Try pattern-based routing (no LLM call). If the router picks a multi-agent
        # mode (PARALLEL, SEQUENTIAL, DEBATE), delegate to the orchestrator.
        # SINGLE mode falls through to the standard brain.think() ReAct loop below.
        if self.orchestrator is not None:
            try:
                from aura.multi_agent.protocol import CollaborationMode
                _routing = self.orchestrator.router.route(goal, llm_func=None)
                if _routing.mode in (CollaborationMode.PARALLEL, CollaborationMode.SEQUENTIAL, CollaborationMode.DEBATE):
                    logger.info(
                        f"[AGENT] Routing to Multi-Agent Orchestrator: "
                        f"agents={_routing.agents}, mode={_routing.mode.value}, "
                        f"confidence={_routing.confidence:.0%}"
                    )
                    _orch_response = self.orchestrator.chat(goal)
                    return self._make_response(
                        goal, _orch_response, fast_path=False,
                        metadata={"orchestrator": True, "agents": _routing.agents, "mode": _routing.mode.value},
                    )
            except Exception as _orch_err:
                logger.warning(f"[AGENT] Orchestrator failed, falling back to direct path: {_orch_err}")

        # Always-On Context
        ace_context = ""
        if self.context_engine is not None:
            try:
                _bundle = self.context_engine.gather(goal)
                ace_context = _bundle.to_system_prompt()
            except (AttributeError, TypeError, ValueError, OSError) as _ace_err:
                logger.warning(f"[Agent] ACE context gather failed: {_ace_err}")
        context = context or {}  # NOTE: reserved for future use, not yet wired into agent loop
        self.brain._last_screenshot_path = None
        self.metacognition.start_goal(goal)
        if self.monologue:
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
            except (AttributeError, TypeError, ValueError, ConnectionError, TimeoutError) as _plan_err:
                logger.debug(f"[Planner] Classification/planning failed: {_plan_err}")

        # ===== MCTS Reasoning Tree — Pre-planning for complex tasks =====
        # For complex multi-step tasks, explore the solution space with tree search
        # before entering the tool loop. This gives the loop a richer starting context.
        _mcts_reasoning_context = ""
        if task_is_complex and hasattr(self, 'reasoning_tree') and self.reasoning_tree:
            try:
                _mcts_result = self.reasoning_tree.execute(
                    "think_deeply",
                    problem=goal,
                    context=ace_context or "",
                    max_iterations=15,
                    max_depth=6,
                )
                if _mcts_result.get("success") and (_mcts_result.get("answer") or _mcts_result.get("summary")):
                    _mcts_reasoning_context = _mcts_result.get("summary") or _mcts_result.get("answer", "")
                    logger.info(
                        f"[MCTS] Pre-planning complete — confidence={_mcts_result.get('confidence', 0):.0%}, "
                        f"iterations={_mcts_result.get('metadata', {}).get('iterations', '?')}"
                    )
                else:
                    logger.debug("[MCTS] Pre-planning returned no usable result, proceeding normally")
            except (AttributeError, TypeError, ValueError, KeyError, ConnectionError, TimeoutError) as _mcts_err:
                logger.debug(f"[MCTS] Pre-planning failed (non-fatal): {_mcts_err}")

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

        # Inject MCTS reasoning tree context if available (from pre-planning above)
        if _mcts_reasoning_context:
            messages.append({
                "role": "assistant",
                "content": f"[Reasoning Tree Analysis]\n\n{_mcts_reasoning_context}\n\nLet me now execute this step by step.",
            })

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

                # Prevent unbounded message growth — keep system+goal + last N messages
                if len(messages) > 30:
                    messages = messages[:2] + messages[-28:]

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
                    except (AttributeError, TypeError, ValueError, ConnectionError, TimeoutError) as _replan_err:
                        logger.debug(f"[Planner] Re-plan check failed: {_replan_err}")

            except Exception as e:  # Catch-all: protects main agent ReAct loop
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
        except (ImportError, AttributeError, KeyError, TypeError) as _tone_err:
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
        except (ImportError, AttributeError, OSError, ValueError) as _profile_err:
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
                except (AttributeError, KeyError, TypeError, ValueError, TimeoutError, OSError) as _ctx_err:
                    logger.debug(f"[Agent] {_key} context retrieval failed: {_ctx_err}")
        except (AttributeError, TypeError, OSError) as _mem_err:
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
            except Exception as e:  # Catch-all: unknown tool implementations may raise anything
                return json.dumps({"error": f"Tool '{tool_name}' failed: {e}"})

        # 3. Web search fallback chain
        if tool_name in ("search_web", "web_search", "search"):
            query = args.get("query", args.get("action", str(args)))
            for sn in ("tavily_search", "brave_search", "web_search"):
                if sn in self.tools:
                    try:
                        result = self.tools[sn].execute(f"search {query}")
                        return json.dumps(result, default=str)[:MAX_RESULT]
                    except (AttributeError, KeyError, TypeError, ValueError, ConnectionError, TimeoutError, OSError) as e:
                        logger.debug(f"[Agent] Search fallback {sn} failed: {e}")
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
        except (AttributeError, TypeError, OSError) as e:
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
        if thought and self.monologue:
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

            # Audit chain: log every tool call (OpenFang-inspired Merkle trail)
            try:
                from aura.security.audit_chain import get_audit_chain
                get_audit_chain().append(
                    action_type="tool_call",
                    action_data={"tool": tool_name, "args_preview": str(args)[:200]},
                    agent_id="main",
                    session_id=getattr(self, '_session_id', 'default'),
                )
            except (ImportError, AttributeError, TypeError, OSError) as _audit_err:
                logger.debug(f"[AuditChain] Append failed: {_audit_err}")

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
                    # Don't reset to 0 — let the outer loop's >= 3 check work.
                    # The warning nudges the LLM, but persistent failures still abort.
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
        if thought and self.monologue:
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
        # SECURITY: Validate LLM-generated code before execution
        is_valid, error_msg = validate_script_code(code, "react_code_block")
        if not is_valid:
            messages.append({"role": "user", "content": f"Code rejected: {error_msg}"})
            consecutive_failures += 1
            if consecutive_failures >= 3:
                return {
                    "status": "fallback_to_tools",
                    "response": "",
                    "tool_calls_count": 0,
                    "consecutive_failures": 0,
                }
            return {
                "status": "continue",
                "response": "",
                "tool_calls_count": 0,
                "consecutive_failures": consecutive_failures,
            }
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

    # _handle_monologue_command — moved to DirectHandlersMixin
    # _handle_knowledge_graph_command — moved to KGBrainMixin

    # _handle_neurodream_command — moved to DirectHandlersMixin
    # _handle_git_command — moved to DirectHandlersMixin

    # _handle_direct_search — moved to DirectHandlersMixin
    # _handle_direct_crypto — moved to DirectHandlersMixin
    # _handle_direct_code — moved to DirectHandlersMixin

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
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"[Agent] Context tracking unavailable: {e}")

        # NeuroDream: check idle trigger FIRST (before resetting the timer), then record activity
        if hasattr(self, 'neurodream') and self.neurodream:
            try:
                if (self.neurodream.check_idle_trigger()
                        and self.neurodream.current_phase == SleepPhase.AWAKE):
                    self.neurodream.enter_sleep(trigger="idle")
                self.neurodream.record_activity()
            except (AttributeError, TypeError, ValueError, OSError) as e:
                logger.debug(f"[NeuroDream] Idle check/activity error: {e}")

        # ===== COHERENT LOOP: Post-response feedback (Phase 3.1) =====
        self._post_response_feedback(message)

        # Handle /init-project command
        if message.strip().lower().startswith("/init-project"):
            parts = message.strip().split(None, 1)
            target_path = parts[1].strip() if len(parts) > 1 else "."
            try:
                from aura.tools.project_context import init_project
                ctx["early_return"] = init_project(target_path)
            except (ImportError, OSError, ValueError) as e:
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
            except (ImportError, AttributeError, KeyError, TypeError, ValueError) as e:
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
            except (ImportError, AttributeError, TypeError) as e:
                logger.debug(f"[Agent] Emotion context tracking unavailable: {e}")

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
                    except (ImportError, AttributeError, TypeError) as e:
                        logger.debug(f"[Agent] Memory recall tracking unavailable: {e}")
                    try:
                        from api.routes.context import track_context_from_memory
                        track_context_from_memory([r.content[:100] for r in unified_results[:5]])
                    except (ImportError, AttributeError, TypeError) as e:
                        logger.debug(f"[Agent] Memory context tracking unavailable: {e}")
            except (ImportError, AttributeError, KeyError, TypeError, ValueError, TimeoutError, OSError) as e:
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
        except (ImportError, AttributeError, TypeError, OSError) as _tg_err:
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
        except (ImportError, AttributeError, OSError, ValueError) as e:
            logger.debug(f"[Agent] User profile load failed: {e}")

        if unified_context:
            context_parts.append(unified_context)

        # NeuroDream learned context
        try:
            if hasattr(self, 'neurodream') and self.neurodream:
                nd_context = self.neurodream.get_learned_context_prompt()
                if nd_context:
                    context_parts.append(f"LEARNED CONTEXT (from memory consolidation):\n{nd_context}")
        except (AttributeError, TypeError, OSError) as e:
            logger.debug(f"[NeuroDream] Learned context error: {e}")

        # Skill Library context
        try:
            if hasattr(self, 'skill_library') and self.skill_library:
                _skill_context = self.skill_library.get_skill_context(message)
                if _skill_context:
                    context_parts.append(f"SKILL CONTEXT:\n{_skill_context}")
                    logger.debug("[SkillLibrary] Injected skill context for: %s", message[:40])
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
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
        except (AttributeError, TypeError, ValueError, ConnectionError, TimeoutError) as _thinker_err:
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
        except (AttributeError, TypeError, ValueError) as _alma_err:
            logger.debug(f"[Agent] ALMA emotional update failed: {_alma_err}")

        # Update narrative self-model for significant interactions (background)
        if len(response) > 200:
            try:
                from aura.narrative_self import get_narrative_self
                _AGENT_EXECUTOR.submit(get_narrative_self().update_from_interaction, message, response, self.brain)
            except (ImportError, AttributeError, TypeError) as _narr_err:
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
            except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
                logger.debug(f"[KG BRAIN] Chat entity extraction error: {e}")

        # User fact extraction now handled by UnifiedMemory write gate

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
                except (ImportError, AttributeError, KeyError, TypeError) as e:
                    logger.debug(f"[ALMA] PAD retrieval failed: {e}")
                _umem_ref = _get_umem()
                import threading as _threading
                _store_fn = getattr(_umem_ref, "store_gated", _umem_ref.store)
                def _safe_store(_fn=_store_fn, _c=_mem_content, _p=_pad):
                    try:
                        _fn(content=_c, source="conversation", importance=0.5, emotional_pad=_p)
                    except Exception as _e:  # Catch-all: runs in background executor thread
                        logger.debug("[UnifiedMemory] Background store error: %s", _e)
                _AGENT_EXECUTOR.submit(_safe_store)
            except (AttributeError, TypeError, OSError) as e:
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
        except (AttributeError, TypeError, ValueError, OSError) as e:
            logger.debug("[SkillLibrary] Record interaction error: %s", e)

        # End monologue session
        if hasattr(self, 'monologue') and self.monologue:
            self.monologue.think("reflect", "Chat response completed")

        # ===== THINKER: Kick off async background reasoning (roadmap 3.6) =====
        if hasattr(self, 'thinker') and self.thinker:
            try:
                _conv_hist = self.brain.conversation_history if hasattr(self.brain, 'conversation_history') else None
                self.thinker.run_async(message, response, _conv_hist)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
            except (ImportError, AttributeError, KeyError, TypeError, ValueError) as e:
                logger.debug(f"[StrategyBandit] Selection error, falling back to CoT: {e}")
                selected_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT if ReasoningStrategy else "chain_of_thought"
        else:
            selected_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT if ReasoningStrategy else "chain_of_thought"

        # ===== Prompt Evolution Engine — Inject evolved prompt =====
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
            except (ImportError, AttributeError, KeyError, TypeError, ValueError) as e:
                logger.debug(f"[TemplateLib] Retrieval error: {e}")

        # Raw strategy results for rich trace capture
        _mcts_raw_result = None

        # Execute the selected strategy
        try:
            if selected_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
                response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

            elif selected_strategy == ReasoningStrategy.MCTS:
                if hasattr(self, 'reasoning_tree') and self.reasoning_tree:
                    try:
                        # Build conversation context for MCTS
                        mcts_context = ""
                        if system_prompt_addon:
                            mcts_context = f"System context: {system_prompt_addon}\n"
                        mcts_result = self.reasoning_tree.execute(
                            "think_deeply", problem=message, context=mcts_context
                        )
                        _mcts_raw_result = mcts_result
                        if mcts_result.get("success"):
                            # Use the summary (includes reasoning path + conclusion)
                            response = mcts_result.get("summary", "") or mcts_result.get("answer", "")
                        else:
                            # MCTS failed to find a good solution, use the answer anyway or fall back
                            response = mcts_result.get("answer", "")
                        if not response:
                            logger.debug("[StrategyBandit] MCTS returned empty, falling back to CoT")
                            response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)
                    except (AttributeError, KeyError, TypeError, ValueError, ConnectionError, TimeoutError) as e:
                        logger.debug(f"[StrategyBandit] MCTS error, falling back to CoT: {e}")
                        response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)
                else:
                    logger.debug("[StrategyBandit] MCTS selected but reasoning_tree not initialized, falling back to CoT")
                    response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

            else:
                # Unknown strategy — safe fallback
                response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier, system_prompt=system_prompt_addon)

        except Exception as e:  # Catch-all: strategy dispatch covers LLM + MCTS + tools
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
                            except Exception as ex:  # Catch-all: runs in background future callback
                                logger.debug(f"[StrategyBandit] Async eval error: {ex}")
                        eval_future.add_done_callback(_on_eval_done)
                    except (ImportError, AttributeError, TypeError, ValueError) as e:
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
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                composite_reward = 0.5
                logger.debug(f"[StrategyBandit] Outcome recording error: {e}")

        # ===== Prompt Evolution Engine — Record invocation =====
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
                    except (AttributeError, KeyError, TypeError, ValueError) as _trace_err:
                        logger.debug(f"[TemplateLib] Trace build fallback: {_trace_err}")
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
            except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
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
        except (ImportError, AttributeError, OSError) as _e:
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

    # _analyze_emotion, _get_soul_prompt, _temporal_grounding — moved to NarrativeMixin
    # _build_aura_context — moved to NarrativeMixin
    # _pre_response_appraisal, _post_response_feedback — moved to NarrativeMixin
    # _handle_aura_command, _handle_evoemo_command — moved to NarrativeMixin
    # get_current_mood, get_mood_emoji — moved to NarrativeMixin

    # KG Brain methods (get_kg_brain_stats, kg_brain_query, etc.) — moved to KGBrainMixin
    # Episodic memory stubs — moved to KGBrainMixin

    # Skill Library methods — moved to SkillManagerMixin

    def _speak(self, text: str, emotion: Optional[str] = None):
        """Speak text using TTS with optional emotional adaptation."""
        try:
            from .services.voice_presence import get_voice_presence
            vps = get_voice_presence()
            if vps._enabled:
                vps.speak(text, emotion=emotion, block=False)
        except (ImportError, AttributeError, TypeError, OSError) as e:
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
            except (AttributeError, TypeError, ValueError, OSError) as e:
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
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"NeuroDream shutdown: {e}")

        # 1.5. AURA v3.0 ALIVE system — removed (migrated to ALMA helpers)

        # 2. Unload Ollama models to free VRAM
        try:
            if hasattr(self, 'brain') and self.brain:
                unload_result = self.brain.unload_all_models()
                for model, success in unload_result.items():
                    if success:
                        results["freed_resources"].append(f"ollama:{model}")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"Ollama unload: {e}")

        # 3. Close browser if open
        try:
            if "browser" in self.tools and hasattr(self.tools["browser"], 'close'):
                self.tools["browser"].close()
                results["freed_resources"].append("browser")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
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
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"Voice unload: {e}")

        # 5. Save knowledge graph
        try:
            if "knowledge_graph" in self.tools:
                self.tools["knowledge_graph"].save()
                results["freed_resources"].append("knowledge_graph:saved")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"KG save: {e}")

        # 7. Save conversation history (preserve on disk, only clear in-memory)
        try:
            if hasattr(self, 'brain') and self.brain:
                self.brain._save_history()
                results["freed_resources"].append("conversation_history:saved")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"History save: {e}")

        # 9. Close Knowledge Graph Brain
        try:
            if hasattr(self, 'kg_brain') and self.kg_brain:
                # Flush pending extractions first
                if self.kg_bridge:
                    self.kg_bridge.flush()
                self.kg_brain.close()
                results["freed_resources"].append("kg_brain")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"KG Brain close: {e}")

        # 10. Episodic Memory — consolidated into UnifiedMemory, no separate close needed

        # 11. Close Skill Library
        try:
            if hasattr(self, 'skill_library') and self.skill_library:
                self.skill_library.shutdown()
                results["freed_resources"].append("skill_library")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"Skill Library close: {e}")

        # 12. Save ALMA emotional state for cross-session continuity
        try:
            from aura.emotion.alma_engine import save_state as alma_save_state
            alma_save_state()
            results["freed_resources"].append("alma_emotional_state:saved")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"ALMA state save: {e}")

        # 13. Stop GatewayDaemon (proactive intelligence)
        try:
            if self.gateway_daemon is not None:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.gateway_daemon.stop())
                except RuntimeError:
                    asyncio.run(self.gateway_daemon.stop())
                results["freed_resources"].append("gateway_daemon")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"GatewayDaemon stop: {e}")

        results["success"] = len(results["errors"]) == 0
        return results

