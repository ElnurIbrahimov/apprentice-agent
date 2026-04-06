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
from aura.core.react_runner import ReactMixin
from aura.core.chat_handler import ChatMixin


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
        compute_quality_metrics,
    )
    STRATEGY_BANDIT_AVAILABLE = True
except ImportError:
    STRATEGY_BANDIT_AVAILABLE = False
    ReasoningStrategy = None
    get_strategy_bandit = None
    compute_quality_metrics = None

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
class ApprenticeAgent(KGBrainMixin, SkillManagerMixin, NarrativeMixin, DirectHandlersMixin, ReactMixin, ChatMixin):
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
        # Start model validation in background — don't block startup
        try:
            Config.validate_models_on_startup(background=True)
            logger.info("[MODELS] Background validation started")
        except (AttributeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"[MODELS] Validation failed to start: {e}")

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
            evoemo=self.tools.get("evoemo"),
            inner_monologue=self.monologue,
            idle_threshold_minutes=30,
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
        self._nd_poll_thread = _threading.Thread(
            target=_neurodream_idle_poll, daemon=True, name="NeuroDream-IdlePoll"
        )
        self._nd_poll_thread.start()
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

                # Prevent unbounded message growth — keep system+goal + last N messages.
                # Trim at a safe boundary: never split a tool_calls/tool result pair.
                if len(messages) > 30:
                    tail = messages[2:]  # skip system + goal
                    # Find safe trim point — walk from oldest toward newest,
                    # ensure we don't start mid-pair (orphaned tool result).
                    trim_start = len(tail) - 28
                    if trim_start < 0:
                        trim_start = 0
                    # If the trim point lands on a "tool" message (result),
                    # back up to include the preceding assistant message with tool_calls.
                    while trim_start < len(tail) and tail[trim_start].get("role") == "tool":
                        trim_start -= 1
                        if trim_start < 0:
                            trim_start = 0
                            break
                    messages = messages[:2] + tail[trim_start:]

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

    # --- ReAct step implementation: see aura.core.react_runner ---

    # _handle_monologue_command — moved to DirectHandlersMixin
    # _handle_knowledge_graph_command — moved to KGBrainMixin

    # _handle_neurodream_command — moved to DirectHandlersMixin
    # _handle_git_command — moved to DirectHandlersMixin

    # _handle_direct_search — moved to DirectHandlersMixin
    # _handle_direct_crypto — moved to DirectHandlersMixin
    # _handle_direct_code — moved to DirectHandlersMixin

    # --- Chat flow (_prepare_chat, _finalize_chat, chat): see aura.core.chat_handler ---

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

        # ===== SEARCH CONTEXT INJECTION (streaming path) =====
        try:
            _search_query = self._needs_web_search(message)
            if _search_query:
                logger.debug(f"[SearchInject/stream] Detected search-needed: '{_search_query[:50]}'")
                from aura.tools.search_fallback import web_search_with_fallback
                _search_result = web_search_with_fallback(query=_search_query, max_results=5)
                if _search_result.get("results"):
                    _search_lines = []
                    for _sr in _search_result["results"][:5]:
                        _title = _sr.get("title", "")
                        _snippet = _sr.get("snippet", _sr.get("content", ""))[:200]
                        _url = _sr.get("url", "")
                        _search_lines.append(f"- {_title}: {_snippet} ({_url})")
                    _search_context = (
                        "WEB SEARCH RESULTS (use these as your primary source — "
                        "do NOT fabricate information beyond what's listed here):\n"
                        f"Query: {_search_query}\n"
                        + "\n".join(_search_lines)
                    )
                    if system_prompt_addon:
                        system_prompt_addon = system_prompt_addon + "\n\n" + _search_context
                    else:
                        system_prompt_addon = _search_context
                    logger.debug(f"[SearchInject/stream] Injected {len(_search_result['results'])} results")
        except (ImportError, AttributeError, TypeError, ValueError,
                ConnectionError, TimeoutError, OSError) as _sinj_err:
            logger.debug(f"[SearchInject/stream] Failed: {_sinj_err}")

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
                if hasattr(self, '_nd_poll_thread'):
                    self._nd_poll_thread.join(timeout=5)
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

        # 2b. Close brain HTTP connection pools
        try:
            if hasattr(self, 'brain') and self.brain:
                self.brain.close()
                results["freed_resources"].append("brain_http_clients")
        except Exception as e:  # Catch-all: shutdown must continue even if one step fails
            results["errors"].append(f"Brain close: {e}")

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

