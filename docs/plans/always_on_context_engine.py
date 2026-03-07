"""
AURA Always-On Context Engine (ACE)
====================================
Complete design for the pre-response context gathering system.

Before EVERY agent response, ACE automatically gathers, ranks, and injects
all relevant context into the system prompt -- so AURA always knows
everything it needs before speaking.

CURRENT STATE DIAGNOSIS
-----------------------
Right now, context gathering is scattered across agent.py:
  - _build_aura_context() only gets mood + thinking prefix (lines 5708-5730)
  - KG/RAG/A-MEM/Unified queries happen inline in chat() (lines 4805-4900)
  - ContextBudget is a simple token allocator (3000 tokens total)
  - Screen context via Screenpipe exists but is NEVER called in chat flow
  - User profile is read from flat markdown (data/memory/user_profile.md)
  - No file auto-read, no URL auto-fetch, no project detection
  - Context is built separately for chat() and chat_stream() (duplicated code)
  - No learning/feedback on what context was actually useful

This design replaces all of that with a single, unified engine.

Author: AURA Development Team
Created: 2026-02-28
"""

import hashlib
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
#                         CONFIGURATION
# ============================================================================

class ContextConfig:
    """All tunable knobs for the context engine."""

    # Token budget (approximate -- 1 token ~ 4 chars)
    TOTAL_TOKEN_BUDGET = 4000           # Max tokens for all injected context
    CHARS_PER_TOKEN = 4                 # Rough approximation

    # Per-source token allocations (initial weights, learned over time)
    SOURCE_WEIGHTS = {
        "screen":       0.15,   # 600 tokens
        "user_profile": 0.08,   # 320 tokens -- small, always included
        "episodic":     0.15,   # 600 tokens
        "amem":         0.15,   # 600 tokens
        "kg":           0.12,   # 480 tokens
        "rag":          0.10,   # 400 tokens
        "file":         0.10,   # 400 tokens
        "url":          0.05,   # 200 tokens
        "project":      0.05,   # 200 tokens
        "conversation": 0.05,   # 200 tokens -- session memory summary
    }

    # Source priorities (higher = survives budget cuts)
    SOURCE_PRIORITY = {
        "user_profile": 100,    # Always included, never cut
        "conversation": 95,     # Current session is critical
        "screen":       85,     # What they're looking at right now
        "project":      80,     # What project they're in
        "amem":         70,     # Personal memories
        "episodic":     65,     # Past episodes
        "kg":           60,     # Knowledge graph
        "rag":          55,     # Indexed documents
        "file":         50,     # Auto-read files
        "url":          40,     # Auto-fetched URLs
    }

    # Timing constraints (milliseconds)
    MAX_GATHER_TIME_MS = 800            # Hard deadline for all context gathering
    SCREEN_POLL_INTERVAL_S = 5          # How often to poll screen (background)
    SCREEN_CACHE_TTL_S = 10             # How long screen context is valid

    # Message analysis thresholds
    MIN_MESSAGE_LENGTH_FOR_DEEP = 10    # Short msgs get shallow context only
    URL_PATTERN = re.compile(r'https?://[^\s<>"\']+')
    FILE_PATH_PATTERN = re.compile(
        r'(?:[A-Za-z]:[/\\]|/|~/)[^\s<>"\'*?|]+\.\w{1,10}'
    )

    # Relevance feedback
    FEEDBACK_DECAY = 0.95               # Weight decay per session
    FEEDBACK_BOOST = 0.1                # Boost when context source is cited


# ============================================================================
#                       DATA STRUCTURES
# ============================================================================

class ContextSource(str, Enum):
    """All sources the context engine can draw from."""
    SCREEN = "screen"
    USER_PROFILE = "user_profile"
    EPISODIC = "episodic"
    AMEM = "amem"
    KG = "kg"
    RAG = "rag"
    FILE = "file"
    URL = "url"
    PROJECT = "project"
    CONVERSATION = "conversation"


@dataclass
class ContextBlock:
    """A single block of context from one source."""
    source: ContextSource
    content: str
    relevance: float            # 0.0 to 1.0
    tokens_estimated: int       # Approximate token count
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0     # When this block was gathered

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.tokens_estimated:
            self.tokens_estimated = len(self.content) // ContextConfig.CHARS_PER_TOKEN

    @property
    def priority_score(self) -> float:
        """Combined score for budget allocation."""
        base_priority = ContextConfig.SOURCE_PRIORITY.get(self.source.value, 50)
        return (base_priority / 100.0) * 0.5 + self.relevance * 0.5


@dataclass
class GatheredContext:
    """Complete context gathered for one message."""
    blocks: List[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0
    gather_time_ms: float = 0.0
    active_project: Optional[str] = None
    active_app: Optional[str] = None
    user_name: Optional[str] = None
    mood: str = "neutral"
    errors: List[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Format all blocks into a system prompt injection string."""
        if not self.blocks:
            return ""
        return format_context_injection(self)


@dataclass
class ConversationTurn:
    """One turn in the current conversation session."""
    role: str                   # "user" or "assistant"
    content: str
    timestamp: float
    tokens: int = 0
    context_sources_used: List[str] = field(default_factory=list)


# ============================================================================
#           CONTEXT GATHERING PIPELINE (THE CORE ENGINE)
# ============================================================================

class AlwaysOnContextEngine:
    """
    The always-on context engine. Called before every agent response.

    Architecture:
        1. ANALYZE message (extract intents, entities, file paths, URLs)
        2. GATHER context from all sources in parallel (with timeout)
        3. RANK and BUDGET-FIT the gathered blocks
        4. FORMAT into system prompt injection
        5. LEARN from response feedback (which context was actually useful)

    Data Flow:
        User message
            |
            v
        [Message Analyzer] --> intents, entities, file_paths, urls
            |
            v
        [Parallel Gatherers] --> ContextBlocks from each source
            |       |       |       |       |       |
            v       v       v       v       v       v
          Screen  Memory  Files   URLs   Project  Conv
            |
            v
        [Budget Manager] --> ranked, trimmed blocks
            |
            v
        [Prompt Formatter] --> system prompt injection string
            |
            v
        [brain.think(message, system_prompt=injection)]
    """

    def __init__(
        self,
        screenpipe_client=None,
        screen_reader=None,
        memory_retriever=None,
        unified_memory=None,
        kg_bridge=None,
        tools: Optional[Dict] = None,
        data_dir: Optional[Path] = None,
    ):
        self._screenpipe = screenpipe_client
        self._screen_reader = screen_reader
        self._memory_retriever = memory_retriever
        self._unified_memory = unified_memory
        self._kg_bridge = kg_bridge
        self._tools = tools or {}
        self._data_dir = data_dir or Path("data")

        # Conversation session memory
        self._session_turns: List[ConversationTurn] = []
        self._session_start = time.time()
        self._session_id = hashlib.md5(
            str(self._session_start).encode()
        ).hexdigest()[:12]

        # Screen context cache (polled in background)
        self._screen_cache: Optional[Dict] = None
        self._screen_cache_time: float = 0
        self._screen_lock = threading.Lock()

        # Relevance feedback weights (learned)
        self._source_weight_adjustments: Dict[str, float] = {}

        # Project detection cache
        self._detected_project: Optional[str] = None
        self._detected_project_time: float = 0

        # Thread pool for parallel gathering
        self._executor = ThreadPoolExecutor(
            max_workers=6,
            thread_name_prefix="ace_gather"
        )

        logger.info("[ACE] Always-On Context Engine initialized")

    # ====================================================================
    #                    MAIN ENTRY POINT
    # ====================================================================

    def gather(self, message: str) -> GatheredContext:
        """
        Gather all context for a message. Called before every LLM call.

        This is the single entry point. Everything is parallelized with
        a hard timeout of MAX_GATHER_TIME_MS.

        Args:
            message: The user's message

        Returns:
            GatheredContext ready for system prompt injection
        """
        start = time.time()
        ctx = GatheredContext()

        # Step 1: Analyze the message
        analysis = self._analyze_message(message)
        is_simple = analysis["is_simple"]

        # Step 2: Launch parallel gatherers
        futures: Dict[Future, ContextSource] = {}

        # Always gather these (fast, critical)
        futures[self._executor.submit(
            self._gather_user_profile
        )] = ContextSource.USER_PROFILE

        futures[self._executor.submit(
            self._gather_conversation_context, message
        )] = ContextSource.CONVERSATION

        # Screen context (from cache or fresh)
        futures[self._executor.submit(
            self._gather_screen_context
        )] = ContextSource.SCREEN

        # For non-simple messages, also gather deep context
        if not is_simple:
            futures[self._executor.submit(
                self._gather_memory_context, message, analysis
            )] = ContextSource.AMEM

            futures[self._executor.submit(
                self._gather_kg_context, message, analysis
            )] = ContextSource.KG

            futures[self._executor.submit(
                self._gather_rag_context, message
            )] = ContextSource.RAG

            futures[self._executor.submit(
                self._gather_project_context, analysis
            )] = ContextSource.PROJECT

        # File paths mentioned in message
        if analysis["file_paths"]:
            futures[self._executor.submit(
                self._gather_file_context, analysis["file_paths"]
            )] = ContextSource.FILE

        # URLs mentioned in message
        if analysis["urls"]:
            futures[self._executor.submit(
                self._gather_url_context, analysis["urls"]
            )] = ContextSource.URL

        # Step 3: Collect results with timeout
        deadline = ContextConfig.MAX_GATHER_TIME_MS / 1000.0
        for future in as_completed(futures, timeout=deadline):
            source = futures[future]
            try:
                blocks = future.result(timeout=0.1)
                if blocks:
                    if isinstance(blocks, list):
                        ctx.blocks.extend(blocks)
                    else:
                        ctx.blocks.append(blocks)
            except Exception as e:
                ctx.errors.append(f"{source.value}: {e}")
                logger.debug(f"[ACE] Gather error for {source.value}: {e}")

        # Step 4: Budget management -- rank and trim
        ctx.blocks = self._apply_budget(ctx.blocks)
        ctx.total_tokens = sum(b.tokens_estimated for b in ctx.blocks)

        # Step 5: Set metadata
        ctx.gather_time_ms = (time.time() - start) * 1000
        ctx.active_project = self._detected_project
        ctx.active_app = self._get_active_app()
        ctx.user_name = self._get_user_name()
        ctx.mood = self._get_mood()

        # Record this as a conversation turn
        self._session_turns.append(ConversationTurn(
            role="user",
            content=message,
            timestamp=time.time(),
            tokens=len(message) // ContextConfig.CHARS_PER_TOKEN,
        ))

        logger.info(
            f"[ACE] Gathered {len(ctx.blocks)} blocks, "
            f"{ctx.total_tokens} tokens in {ctx.gather_time_ms:.0f}ms"
        )

        return ctx

    def record_response(self, response: str, context_used: GatheredContext):
        """
        Record the assistant's response for conversation memory.
        Called after LLM generates a response.
        """
        self._session_turns.append(ConversationTurn(
            role="assistant",
            content=response,
            timestamp=time.time(),
            tokens=len(response) // ContextConfig.CHARS_PER_TOKEN,
            context_sources_used=[b.source.value for b in context_used.blocks],
        ))

    # ====================================================================
    #               STEP 1: MESSAGE ANALYSIS
    # ====================================================================

    def _analyze_message(self, message: str) -> Dict[str, Any]:
        """
        Fast analysis of the user's message to determine what context to fetch.

        Returns:
            Dict with:
              - is_simple: bool (greetings, yes/no, short msgs)
              - keywords: List[str] (meaningful words)
              - entities: List[str] (proper nouns, project names)
              - file_paths: List[str] (detected file/directory paths)
              - urls: List[str] (detected URLs)
              - references_past: bool (mentions "yesterday", "last time", etc.)
              - is_follow_up: bool (continues previous topic)
              - topic_shift: bool (different topic from last turn)
        """
        msg_lower = message.lower().strip()
        word_count = len(message.split())

        # Simple message detection
        simple_patterns = [
            r'^(hi|hey|hello|yo|sup|morning|evening|night)\b',
            r'^(yes|no|yeah|nah|ok|okay|sure|thanks|thank you|bye|goodbye)\b',
            r'^(how are you|what\'s up|what are you)\b',
        ]
        is_simple = (
            word_count <= 5
            or any(re.match(p, msg_lower) for p in simple_patterns)
        )

        # Extract file paths
        file_paths = ContextConfig.FILE_PATH_PATTERN.findall(message)

        # Extract URLs
        urls = ContextConfig.URL_PATTERN.findall(message)

        # Extract keywords (filter stop words)
        stop_words = {
            "i", "you", "the", "a", "an", "is", "are", "was", "were",
            "it", "this", "that", "what", "how", "why", "when", "where",
            "do", "does", "did", "have", "has", "had", "be", "been",
            "will", "would", "could", "should", "can", "may", "might",
            "to", "for", "of", "in", "on", "at", "by", "with", "about",
            "my", "me", "your", "our", "their", "its",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', msg_lower)
        keywords = [w for w in words if w not in stop_words]

        # Detect temporal references (need episodic memory)
        temporal_words = [
            "yesterday", "last time", "earlier", "before", "previously",
            "remember when", "last week", "last month", "ago",
        ]
        references_past = any(t in msg_lower for t in temporal_words)

        # Detect follow-up vs topic shift
        is_follow_up = False
        topic_shift = False
        if self._session_turns:
            last_turn = self._session_turns[-1]
            last_keywords = set(
                re.findall(r'\b[a-zA-Z]{3,}\b', last_turn.content.lower())
            ) - stop_words
            current_keywords = set(keywords)
            overlap = len(last_keywords & current_keywords)
            is_follow_up = overlap >= 2 or msg_lower.startswith(("and ", "also ", "what about ", "but "))
            topic_shift = overlap == 0 and word_count > 5

        # Extract potential entity names (capitalized words)
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', message)
        # Filter out sentence-start capitalization
        entities = [e for e in entities if len(e) > 2]

        return {
            "is_simple": is_simple,
            "keywords": keywords,
            "entities": entities,
            "file_paths": file_paths,
            "urls": urls,
            "references_past": references_past,
            "is_follow_up": is_follow_up,
            "topic_shift": topic_shift,
            "word_count": word_count,
        }

    # ====================================================================
    #               STEP 2: PARALLEL GATHERERS
    # ====================================================================

    # ---------- 1. Screen Context ----------

    def _gather_screen_context(self) -> Optional[ContextBlock]:
        """
        Get what's on the user's screen right now.

        Uses Screenpipe (preferred, always-on OCR) or ScreenReader (on-demand).
        Returns cached result if fresh enough.

        Data flow:
            Screenpipe API (localhost:3030)
                -> get_screen_context_filtered()
                -> current_app, window_name, recent OCR text, error detection
                -> ContextBlock

        Screenpipe provides:
            - Current application name (e.g., "VS Code")
            - Current window title (e.g., "agent.py - AURA")
            - OCR text from screen (last 2 minutes)
            - Error detection (tracebacks, 404s, etc.)

        Privacy:
            - Password managers, incognito windows are auto-filtered
            - See ScreenpipeClient._is_private()
        """
        # Check cache first
        now = time.time()
        if (self._screen_cache
                and (now - self._screen_cache_time) < ContextConfig.SCREEN_CACHE_TTL_S):
            cached = self._screen_cache
            if cached.get("available"):
                content = self._format_screen_context(cached)
                return ContextBlock(
                    source=ContextSource.SCREEN,
                    content=content,
                    relevance=0.7,
                    tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
                    metadata={"cached": True, "app": cached.get("current_app")},
                )
            return None

        # Try Screenpipe first (preferred -- always-on, no screenshot needed)
        if self._screenpipe:
            try:
                ctx = self._screenpipe.get_screen_context_filtered(
                    minutes=2,
                    max_chars=1500,
                    only_if_changed=False,
                )
                with self._screen_lock:
                    self._screen_cache = ctx
                    self._screen_cache_time = now

                if ctx.get("available") and ctx.get("recent_text"):
                    content = self._format_screen_context(ctx)
                    # Boost relevance if errors detected
                    relevance = 0.8 if ctx.get("has_errors") else 0.6
                    return ContextBlock(
                        source=ContextSource.SCREEN,
                        content=content,
                        relevance=relevance,
                        tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
                        metadata={
                            "app": ctx.get("current_app"),
                            "window": ctx.get("current_window"),
                            "has_errors": ctx.get("has_errors"),
                        },
                    )
            except Exception as e:
                logger.debug(f"[ACE] Screenpipe error: {e}")

        # Fallback: ScreenReader (takes a screenshot + OCR -- slower)
        if self._screen_reader:
            try:
                window_info = self._screen_reader.get_active_window()
                if window_info.get("success"):
                    content = (
                        f"Active: {window_info.get('title', 'Unknown')} "
                        f"({window_info.get('process', '')})"
                    )
                    return ContextBlock(
                        source=ContextSource.SCREEN,
                        content=content,
                        relevance=0.4,
                        tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
                        metadata={"fallback": True},
                    )
            except Exception:
                pass

        return None

    def _format_screen_context(self, screen_data: Dict) -> str:
        """Format screen context data into a readable string."""
        parts = []
        app = screen_data.get("current_app", "Unknown")
        window = screen_data.get("current_window", "")
        parts.append(f"Current app: {app}")
        if window and window != app:
            parts.append(f"Window: {window}")

        if screen_data.get("has_errors"):
            parts.append("** ERRORS DETECTED on screen **")

        text = screen_data.get("recent_text", "")
        if text:
            # Truncate intelligently -- keep first and last parts
            if len(text) > 800:
                text = text[:600] + "\n...\n" + text[-200:]
            parts.append(f"Screen text:\n{text}")

        apps_used = screen_data.get("apps_used", [])
        if len(apps_used) > 1:
            parts.append(f"Recent apps: {', '.join(apps_used[:5])}")

        return "\n".join(parts)

    # ---------- 2. User Profile ----------

    def _gather_user_profile(self) -> Optional[ContextBlock]:
        """
        Load user profile facts. Always included, highest priority.

        Sources:
            - data/memory/user_profile.md (key-value facts)
            - MemoryRetriever.user_profile dict
            - world_state.json relationships

        This is compact and always included. Contains name, preferences,
        relationship context.
        """
        parts = []

        # From MemoryRetriever
        if self._memory_retriever:
            profile = self._memory_retriever.user_profile
            if profile:
                for key, value in profile.items():
                    parts.append(f"{key}: {value}")

        # Fallback: read file directly
        if not parts:
            profile_path = self._data_dir / "memory" / "user_profile.md"
            if profile_path.exists():
                try:
                    content = profile_path.read_text(encoding="utf-8").strip()
                    # Parse key-value pairs
                    for line in content.split("\n"):
                        line = line.strip()
                        if ":" in line and not line.startswith("#") and not line.startswith("*"):
                            parts.append(line.replace("**", ""))
                except Exception:
                    pass

        if not parts:
            return None

        content = "User: " + " | ".join(parts)
        return ContextBlock(
            source=ContextSource.USER_PROFILE,
            content=content,
            relevance=1.0,  # Always relevant
            tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
        )

    # ---------- 3. Memory Context (A-MEM + Episodic) ----------

    def _gather_memory_context(
        self, message: str, analysis: Dict
    ) -> List[ContextBlock]:
        """
        Query personal memory systems for relevant past knowledge.

        Sources queried:
            1. A-MEM (Associative Memory) -- self-organizing memory notes
               - Stores: facts, conversations, insights with embeddings
               - Query: semantic search via FAISS/numpy similarity
               - Returns: MemoryNote objects with content, category, keywords

            2. Episodic Memory (Qdrant) -- timestamped episodes
               - Stores: conversations, events, user preferences
               - Query: vector search + temporal/emotional filters
               - Returns: Episodes with content, importance, valence

            3. Unified Memory -- fan-out query across all backends
               - Deduplicates across A-MEM, KG, RAG, Episodic
               - Returns: UnifiedResult with blended scores

        The unified memory already handles parallel querying and dedup,
        so we use it as the primary path and only fall back to direct
        A-MEM queries if unified is unavailable.
        """
        blocks = []

        # Primary: Unified Memory (queries everything, deduplicates)
        if self._unified_memory:
            try:
                results = self._unified_memory.query(
                    query=message,
                    k=5,
                    sources=["amem", "episodic"],  # KG and RAG gathered separately
                    min_score=0.15,
                )
                if results:
                    memory_parts = []
                    for r in results:
                        source_label = r.source.upper()
                        memory_parts.append(
                            f"[{source_label}] {r.content[:300]}"
                        )

                    content = "\n".join(memory_parts)
                    blocks.append(ContextBlock(
                        source=ContextSource.AMEM,
                        content=content,
                        relevance=max(r.score for r in results),
                        tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
                        metadata={
                            "result_count": len(results),
                            "sources": list(set(r.source for r in results)),
                        },
                    ))
            except Exception as e:
                logger.debug(f"[ACE] Unified memory error: {e}")

        # Fallback: Direct A-MEM query
        if not blocks and "amem" in self._tools:
            try:
                amem_tool = self._tools["amem"]
                memories_raw = amem_tool.amem.search(message, k=3)
                if memories_raw:
                    memories = [note for note, score in memories_raw]
                    memory_parts = [
                        f"- {m.content[:200]}" for m in memories if m.content
                    ]
                    if memory_parts:
                        content = "\n".join(memory_parts)
                        blocks.append(ContextBlock(
                            source=ContextSource.AMEM,
                            content=content,
                            relevance=0.6,
                            tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
                        ))
            except Exception as e:
                logger.debug(f"[ACE] A-MEM fallback error: {e}")

        # If temporal references, also query episodic directly for timeline
        if analysis.get("references_past") and self._unified_memory:
            try:
                from aura_episodic_memory.episode import EpisodeQuery
                # This would search specifically for temporal episodes
                # The unified memory above already queries episodic,
                # but temporal references benefit from a timeline-focused query
                pass  # Handled by unified memory above
            except ImportError:
                pass

        return blocks

    # ---------- 4. Knowledge Graph ----------

    def _gather_kg_context(
        self, message: str, analysis: Dict
    ) -> Optional[ContextBlock]:
        """
        Query the knowledge graph for entity relationships.

        The KG stores:
            - Nodes: concepts, entities, people, projects, tools, events
            - Edges: relates_to, is_a, part_of, causes, solves, uses, etc.
            - Each node has: label, properties, confidence, access history
            - Bi-temporal tracking (valid_from/valid_to)

        Data flow:
            TitansKGBridge.get_context_for_query(message, max_entities=3)
                -> finds matching nodes via label/property search
                -> retrieves connected edges
                -> formats as readable context string
        """
        if not self._kg_bridge:
            return None

        try:
            kg_context = self._kg_bridge.get_context_for_query(
                message, max_entities=3
            )
            if kg_context and len(kg_context.strip()) > 10:
                return ContextBlock(
                    source=ContextSource.KG,
                    content=kg_context,
                    relevance=0.65,
                    tokens_estimated=len(kg_context) // ContextConfig.CHARS_PER_TOKEN,
                    metadata={"source": "kg_bridge"},
                )
        except Exception as e:
            logger.debug(f"[ACE] KG context error: {e}")

        return None

    # ---------- 5. RAG (Indexed Documents) ----------

    def _gather_rag_context(self, message: str) -> Optional[ContextBlock]:
        """
        Query locally indexed documents via RAG.

        The RAG system indexes files the user has loaded and provides
        chunk-level semantic search. Good for technical docs, code files,
        and reference material.
        """
        if "local_rag" not in self._tools:
            return None

        try:
            rag_tool = self._tools["local_rag"]
            rag_context = rag_tool.rag.get_context(
                message, top_k=3, max_tokens=1000
            )
            if rag_context and len(rag_context.strip()) > 10:
                return ContextBlock(
                    source=ContextSource.RAG,
                    content=rag_context,
                    relevance=0.55,
                    tokens_estimated=len(rag_context) // ContextConfig.CHARS_PER_TOKEN,
                    metadata={"source": "local_rag"},
                )
        except Exception as e:
            logger.debug(f"[ACE] RAG context error: {e}")

        return None

    # ---------- 6. File Context (Auto-read) ----------

    def _gather_file_context(
        self, file_paths: List[str]
    ) -> List[ContextBlock]:
        """
        Auto-read files mentioned in the user's message.

        When the user says "look at D:/project/main.py", we read it
        and include relevant portions in context.

        Safety:
            - Only reads text files (not binaries)
            - Max 2000 chars per file
            - Max 3 files per message
            - Won't read files in sensitive directories
        """
        blocks = []
        sensitive_dirs = {"password", "secret", "credential", ".ssh", ".gnupg"}

        for path_str in file_paths[:3]:  # Max 3 files
            path = Path(path_str)

            # Safety check
            if any(s in str(path).lower() for s in sensitive_dirs):
                continue

            if not path.exists() or not path.is_file():
                continue

            # Skip binary files
            try:
                suffix = path.suffix.lower()
                binary_extensions = {
                    ".exe", ".dll", ".so", ".bin", ".zip", ".tar",
                    ".gz", ".png", ".jpg", ".jpeg", ".gif", ".mp3",
                    ".mp4", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
                }
                if suffix in binary_extensions:
                    blocks.append(ContextBlock(
                        source=ContextSource.FILE,
                        content=f"[Binary file: {path.name} ({suffix})]",
                        relevance=0.3,
                        tokens_estimated=10,
                        metadata={"path": str(path), "binary": True},
                    ))
                    continue

                content = path.read_text(encoding="utf-8", errors="replace")
                # Truncate to 2000 chars
                if len(content) > 2000:
                    content = content[:1800] + f"\n... [{len(content)} chars total]"

                blocks.append(ContextBlock(
                    source=ContextSource.FILE,
                    content=f"File: {path.name}\n{content}",
                    relevance=0.7,
                    tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
                    metadata={"path": str(path), "size": path.stat().st_size},
                ))
            except Exception as e:
                logger.debug(f"[ACE] File read error for {path}: {e}")

        return blocks

    # ---------- 7. URL Context (Auto-fetch) ----------

    def _gather_url_context(self, urls: List[str]) -> List[ContextBlock]:
        """
        Auto-fetch URLs mentioned in the user's message.

        When the user pastes a URL, we fetch its title/summary and
        include it in context so AURA can discuss it immediately.

        Implementation options:
            - Firecrawl MCP tool (if available)
            - httpx/requests with BeautifulSoup
            - trafilatura for article extraction

        Safety:
            - Max 1 URL per message (to avoid slow gathering)
            - Timeout of 3 seconds
            - Only extract title + first 500 chars of content
        """
        blocks = []

        for url in urls[:1]:  # Max 1 URL to avoid latency
            try:
                import httpx
                with httpx.Client(timeout=3.0, follow_redirects=True) as client:
                    response = client.get(url)
                    if response.status_code == 200:
                        html = response.text[:10000]
                        # Extract title
                        title_match = re.search(
                            r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL
                        )
                        title = title_match.group(1).strip() if title_match else url
                        # Basic text extraction
                        text = re.sub(r'<[^>]+>', ' ', html)
                        text = re.sub(r'\s+', ' ', text).strip()[:500]

                        content = f"URL: {title}\n{text}"
                        blocks.append(ContextBlock(
                            source=ContextSource.URL,
                            content=content,
                            relevance=0.5,
                            tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
                            metadata={"url": url, "title": title},
                        ))
            except Exception as e:
                logger.debug(f"[ACE] URL fetch error for {url}: {e}")

        return blocks

    # ---------- 8. Project Context ----------

    def _gather_project_context(
        self, analysis: Dict
    ) -> Optional[ContextBlock]:
        """
        Detect and provide context about the active project.

        Detection strategy (ordered by confidence):
            1. Screen context: Window title often contains project path
               e.g., "agent.py - AURA" -> project is AURA
            2. Recent file paths: If user mentioned files, infer project
               from directory structure
            3. Conversation history: What project was discussed recently
            4. World model: world_state.json has a projects list

        Output:
            Project name, recent activity, key files, current state.
        """
        # Try to detect from screen
        active_app = self._get_active_app()
        window_title = ""
        if self._screen_cache:
            window_title = self._screen_cache.get("current_window", "")

        project = None

        # Strategy 1: Parse window title for project name
        # Common patterns: "file.py - ProjectName", "ProjectName [path]"
        if window_title:
            # VS Code pattern: "filename - ProjectFolder - Visual Studio Code"
            parts = window_title.split(" - ")
            if len(parts) >= 3 and "Visual Studio Code" in parts[-1]:
                project = parts[-2].strip()
            elif len(parts) >= 2:
                project = parts[-1].strip()

        # Strategy 2: From file paths in message
        if not project and analysis.get("file_paths"):
            path = Path(analysis["file_paths"][0])
            # Walk up to find a project root (has .git, package.json, etc.)
            for parent in path.parents:
                if (parent / ".git").exists() or (parent / "package.json").exists():
                    project = parent.name
                    break

        # Strategy 3: From world model
        if not project:
            try:
                world_state_path = self._data_dir / "world_state.json"
                if world_state_path.exists():
                    import json
                    world = json.loads(
                        world_state_path.read_text(encoding="utf-8")
                    )
                    projects = world.get("projects", [])
                    if projects:
                        # Return most recently mentioned project
                        project = projects[-1].get("name", "")
            except Exception:
                pass

        if project:
            self._detected_project = project
            self._detected_project_time = time.time()

            content = f"Active project: {project}"
            if active_app:
                content += f" (in {active_app})"

            return ContextBlock(
                source=ContextSource.PROJECT,
                content=content,
                relevance=0.6,
                tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
                metadata={"project": project},
            )

        return None

    # ---------- 9. Conversation Context (Session Memory) ----------

    def _gather_conversation_context(
        self, current_message: str
    ) -> Optional[ContextBlock]:
        """
        Summarize the current session's conversation so far.

        This is how AURA remembers what was said earlier in THIS session.
        Without this, each message would be contextless.

        Strategy:
            - Last 3 turns: Include verbatim (most relevant)
            - Earlier turns: Compress to key topics/entities
            - Track topic thread: What are we talking about?

        Token budget:
            ~200 tokens for conversation context. This is compact
            because the LLM already has chat history in its context window.
            This is supplementary -- reminding about key facts/decisions
            from earlier in the session.
        """
        if not self._session_turns:
            return None

        parts = []
        turns = self._session_turns

        # If many turns, summarize older ones
        if len(turns) > 6:
            # Compress old turns into topic list
            old_turns = turns[:-6]
            topics = set()
            for turn in old_turns:
                # Extract key nouns from each turn
                words = re.findall(r'\b[A-Z][a-z]+\b', turn.content)
                topics.update(words[:3])
            if topics:
                parts.append(f"Earlier topics: {', '.join(list(topics)[:8])}")

        # Last 3 exchanges verbatim (truncated)
        recent = turns[-6:]
        for turn in recent:
            role = "User" if turn.role == "user" else "Aura"
            snippet = turn.content[:150]
            if len(turn.content) > 150:
                snippet += "..."
            parts.append(f"{role}: {snippet}")

        if not parts:
            return None

        content = "\n".join(parts)
        return ContextBlock(
            source=ContextSource.CONVERSATION,
            content=content,
            relevance=0.85,
            tokens_estimated=len(content) // ContextConfig.CHARS_PER_TOKEN,
            metadata={"turns": len(self._session_turns)},
        )

    # ====================================================================
    #               STEP 3: BUDGET MANAGEMENT
    # ====================================================================

    def _apply_budget(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """
        Rank blocks and trim to fit within TOTAL_TOKEN_BUDGET.

        Algorithm:
            1. Sort blocks by priority_score (descending)
            2. Greedily include blocks until budget is exhausted
            3. If a block would exceed budget, truncate its content
            4. Apply learned weight adjustments

        The priority_score combines:
            - Source priority (user_profile=100, url=40)
            - Relevance score (0.0 to 1.0)
            => priority_score = (source_priority/100)*0.5 + relevance*0.5
        """
        if not blocks:
            return []

        # Apply learned weight adjustments
        for block in blocks:
            adjustment = self._source_weight_adjustments.get(
                block.source.value, 0.0
            )
            block.relevance = min(1.0, max(0.0, block.relevance + adjustment))

        # Sort by priority score (highest first)
        blocks.sort(key=lambda b: b.priority_score, reverse=True)

        budget = ContextConfig.TOTAL_TOKEN_BUDGET
        selected = []
        used = 0

        for block in blocks:
            if used >= budget:
                break

            remaining = budget - used
            if block.tokens_estimated <= remaining:
                # Fits entirely
                selected.append(block)
                used += block.tokens_estimated
            elif remaining >= 50:  # Worth truncating if >= 50 tokens left
                # Truncate content to fit
                max_chars = remaining * ContextConfig.CHARS_PER_TOKEN
                block.content = block.content[:max_chars] + "..."
                block.tokens_estimated = remaining
                selected.append(block)
                used += remaining
                break  # Budget exhausted

        return selected

    # ====================================================================
    #            STEP 4: SYSTEM PROMPT FORMATTING
    # ====================================================================
    # (see format_context_injection() below)

    # ====================================================================
    #            STEP 5: RELEVANCE FEEDBACK / LEARNING
    # ====================================================================

    def provide_feedback(
        self,
        context: GatheredContext,
        was_helpful: bool,
        cited_sources: Optional[List[str]] = None,
    ):
        """
        Learn from whether the gathered context was actually useful.

        Called after the response, based on:
            - Did the LLM cite/reference the injected context?
            - Did the user seem satisfied? (no "that's wrong", no retry)
            - Which sources were most/least useful?

        This adjusts source weights over time so AURA learns to
        prioritize the most useful context sources.
        """
        for block in context.blocks:
            source_key = block.source.value

            if cited_sources and source_key in cited_sources:
                # This source was explicitly referenced -- boost it
                current = self._source_weight_adjustments.get(source_key, 0.0)
                self._source_weight_adjustments[source_key] = min(
                    0.3, current + ContextConfig.FEEDBACK_BOOST
                )
            elif not was_helpful:
                # Response wasn't helpful -- decay this source slightly
                current = self._source_weight_adjustments.get(source_key, 0.0)
                self._source_weight_adjustments[source_key] = max(
                    -0.2, current - 0.02
                )

        # Decay all adjustments toward zero over time
        for key in list(self._source_weight_adjustments):
            self._source_weight_adjustments[key] *= ContextConfig.FEEDBACK_DECAY

    # ====================================================================
    #               HELPER METHODS
    # ====================================================================

    def _get_active_app(self) -> Optional[str]:
        """Get the currently active application name."""
        if self._screen_cache:
            return self._screen_cache.get("current_app")
        return None

    def _get_user_name(self) -> Optional[str]:
        """Get the user's name from profile."""
        if self._memory_retriever:
            return self._memory_retriever.get_fact("name") or None
        return None

    def _get_mood(self) -> str:
        """Get current emotional state from ALMA."""
        try:
            from aura.emotion.alma_engine import get_alma_engine
            alma = get_alma_engine()
            if alma:
                state = alma.get_emotional_state()
                return state.get("dominant_emotion", "neutral")
        except Exception:
            pass
        return "neutral"

    def start_background_screen_polling(self):
        """
        Start background thread that polls screen context periodically.

        This keeps the screen cache fresh so gather() doesn't have to
        wait for a Screenpipe API call every time.
        """
        def _poll_loop():
            while True:
                try:
                    if self._screenpipe and self._screenpipe.is_available():
                        ctx = self._screenpipe.get_screen_context_filtered(
                            minutes=2,
                            max_chars=1500,
                            only_if_changed=True,
                        )
                        if ctx.get("changed", True):
                            with self._screen_lock:
                                self._screen_cache = ctx
                                self._screen_cache_time = time.time()

                            # Update project detection from screen
                            self._update_project_from_screen(ctx)
                except Exception as e:
                    logger.debug(f"[ACE] Screen poll error: {e}")

                time.sleep(ContextConfig.SCREEN_POLL_INTERVAL_S)

        thread = threading.Thread(
            target=_poll_loop,
            daemon=True,
            name="ace_screen_poll"
        )
        thread.start()
        logger.info("[ACE] Background screen polling started")

    def _update_project_from_screen(self, screen_data: Dict):
        """Detect active project from screen context changes."""
        window = screen_data.get("current_window", "")
        if not window:
            return

        # VS Code: "filename - ProjectFolder - Visual Studio Code"
        parts = window.split(" - ")
        if len(parts) >= 3 and "Visual Studio Code" in parts[-1]:
            project = parts[-2].strip()
            if project != self._detected_project:
                self._detected_project = project
                self._detected_project_time = time.time()
                logger.info(f"[ACE] Project detected from screen: {project}")

    def get_session_summary(self) -> str:
        """Get a summary of the current conversation session."""
        if not self._session_turns:
            return "No conversation yet."

        turn_count = len(self._session_turns)
        user_turns = sum(1 for t in self._session_turns if t.role == "user")
        duration = time.time() - self._session_start
        minutes = int(duration // 60)

        topics = set()
        for turn in self._session_turns:
            words = re.findall(r'\b[A-Z][a-z]{2,}\b', turn.content)
            topics.update(words[:3])

        return (
            f"Session: {turn_count} turns ({user_turns} user), "
            f"{minutes} min, topics: {', '.join(list(topics)[:10])}"
        )

    def shutdown(self):
        """Clean shutdown of the context engine."""
        self._executor.shutdown(wait=False)
        logger.info("[ACE] Context engine shut down")


# ============================================================================
#         STEP 4: SYSTEM PROMPT INJECTION FORMAT
# ============================================================================

def format_context_injection(ctx: GatheredContext) -> str:
    """
    Format gathered context into the system prompt injection string.

    This is the exact text that gets appended to the system prompt
    before every LLM call. The format is designed to be:
        - Clearly structured (LLM can parse sections)
        - Priority-ordered (most important first)
        - Concise (no wasted tokens)
        - Action-oriented (tells the LLM how to use the context)

    EXACT OUTPUT FORMAT:
    ===================================================================
    <CONTEXT>
    [User] Name: Elnur | Mood: neutral

    [Screen] Current app: VS Code | Window: agent.py - AURA
    ** ERRORS DETECTED on screen **
    Screen text: Traceback (most recent call last)...

    [Session] (12 turns, 8 min)
    User: Can you fix the import error in agent.py?
    Aura: I see the issue -- the module path changed. Let me...
    User: Great, now what about the tests?

    [Memory]
    [AMEM] User prefers direct communication, dislikes verbose output
    [EPISODIC] Yesterday: debugged CUDA OOM error in BroadMind training

    [Knowledge]
    Entity: BroadMind (project) -- custom AI architecture, v0.74-v0.78
    Related: Causeway (causes -> causal reasoning), FluxMind (part_of -> metalearning)

    [Project] Active project: AURA (in VS Code)

    [Documents]
    [RAG] From architecture_design.md: "The agent loop follows observe-plan-act..."

    [File] agent.py (first 500 chars):
    def main(): ...

    Use this context when relevant. Address the user by name. If errors
    are visible on screen, proactively offer help.
    </CONTEXT>
    ===================================================================
    """
    if not ctx.blocks:
        return ""

    sections = []

    # Header with user info
    header_parts = []
    if ctx.user_name:
        header_parts.append(f"Name: {ctx.user_name}")
    if ctx.mood and ctx.mood != "neutral":
        header_parts.append(f"Mood: {ctx.mood}")
    if header_parts:
        sections.append(f"[User] {' | '.join(header_parts)}")

    # Group blocks by source for clean formatting
    source_order = [
        ContextSource.SCREEN,
        ContextSource.CONVERSATION,
        ContextSource.AMEM,
        ContextSource.KG,
        ContextSource.PROJECT,
        ContextSource.RAG,
        ContextSource.FILE,
        ContextSource.URL,
    ]

    # Build section labels
    section_labels = {
        ContextSource.SCREEN: "Screen",
        ContextSource.USER_PROFILE: "User",
        ContextSource.CONVERSATION: "Session",
        ContextSource.AMEM: "Memory",
        ContextSource.EPISODIC: "Memory",
        ContextSource.KG: "Knowledge",
        ContextSource.RAG: "Documents",
        ContextSource.FILE: "File",
        ContextSource.URL: "Web",
        ContextSource.PROJECT: "Project",
    }

    # Organize blocks by source
    blocks_by_source: Dict[ContextSource, List[ContextBlock]] = {}
    for block in ctx.blocks:
        if block.source == ContextSource.USER_PROFILE:
            continue  # Already in header
        if block.source not in blocks_by_source:
            blocks_by_source[block.source] = []
        blocks_by_source[block.source].append(block)

    # Render in priority order
    for source in source_order:
        if source not in blocks_by_source:
            continue
        label = section_labels.get(source, source.value)
        source_blocks = blocks_by_source[source]

        if len(source_blocks) == 1 and len(source_blocks[0].content) < 200:
            # Compact single-line format
            sections.append(f"[{label}] {source_blocks[0].content}")
        else:
            # Multi-line format
            content_parts = [b.content for b in source_blocks]
            sections.append(f"[{label}]\n" + "\n".join(content_parts))

    # Footer instruction
    instructions = []
    instructions.append(
        "Use this context when relevant. "
        "Address the user by name if known."
    )
    if any(
        b.source == ContextSource.SCREEN
        and b.metadata.get("has_errors")
        for b in ctx.blocks
    ):
        instructions.append(
            "Errors detected on screen -- proactively offer help."
        )

    # Assemble
    body = "\n\n".join(sections)
    footer = " ".join(instructions)

    return f"<CONTEXT>\n{body}\n\n{footer}\n</CONTEXT>"


# ============================================================================
#         INTEGRATION POINT: How this replaces current code in agent.py
# ============================================================================

"""
INTEGRATION GUIDE
=================

Currently in agent.py, the chat() method (lines ~4680-5080) does:
    1. _build_aura_context(message) -> mood + thinking prefix
    2. Inline KG query (~4810)
    3. Inline RAG query (~4830)
    4. Inline A-MEM query (~4850)
    5. Inline Unified Memory query (~4880)
    6. Read user_profile.md (~4920)
    7. Assemble context_parts list
    8. Build system_prompt_addon string
    9. Pass to brain.think(message, system_prompt=system_prompt_addon)

The same thing is duplicated in chat_stream() (lines ~5300-5460).

REPLACE ALL OF THAT WITH:

    class AuraAgent:
        def __init__(self, ...):
            ...
            # Initialize the always-on context engine
            self.context_engine = AlwaysOnContextEngine(
                screenpipe_client=self._screenpipe,
                screen_reader=self._screen_reader,
                memory_retriever=self.memory_retriever,
                unified_memory=get_unified_memory(),
                kg_bridge=self.kg_bridge,
                tools=self.tools,
                data_dir=Path("data"),
            )
            # Start background screen polling
            self.context_engine.start_background_screen_polling()

        def chat(self, message: str, ...) -> str:
            # ... fast path, command handlers, etc. (unchanged) ...

            # ONE LINE replaces 200+ lines of inline context gathering:
            context = self.context_engine.gather(message)
            system_prompt_addon = context.to_system_prompt()

            # Still apply Soul prompt, PromptEvolution, TemplateLib, etc.
            if soul_prompt:
                system_prompt_addon = f"PERSONALITY:\\n{soul_prompt}\\n\\n{system_prompt_addon}"

            # LLM call (unchanged)
            response = self.brain.think(
                message,
                task_type=task_type,
                tone_modifier=tone_modifier,
                system_prompt=system_prompt_addon,
            )

            # Record for session memory
            self.context_engine.record_response(response, context)

            return response

        def chat_stream(self, message: str, ...) -> Generator:
            # Same pattern -- no more duplicated context code
            context = self.context_engine.gather(message)
            system_prompt_addon = context.to_system_prompt()
            ...


KEY IMPROVEMENTS OVER CURRENT SYSTEM
=====================================

1. SCREEN AWARENESS (completely missing today)
   - Current: Zero screen context. AURA is blind.
   - New: Screenpipe OCR text, active app, error detection, project inference.
   - Background polling keeps cache fresh (5s interval).

2. FILE AUTO-READ (completely missing today)
   - Current: User must use tools to read files.
   - New: Mention a path -> it's automatically read into context.
   - Safety: binary detection, sensitive dir filtering, size limits.

3. URL AUTO-FETCH (completely missing today)
   - Current: User must ask "search for" or use tools.
   - New: Paste a URL -> title + content summary auto-injected.

4. PROJECT DETECTION (completely missing today)
   - Current: No concept of "current project."
   - New: Inferred from window title, file paths, or world model.
   - Persists across messages until project changes.

5. CONVERSATION MEMORY (minimal today)
   - Current: LLM has chat history but no structured session summary.
   - New: Compressed session summary with topic tracking.
   - Old turns compressed to topic keywords, recent turns verbatim.

6. NO MORE DUPLICATE CODE
   - Current: Context gathering duplicated between chat() and chat_stream().
   - New: Single gather() call used by both.

7. PARALLEL GATHERING WITH TIMEOUT
   - Current: Sequential queries (KG, then RAG, then A-MEM, then Unified).
   - New: All queries launch in parallel, hard 800ms deadline.
   - Worst-case: some sources timeout, best sources still included.

8. SMART BUDGET MANAGEMENT
   - Current: ContextBudget is a simple token counter, no prioritization.
   - New: Priority-based allocation. User profile always included.
          Low-priority sources (URL, file) trimmed first.

9. RELEVANCE LEARNING
   - Current: No feedback loop. Same retrieval quality forever.
   - New: Tracks which sources the LLM actually used.
          Adjusts weights over time. Sources that are never cited get deprioritized.

10. STRUCTURED OUTPUT FORMAT
    - Current: Raw text concatenation with basic labels.
    - New: XML-tagged <CONTEXT> block with labeled sections.
           LLM can parse sections reliably. Instruction footer.


TOKEN BUDGET BREAKDOWN (4000 tokens total)
==========================================

Source          | Weight | Tokens | What goes here
----------------|--------|--------|-------------------------------------------
user_profile    | 0.08   |   320  | Name, preferences, relationship facts
conversation    | 0.05   |   200  | Session summary (last 3 turns + topics)
screen          | 0.15   |   600  | Current app, window, OCR text, errors
project         | 0.05   |   200  | Active project name + context
amem            | 0.15   |   600  | A-MEM + Episodic memories (via unified)
kg              | 0.12   |   480  | Knowledge graph entities + relationships
rag             | 0.10   |   400  | RAG document chunks
file            | 0.10   |   400  | Auto-read file contents
url             | 0.05   |   200  | Auto-fetched URL summaries

Priority cut order (when over budget):
    1. URL (40) -- cut first
    2. File (50)
    3. RAG (55)
    4. KG (60)
    5. Episodic (65)
    6. A-MEM (70)
    7. Project (80)
    8. Screen (85)
    9. Conversation (95)
   10. User Profile (100) -- NEVER cut


CONTEXT RELEVANCE LEARNING
===========================

After each response, the engine checks:
  - Did the LLM mention/cite any of the injected context?
  - Did the user respond positively (no correction, no retry)?
  - Which source types were used vs. ignored?

This updates per-source weight adjustments:
  - Cited source: +0.1 relevance boost (capped at +0.3)
  - Unhelpful response: -0.02 relevance penalty (floor at -0.2)
  - All adjustments decay by 0.95x per session

Over time, AURA learns:
  - "Screen context is very useful when user is coding"
  - "KG context is rarely cited for casual conversation"
  - "A-MEM memories are most useful when user references past"


ACTIVE PROJECT DETECTION
=========================

Detection cascade:
  1. Window title parsing (highest confidence)
     - VS Code: "file.py - ProjectName - Visual Studio Code"
     - Terminal: "user@host: ~/ProjectName"
     - Browser: "GitHub - user/ProjectName"
  2. File paths in message
     - Walk parent dirs looking for .git, package.json, pyproject.toml
  3. World model (world_state.json projects list)
  4. Recent conversation entities matching known project names

Persistence:
  - Detected project sticks until a new one is detected
  - Background screen polling updates project on app/window changes
  - Project change triggers a contextual note in next gather()
"""
