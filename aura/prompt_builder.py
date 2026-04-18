"""System prompt builder — extracted from brain.py.

Builds the layered system prompt that OllamaBrain passes to LLM calls.
Owns caching state for subsystem additions and project context detection.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from .config import Config
from .identity import get_identity_prompt

logger = logging.getLogger(__name__)

MAX_SYSTEM_PROMPT_CHARS = 25000


# ---------------------------------------------------------------------------
# Standalone helpers (also used by brain.py for neuro LLM options)
# ---------------------------------------------------------------------------

def classify_budget(query: str) -> int:
    """Classify the expected response budget from query complexity."""
    q = query.lower().strip()
    if q in ("yes", "no", "ok", "thanks", "thx", "bye", "k"):
        return Config.BUDGET_SMALL
    if any(kw in q for kw in ("explain", "analyze", "compare", "research", "write", "implement")):
        return Config.BUDGET_LARGE
    return Config.BUDGET_MEDIUM


def build_budget_instruction(budget: int) -> str:
    return f"\n\n[Response budget: ~{budget} tokens. Be appropriately concise.]"


# ---------------------------------------------------------------------------
# SystemPromptBuilder
# ---------------------------------------------------------------------------

class SystemPromptBuilder:
    """Builds the multi-layer system prompt for OllamaBrain.

    Layers (in order):
        identity → caller system_prompt → web search instruction →
        design system → subsystem context (TTL-cached) → project context →
        semantic codebase → skill catalog → deferred tools → emotional tone →
        episodic memory → budget instruction → safety cap
    """

    def __init__(self) -> None:
        # Subsystem additions cache
        self._cached_system_additions: Optional[str] = None
        self._system_additions_ts: float = 0.0
        self._system_additions_lock = threading.RLock()

        # Project context cache
        self._project_ctx_cache = None
        self._project_ctx_ts: float = 0.0
        self._project_ctx_cwd: str = ""

        # Codebase context cache (expensive index search — 5 min TTL)
        self._codebase_ctx_cache: Optional[str] = None
        self._codebase_ctx_ts: float = 0.0
        self._codebase_ctx_prompt: str = ""
        _CODEBASE_CTX_TTL = 300  # 5 minutes

        # Episodic memory cache (semantic + BM25 search — 30s TTL)
        self._episodic_cache: Optional[str] = None
        self._episodic_cache_ts: float = 0.0
        self._episodic_cache_prompt: str = ""
        _EPISODIC_CTX_TTL = 30

        # Skill catalog cache (10 min TTL — skills rarely change mid-session)
        self._skill_catalog_cache: Optional[str] = None
        self._skill_catalog_ts: float = 0.0
        _SKILL_CATALOG_TTL = 600

        # Deferred tools cache (10 min TTL)
        self._deferred_tools_cache: Optional[str] = None
        self._deferred_tools_ts: float = 0.0
        _DEFERRED_TOOLS_TTL = 600

    def build(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tone_modifier: Optional[str],
        *,
        action_mode: Optional[str] = None,
        alma_enabled: bool = False,
        auto_emotional_tone: bool = True,
        skill_list_fn: Optional[Callable[[], List[dict]]] = None,
    ) -> str:
        """Build the complete system prompt for a think/think_stream call."""

        identity_prompt = get_identity_prompt()
        full = f"{identity_prompt}\n\n{system_prompt}" if system_prompt else identity_prompt

        # === WEB SEARCH INSTRUCTION ===
        full = f"{full}\n\n{self._web_search_instruction()}"

        # === DESIGN SYSTEM INJECTION ===
        full = self._inject_design_system(full, action_mode)

        # === SUBSYSTEM CONTEXT (cached, TTL=12s) ===
        self._observe_user_message(prompt)
        sys_additions = self._get_cached_system_additions()
        if sys_additions:
            full = f"{full}\n\n{sys_additions}"

        # === PROJECT CONTEXT ===
        full = self._inject_project_context(full, prompt)

        # === SEMANTIC CODEBASE CONTEXT ===
        full = self._inject_codebase_context(full, prompt)

        # === PROGRESSIVE SKILL CATALOG ===
        full = self._inject_skill_catalog(full, skill_list_fn)

        # === DEFERRED TOOL LISTING ===
        full = self._inject_deferred_tools(full)

        # Skill/tool discovery hint
        hint = "Use load_skill or tool_search when a task matches a listed skill or requires a specialized tool."
        if len(full) + len(hint) + 4 < MAX_SYSTEM_PROMPT_CHARS:
            full = f"{full}\n\n{hint}"

        # === EMOTIONAL STYLE ===
        full = self._inject_emotional_style(full, tone_modifier, alma_enabled, auto_emotional_tone)

        # === EPISODIC MEMORY AUTO-RECALL ===
        full = self._inject_episodic_memory(full, prompt)

        # === BUDGET INSTRUCTION ===
        budget = classify_budget(prompt)
        full = f"{full}{build_budget_instruction(budget)}"

        # === SAFETY CAP ===
        if len(full) > MAX_SYSTEM_PROMPT_CHARS:
            logger.warning(
                f"[PromptBuilder] System prompt too large ({len(full)} chars), "
                f"truncating to {MAX_SYSTEM_PROMPT_CHARS}"
            )
            cut = full[:MAX_SYSTEM_PROMPT_CHARS].rfind('\n\n')
            if cut > MAX_SYSTEM_PROMPT_CHARS // 2:
                full = full[:cut]
            else:
                full = full[:MAX_SYSTEM_PROMPT_CHARS]
            full += "\n\n[System context truncated for length]"

        return full

    # ------------------------------------------------------------------
    # Private layer methods
    # ------------------------------------------------------------------

    @staticmethod
    def _web_search_instruction() -> str:
        return (
            "IMPORTANT: If the user asks about something you are not sure about, "
            "something recent, current events, news, real-time data (dates, prices, "
            "weather, scores, stock prices, exchange rates), or asks you to look "
            "something up or verify information — you MUST use the web_search or "
            "tavily tool to search the internet FIRST. Do NOT guess or make up "
            "answers. Always verify uncertain facts by searching."
        )

    @staticmethod
    def _inject_design_system(full: str, action_mode: Optional[str]) -> str:
        try:
            from aura.prompts.design_system import DESIGN_SYSTEM_MODES, DESIGN_SYSTEM_PROMPT
            if action_mode and action_mode in DESIGN_SYSTEM_MODES:
                full = f"{full}\n\n{DESIGN_SYSTEM_PROMPT}"
                logger.debug(f"[PromptBuilder] Design system prompt injected for mode: {action_mode}")
        except ImportError:
            logger.debug("[PromptBuilder] aura.prompts.design_system not available")
        return full

    def _observe_user_message(self, prompt: str) -> None:
        """Side-effect: let subsystems observe the user message."""
        try:
            if Config.MULTI_USER_ENABLED:
                from aura.multi_user import get_multi_user_manager
                manager = get_multi_user_manager()
                user_model = manager.get_active_user_model()
                if user_model:
                    user_model.observe_message(prompt, role="user")
            else:
                from aura.proactive.theory_of_mind import get_theory_of_mind
                tom = get_theory_of_mind()
                tom.observe_message(prompt, role="user")
        except Exception as e:
            logger.debug(f"[PromptBuilder] ToM observe failed: {e}")
        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
            get_intrinsic_motivation().record_interaction()
        except Exception as e:
            logger.debug(f"[PromptBuilder] Motivation record failed: {e}")

    def _get_cached_system_additions(self) -> str:
        """Return TTL-cached subsystem additions (12s TTL). Thread-safe."""
        with self._system_additions_lock:
            if (
                self._cached_system_additions is not None
                and (time.time() - self._system_additions_ts) < 12.0
            ):
                return self._cached_system_additions

            additions = []
            _PER_SOURCE_CAP = 1000
            _TOTAL_CAP = 2000

            def _cap(text: str, source: str) -> str:
                if text and len(text) > _PER_SOURCE_CAP:
                    logger.warning(
                        f"[PromptBuilder] Subsystem '{source}' returned {len(text)} chars, "
                        f"capping to {_PER_SOURCE_CAP}"
                    )
                    cut = text[:_PER_SOURCE_CAP].rfind('. ')
                    return text[:cut + 1] if cut > _PER_SOURCE_CAP // 2 else text[:_PER_SOURCE_CAP]
                return text

            # Learned context (NeuroDream)
            try:
                from aura.tools.neurodream import get_neurodream
                nd = get_neurodream()
                learned_ctx = nd.get_learned_context_prompt()
                if learned_ctx:
                    additions.append(_cap(learned_ctx, "NeuroDream"))
            except Exception as e:
                logger.debug(f"[PromptBuilder] NeuroDream context failed: {e}")

            # Calendar context
            try:
                from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
                cm = get_calendar_monitor()
                cal_ctx = cm.get_context_for_prompt()
                if cal_ctx:
                    additions.append(_cap(cal_ctx, "CalendarMonitor"))
            except Exception as e:
                logger.debug(f"[PromptBuilder] Calendar context failed: {e}")

            # Metacognitive self-model
            try:
                from aura.consciousness.metacognition import get_metacognitive_engine
                mc = get_metacognitive_engine()
                self_model_ctx = mc.get_self_model_prompt()
                if self_model_ctx:
                    additions.append(_cap(self_model_ctx, "MetacognitionEngine"))
            except Exception as e:
                logger.debug(f"[PromptBuilder] Metacognition context failed: {e}")

            # User model (Theory of Mind)
            try:
                if Config.MULTI_USER_ENABLED:
                    from aura.multi_user import get_multi_user_manager
                    manager = get_multi_user_manager()
                    user_model = manager.get_active_user_model()
                    if user_model:
                        user_model_ctx = user_model.get_context_for_prompt()
                        if user_model_ctx:
                            additions.append(_cap(user_model_ctx, "TheoryOfMind/MultiUser"))
                else:
                    from aura.proactive.theory_of_mind import get_theory_of_mind
                    tom = get_theory_of_mind()
                    user_model_ctx = tom.get_context_for_prompt()
                    if user_model_ctx:
                        additions.append(_cap(user_model_ctx, "TheoryOfMind"))
            except Exception as e:
                logger.debug(f"[PromptBuilder] ToM context failed: {e}")

            # Intrinsic motivation
            try:
                from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
                im = get_intrinsic_motivation()
                motivation_ctx = im.get_context_for_prompt()
                if motivation_ctx:
                    additions.append(_cap(motivation_ctx, "IntrinsicMotivation"))
            except Exception as e:
                logger.debug(f"[PromptBuilder] Motivation context failed: {e}")

            # World model
            try:
                from aura.consciousness.world_model import get_world_model
                wm = get_world_model()
                world_ctx = wm.get_context_summary()
                if world_ctx:
                    additions.append(_cap(world_ctx, "WorldModel"))
            except Exception as e:
                logger.debug(f"[PromptBuilder] World model context failed: {e}")

            result = "\n\n".join(additions)
            if len(result) > _TOTAL_CAP:
                logger.warning(
                    f"[PromptBuilder] System additions total {len(result)} chars "
                    f"(from {len(additions)} sources), truncating to {_TOTAL_CAP}"
                )
                result = result[:_TOTAL_CAP]
            self._cached_system_additions = result
            self._system_additions_ts = time.time()
            return result

    def _inject_project_context(self, full: str, prompt: str) -> str:
        try:
            _now = time.time()
            _cwd = os.getcwd()
            if (
                self._project_ctx_cache is None
                or _now - self._project_ctx_ts > 60
                or self._project_ctx_cwd != _cwd
            ):
                from aura.tools.project_context import detect_and_load_context
                self._project_ctx_cache = detect_and_load_context(_cwd)
                self._project_ctx_ts = _now
                self._project_ctx_cwd = _cwd

            ctx = self._project_ctx_cache
            if ctx and ctx.get("has_aura_md"):
                full = f"{full}\n\n## Active Project Context\n{ctx['aura_md_content']}"
            elif ctx and ctx.get("project_type") and ctx["project_type"] != "unknown":
                parts = [f"**Type:** {ctx['project_type']}"]
                if ctx.get("stack"):
                    parts.append(f"**Stack:** {', '.join(ctx['stack'])}")
                if ctx.get("frameworks"):
                    parts.append(f"**Frameworks:** {', '.join(ctx['frameworks'])}")
                if ctx.get("key_files"):
                    parts.append(f"**Key Files:** {', '.join(ctx['key_files'][:10])}")
                full = f"{full}\n\n## Auto-Detected Project Context\n" + "\n".join(parts)
        except Exception as e:
            logger.debug(f"[PromptBuilder] Project context failed: {e}")
        return full

    def _inject_codebase_context(self, full: str, prompt: str) -> str:
        if len(full) >= MAX_SYSTEM_PROMPT_CHARS - 1000:
            return full
        # TTL cache: reuse result if same prompt within 5 minutes
        _now = time.time()
        if (
            self._codebase_ctx_cache is not None
            and _now - self._codebase_ctx_ts < 300
            and self._codebase_ctx_prompt == prompt
        ):
            if self._codebase_ctx_cache:
                full = f"{full}\n\n{self._codebase_ctx_cache}"
            return full
        try:
            from aura.tools.codebase_index import CodebaseIndex
            _cwd = os.getcwd()
            _idx_db = Path("data/codebase_index/index.db")
            _idx_db_legacy = Path(_cwd) / ".aura" / "index.db"
            code_section = ""
            if (_idx_db.exists() or _idx_db_legacy.exists()) and len(full) < MAX_SYSTEM_PROMPT_CHARS - 1000:
                idx = CodebaseIndex(_cwd)
                try:
                    if idx.stats()["total_chunks"] > 0:
                        relevant = idx.search(prompt, top_k=3)
                        if relevant and relevant[0]["score"] > 0.3:
                            ctx_parts = []
                            for r in relevant:
                                if r["score"] > 0.3:
                                    ctx_parts.append(
                                        f"**{r['file_path']}:{r['line_start']}** "
                                        f"({r['kind']} `{r['name']}`):\n"
                                        f"```\n{r['content'][:300]}\n```"
                                    )
                            if ctx_parts:
                                code_section = "## Relevant Code\n" + "\n\n".join(ctx_parts)
                finally:
                    idx.close()
            self._codebase_ctx_cache = code_section
            self._codebase_ctx_ts = _now
            self._codebase_ctx_prompt = prompt
            if code_section:
                full = f"{full}\n\n{code_section}"
        except Exception as e:
            logger.debug(f"[PromptBuilder] Code context retrieval failed: {e}")
        return full

    def _inject_skill_catalog(
        self, full: str, skill_list_fn: Optional[Callable] = None
    ) -> str:
        if len(full) >= MAX_SYSTEM_PROMPT_CHARS - 1500:
            return full
        # TTL cache: skills rarely change mid-session (10 min)
        _now = time.time()
        if self._skill_catalog_cache is not None and _now - self._skill_catalog_ts < 600:
            if self._skill_catalog_cache:
                full = f"{full}\n\n{self._skill_catalog_cache}"
            return full
        try:
            catalog = ""
            if skill_list_fn and callable(skill_list_fn):
                skill_summaries = skill_list_fn()
                if skill_summaries:
                    skill_lines = "\n".join(
                        f"- {s['name']}: {s.get('description', 'no description')}"
                        for s in skill_summaries
                    )
                    if len(skill_lines) > 500:
                        skill_lines = skill_lines[:500].rsplit("\n", 1)[0]
                    catalog = (
                        f"[Available Skills - use load_skill tool to "
                        f"load full procedures]\n{skill_lines}"
                    )
            self._skill_catalog_cache = catalog
            self._skill_catalog_ts = _now
            if catalog:
                full = f"{full}\n\n{catalog}"
        except Exception as e:
            logger.debug(f"[PromptBuilder] Skill catalog injection failed: {e}")
        return full

    def _inject_deferred_tools(self, full: str) -> str:
        if len(full) >= MAX_SYSTEM_PROMPT_CHARS - 1000:
            return full
        # TTL cache: deferred tools rarely change (10 min)
        _now = time.time()
        if self._deferred_tools_cache is not None and _now - self._deferred_tools_ts < 600:
            if self._deferred_tools_cache:
                full = f"{full}\n\n{self._deferred_tools_cache}"
            return full
        try:
            tools_section = ""
            from aura.tools.loader import get_deferred_tool_list
            deferred_tools = get_deferred_tool_list()
            if deferred_tools:
                tool_lines = "\n".join(
                    f"- {t['name']}: {t.get('description', 'no description')}"
                    for t in deferred_tools
                )
                if len(tool_lines) > 800:
                    tool_lines = tool_lines[:800].rsplit("\n", 1)[0]
                tools_section = (
                    f"[Additional Tools - use tool_search to "
                    f"find and activate]\n{tool_lines}"
                )
            self._deferred_tools_cache = tools_section
            self._deferred_tools_ts = _now
            if tools_section:
                full = f"{full}\n\n{tools_section}"
        except ImportError:
            logger.debug("[PromptBuilder] get_deferred_tool_list not available yet")
        except Exception as e:
            logger.debug(f"[PromptBuilder] Deferred tool listing failed: {e}")
        return full

    @staticmethod
    def _inject_emotional_style(
        full: str,
        tone_modifier: Optional[str],
        alma_enabled: bool,
        auto_emotional_tone: bool,
    ) -> str:
        if tone_modifier:
            full = f"{full}\n\n{tone_modifier}"
        elif alma_enabled and auto_emotional_tone:
            try:
                from aura.emotion.integration import get_emotional_style_prompt
                alma_style = get_emotional_style_prompt()
                if alma_style:
                    full = f"{full}\n\n{alma_style}"
            except Exception as e:
                logger.debug(f"[PromptBuilder] ALMA style prompt failed: {e}")
        return full

    def _inject_episodic_memory(self, full: str, prompt: str) -> str:
        if len(prompt) <= 25 or len(full) >= MAX_SYSTEM_PROMPT_CHARS - 500:
            return full
        # TTL cache: reuse if same prompt within 30s
        _now = time.time()
        if (
            self._episodic_cache is not None
            and _now - self._episodic_cache_ts < 30
            and self._episodic_cache_prompt == prompt
        ):
            if self._episodic_cache:
                full = f"{full}{self._episodic_cache}"
            return full
        try:
            from aura.memory.unified_memory import get_unified_memory
            um_results = get_unified_memory().query(prompt, k=3)
            memory_section = ""
            if um_results:
                memory_section = "\n\n## Relevant Past Context\n"
                for m in um_results:
                    ts = m.metadata.get("created_at", "")[:10] if m.metadata.get("created_at") else ""
                    memory_section += f"- [{ts}] {m.content[:120]}\n"

            # Track hand finding references for adaptive scheduling
            try:
                from aura.hands.manager import get_hand_manager
                mgr = get_hand_manager()
                for m in um_results:
                    hand_name = m.metadata.get("hand") if hasattr(m, 'metadata') and m.metadata else None
                    if hand_name:
                        mgr.record_finding_referenced(hand_name)
            except Exception as e:
                logger.debug(f"[PromptBuilder] Failed to record hand finding references for adaptive scheduling: {e}")

            self._episodic_cache = memory_section
            self._episodic_cache_ts = _now
            self._episodic_cache_prompt = prompt
            if memory_section:
                full = f"{full}{memory_section}"
        except Exception as e:
            logger.debug(f"[PromptBuilder] Episodic memory recall failed: {e}")
        return full
