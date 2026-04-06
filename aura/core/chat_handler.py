"""Chat flow mixin — _prepare_chat, _finalize_chat, chat entry point.

Extracted from agent.py (2026-04-06) to reduce class size.
All methods assume self has: brain, tools, memory, monologue, metacognition,
identity, neurodream, aura_enabled, _soul, _visible_thinking, _temporal_lock,
_prev_message, _prev_response, kg_bridge, _kg_queue_lock, skill_library,
thinker, use_fastpath, fast_path_handler, context_engine.
"""

import json
import logging
import time
import concurrent.futures
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ChatMixin:
    """Mixin providing the main chat flow: prepare, finalize, and chat entry point."""

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
        from aura.core.thought_recorder import record_thought as _record_thought
        from aura.pools import bg_pool as _bg_pool_fn
        _AGENT_EXECUTOR = _bg_pool_fn()
        from aura.memory.unified_memory import get_unified_memory
        from aura.tools import get_tone_modifier, SleepPhase
        from aura.brain import TaskType

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
                    from aura.memory.context_budget import ContextBudget
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
                self.thinker.touch()  # Reset staleness clock on new user message
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
        from aura.pools import bg_pool as _bg_pool_fn
        _AGENT_EXECUTOR = _bg_pool_fn()
        from aura.memory.unified_memory import get_unified_memory

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

        try:
            from aura.consciousness.reasoning_templates import (
                get_template_library,
                build_trace_from_mcts,
            )
            TEMPLATE_LIBRARY_AVAILABLE = True
        except ImportError:
            TEMPLATE_LIBRARY_AVAILABLE = False

        from aura.config import Config

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

        # ===== SEARCH CONTEXT INJECTION =====
        # If this message might need current information and direct_search didn't
        # catch it, try a web search and inject results into the system prompt.
        # This prevents the LLM from hallucinating about current events.
        _search_context = None
        try:
            _search_query = self._needs_web_search(message)
            if _search_query:
                logger.debug(f"[SearchInject] Detected search-needed query: '{_search_query[:50]}'")
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
                    logger.debug(f"[SearchInject] Injected {len(_search_result['results'])} results into prompt")
        except (ImportError, AttributeError, TypeError, ValueError,
                ConnectionError, TimeoutError, OSError) as _sinj_err:
            logger.debug(f"[SearchInject] Failed: {_sinj_err}")

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

                # Compute cheap quality heuristics so the bandit learns
                # from more than just latency (coherence + judge_score)
                if compute_quality_metrics is not None:
                    try:
                        quality = compute_quality_metrics(message, response)
                        metrics.update(quality)
                    except Exception as qe:
                        logger.debug(f"[StrategyBandit] Quality metrics error: {qe}")

                # Always record basic outcome with latency + quality
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
