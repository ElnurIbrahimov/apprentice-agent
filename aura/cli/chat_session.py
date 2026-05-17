"""ChatSession — extracted from run_chat_mode() god-function.

Encapsulates all CLI chat session state and the main input loop
as a proper class with named methods instead of 900+ lines of
nested closures and local variables.
"""
from __future__ import annotations

import atexit
import logging
import os
from typing import Any, Optional

from .chat_session_execution import SessionExecutionController
from .chat_session_runtime import SessionRuntimeController
from .chat_session_signals import SessionSignalController
from .chat_session_status import SessionStatusController, show_startup_diagnostics

logger = logging.getLogger(__name__)


class ChatSession:
    """Manages a single interactive CLI chat session."""

    def __init__(
        self,
        agent: Any,
        *,
        speak: bool = False,
        trust: bool = False,
        model: Optional[str] = None,
        verbose: bool = False,
        tier: Optional[str] = None,
        bridge: Any = None,
        preference: Optional[str] = None,
    ) -> None:
        # ── Store basic references ──
        from .display import console
        self.agent = agent
        self.console = console
        self.speak = speak
        self.verbose = verbose
        self.bridge = bridge

        # ── Phase 1: Display (theme, banner, diagnostics) ──
        self._init_display(agent)

        # ── Phase 2: Session core (project, bootstrap, permissions, agentic loop) ──
        boot, aura_config, project_root = self._init_session_core(
            agent=agent, tier=tier, model=model, trust=trust, preference=preference,
        )

        # ── Phase 3: Checkpoint, optional subsystems, steering, activity log ──
        self._init_subsystems_and_steering(
            aura_config=aura_config, project_root=project_root, verbose=verbose,
        )

        # ── Phase 4: Context, state, controllers, UI, channels ──
        self._init_ui_and_state(
            agent=agent, speak=speak, verbose=verbose,
            boot=boot, aura_config=aura_config, project_root=project_root,
            bridge=bridge,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Initialization phases
    # ═══════════════════════════════════════════════════════════════════════

    def _init_display(self, agent: Any) -> None:
        """Phase 1: Theme, banner, and startup diagnostics."""
        from .display import show_banner, show_welcome_info
        from .themes import load_theme_preference, set_theme

        saved_theme = load_theme_preference()
        set_theme(saved_theme)

        _quiet = os.environ.get("AURA_QUIET") == "1"
        if _quiet:
            try:
                from aura import __version__ as _v
                _m = agent.brain._model_override or "auto"
                self.console.print(
                    f"  [dim]aura v{_v} · {_m} · {os.getcwd()}[/dim]"
                )
            except Exception:
                logger.debug("quiet_banner_failed", exc_info=True)
        else:
            show_banner()
            show_welcome_info(agent)
            show_startup_diagnostics(self.console)

    def _init_session_core(
        self, *, agent: Any, tier: Optional[str], model: Optional[str],
        trust: bool, preference: Optional[str],
    ) -> tuple:
        """Phase 2: Project detection, session bootstrap, permissions, agentic loop.

        Returns (boot, aura_config, project_root).
        """
        from aura.core.agentic_loop import AgenticLoop
        from aura.core.session import AgenticSession

        from .session_bootstrap import build_permission_manager, build_session_bootstrap

        # ── Project detection ──
        self._project_type = ""
        try:
            from aura.tools.project_context import detect_and_load_context
            ctx = detect_and_load_context(".")
            self._project_type = ctx.get("project_type", "") if isinstance(ctx, dict) else ""
        except ImportError:
            logger.warning("project_context unavailable — project type detection disabled")
        except (OSError, ValueError, KeyError, TypeError):
            logger.debug("project_type_detection_failed", exc_info=True)

        from types import SimpleNamespace
        boot = build_session_bootstrap(
            SimpleNamespace(tier=tier, budget=None, model=model),
            brain=agent.brain,
        )
        project_root = boot.project_root
        project_context = boot.project_context
        aura_config = boot.aura_config

        # ── Permissions ──
        self.perm_mode = "full_auto" if trust else "auto_edit"
        self.permissions = build_permission_manager(
            aura_config=aura_config,
            trust=trust,
            default_mode="auto_edit",
            confirm_callback=self._cli_confirm,
        )

        # ── Session ──
        self.agentic_session = AgenticSession()
        self._session_initialized = False
        atexit.register(self._save_session_if_initialized)

        # ── AgenticLoop ──
        self.agentic = AgenticLoop(
            brain=agent.brain,
            project_root=project_root,
            permissions=self.permissions,
            model_override=boot.model,
            max_iterations=aura_config.get("max_iterations", 25),
            budget_usd=boot.budget,
            context=project_context,
            session=self.agentic_session,
            aura_config=aura_config,
            router=boot.router,
        )

        # ── Neural routing preference ──
        _PREF_MAP = {"fast": "prefer-fast", "balanced": "balanced", "quality": "prefer-quality"}
        self._routing_preference = _PREF_MAP.get(preference or "balanced", "balanced")
        agent.brain._routing_preference = self._routing_preference

        return boot, aura_config, project_root

    def _init_subsystems_and_steering(
        self, *, aura_config: Any, project_root: str, verbose: bool,
    ) -> None:
        """Phase 3: Checkpoint manager, optional subsystems, steering, activity log."""
        from .checkpoint import CheckpointManager

        # ── Checkpoint ──
        try:
            self.checkpoint_mgr: Optional[Any] = CheckpointManager()
        except (OSError, PermissionError):
            self.checkpoint_mgr = None
        if self.checkpoint_mgr:
            self.agentic._checkpoint_mgr = self.checkpoint_mgr
            self.agentic.executor._checkpoint_mgr = self.checkpoint_mgr

        # ── Optional subsystems ──
        self._load_optional_subsystems(aura_config=aura_config, project_root=project_root, verbose=verbose)

        # ── Steering ──
        from .steering import SteeringQueue
        self.steering = SteeringQueue()
        self.steering.set_preempt_callback(self.agentic.cancel)

        # ── Activity log ──
        try:
            from .activity_log import ActivityLog
            self.activity_log: Optional[Any] = ActivityLog()
        except ImportError:
            logger.warning("ActivityLog unavailable — session activity logging disabled")
            self.activity_log = None
        except OSError:
            logger.warning("ActivityLog failed to initialize (OS error)")
            self.activity_log = None

    def _init_ui_and_state(
        self, *, agent: Any, speak: bool, verbose: bool,
        boot: Any, aura_config: Any, project_root: str, bridge: Any,
    ) -> None:
        """Phase 4: Context, state variables, controllers, status bar, prompt, channels."""
        from .context_bar import get_context_limit
        from .display import show_info
        from .input import create_session
        from .model_picker import update_model_roles_from_config

        # ── Build CLIContext and ConversationManager sync ──
        self._setup_context(agent=agent, speak=speak, verbose=verbose)

        # ── State variables ──
        self.apply_model_override(boot.model)
        self.session_title = ""
        self.msg_count = 0
        self.token_used = 0
        self.token_limit = get_context_limit(self.current_model)

        self._last_ctrl_c_time = 0.0
        self._last_ipc_heartbeat = 0.0
        self._daemon_unreachable_until = 0.0
        self._ipc_backoff_seconds = 5.0
        self._injected_input: Optional[str] = None

        import threading
        self._channel_lock = threading.Lock()

        self._pending_follow_up: Optional[str] = None
        self._follow_up_depth = 0
        self._MAX_FOLLOW_UP_DEPTH = 3
        self._streamer_displayed = False
        self.last_user_input = ""

        self._status_controller = SessionStatusController(
            console=self.console,
            cli_ctx=self._cli_ctx,
            steering_queue=self.steering,
            create_background_indicator=self._create_background_indicator,
            create_research_indicator=self._create_research_indicator,
            create_mood_indicator=self._create_mood_indicator,
        )
        self._signal_controller = SessionSignalController(self)
        self._execution_controller = SessionExecutionController(self)
        self._runtime_controller = SessionRuntimeController(self)

        # ── Permission banner + initial status bar ──
        self._show_perm_banner(self.perm_mode)
        self._show_bar(
            model=self.current_model, project_type=self._project_type,
            session_title=self.session_title, message_count=self.msg_count,
            token_used=self.token_used, token_limit=self.token_limit,
            permission_mode=self.perm_mode,
        )

        if speak:
            show_info("Voice output enabled")

        update_model_roles_from_config()

        # ── prompt_toolkit session ──
        self._pt_session = create_session()

        # ── Channel bridge setup ──
        if bridge:
            from .chat_loop import _display_channel_message
            show_info(f"Channel bridge active: {', '.join(s['channel'] for s in bridge.status())}")
            def _channel_notify(msg: Any) -> None:
                _display_channel_message(self.console, msg)
            bridge.set_on_message_callback(_channel_notify)

        # ── Store config references ──
        self._aura_config = aura_config
        self._project_root = project_root
        self._auto_test_enabled = bool(aura_config.get("auto_test", False)) if aura_config else False

    # ── Optional subsystem loading ────────────────────────────────────────

    def _load_optional_subsystems(self, *, aura_config: Any, project_root: str, verbose: bool) -> None:
        """Load optional subsystem classes (ImportError-safe) and instantiate them."""

        def _unavailable_indicator(name: str):
            """Return a no-op indicator that logs a one-time warning."""
            _warned = False
            def _indicator(*a: Any, **k: Any) -> str:
                nonlocal _warned
                if not _warned:
                    logger.debug(f"{name} indicator called but subsystem unavailable")
                    _warned = True
                return ""
            _indicator.__name__ = f"_{name}_unavailable"
            return _indicator

        try:
            from .background import (
                BackgroundManager,
                create_background_indicator,
                notify_completion,
            )
            self._BackgroundManager = BackgroundManager
            self._notify_completion = notify_completion
            self._create_background_indicator = create_background_indicator
        except ImportError:
            logger.warning("BackgroundManager unavailable — background tasks disabled")
            self._BackgroundManager = None
            self._notify_completion = None
            self._create_background_indicator = _unavailable_indicator("background")

        try:
            from .research_mode import ResearchContext, create_research_indicator
            self._ResearchContext = ResearchContext
            self._create_research_indicator = create_research_indicator
        except ImportError:
            logger.warning("ResearchContext unavailable — /research mode disabled")
            self._ResearchContext = None
            self._create_research_indicator = _unavailable_indicator("research")

        try:
            from .hooks import HookEvent, HookManager
            self._HookManager = HookManager
            self._HookEvent = HookEvent
        except ImportError:
            logger.warning("HookManager unavailable — hooks from AURA.md will not fire")
            self.console.print("  [dim yellow]Hooks unavailable[/dim yellow]")
            self._HookManager = None
            self._HookEvent = None

        try:
            from .mood_display import create_mood_indicator
            self._create_mood_indicator = create_mood_indicator
        except ImportError:
            logger.warning("mood_display unavailable — mood indicator disabled")
            self._create_mood_indicator = _unavailable_indicator("mood")

        self.bg_manager = self._BackgroundManager() if self._BackgroundManager else None
        if self.bg_manager and self._notify_completion:
            self.bg_manager.set_completion_callback(self._notify_completion)

        self.research_ctx = self._ResearchContext() if self._ResearchContext else None

        self.hook_mgr = self._HookManager() if self._HookManager else None
        if self.hook_mgr and aura_config:
            # Built-in hooks (shipped with Aura) load unconditionally.
            self.hook_mgr.load_builtin_hooks(aura_config)
            # User-defined hooks from AURA.md require project trust to prevent a
            # cloned repo from auto-executing arbitrary shell commands.
            self._maybe_load_project_hooks(aura_config, project_root)

        # Always install a DisclosureManager so tool output gets wrapped in
        # collapsible sections. --verbose flips default_expanded=True so the
        # full content shows inline; otherwise sections render as one-line
        # summaries and the scroll stays clean.
        from . import display as _display_mod
        from .disclosure import DisclosureManager
        _display_mod._disclosure = DisclosureManager(default_expanded=bool(verbose))
        if verbose:
            _display_mod._disclosure.set_verbose(True)

        if self.hook_mgr:
            self.hook_mgr.fire(self._HookEvent.SESSION_START, {"project_root": project_root})

    def _maybe_load_project_hooks(self, aura_config: dict, project_root: str) -> None:
        """Load AURA.md hooks after a first-run per-project trust prompt.

        Trust is keyed by (project_root, sha256(AURA.md)) — changing AURA.md
        re-prompts, matching the VSCode tasks.json trust model.
        """
        import os as _os

        hooks = aura_config.get("hooks") if aura_config else None
        if not hooks:
            return  # No hooks to load — nothing to gate.

        # AURA.md lives at project_root/AURA.md by convention (see
        # aura/core/context.py:_load_aura_md). If the file isn't where we
        # expect, fall through and load the hooks anyway; the trust check
        # can't prove the source.
        aura_md_path = _os.path.join(project_root, "AURA.md")
        if not _os.path.isfile(aura_md_path):
            try:
                self.hook_mgr.load_from_config(aura_config)
            except Exception:
                logger.debug("hook_load_failed_no_aura_md", exc_info=True)
            return

        try:
            from .project_trust import is_trusted, mark_trusted
        except ImportError:
            # Fail closed: if the trust store module is missing, skip hooks
            # rather than loading unreviewed ones.
            logger.warning("project_trust unavailable — skipping AURA.md hooks")
            return

        if is_trusted(project_root, aura_md_path):
            self.hook_mgr.load_from_config(aura_config)
            return

        # Untrusted — ask the user.
        from .permissions_dialog import request_project_trust
        choice = request_project_trust(self.console, project_root, aura_md_path, hooks)
        if choice == "trust":
            mark_trusted(project_root, aura_md_path)
            self.hook_mgr.load_from_config(aura_config)
        elif choice == "skip_hooks":
            self.console.print(
                "  [dim yellow]Skipping AURA.md hooks for this session.[/dim yellow]"
            )
        else:  # "abort"
            self.console.print("  [red]Aborted by user.[/red]")
            import sys as _sys
            _sys.exit(1)

    # ── Context setup ─────────────────────────────────────────────────────

    def _setup_context(self, *, agent: Any, speak: bool, verbose: bool) -> None:
        """Build CLIContext, sync ConversationManager, and resume session if requested."""
        from .context import CLIContext, set_ctx
        from .display import show_info

        # ── Build CLIContext and publish it ──
        cli_ctx = CLIContext(
            agent=agent,
            agentic_loop=self.agentic,
            permissions=self.permissions,
            session=self.agentic_session,
            bg_manager=self.bg_manager,
            research_ctx=self.research_ctx,
            hook_manager=self.hook_mgr,
            steering=self.steering,
            speak=speak,
            verbose=verbose,
            resume_session_id=getattr(agent, '_resume_session_id', None),
            chat_session=self,
        )
        set_ctx(cli_ctx)
        self._cli_ctx = cli_ctx

        # ── Cross-surface sync via ConversationManager ──
        self._cm_conv_id = None
        try:
            from aura.core.conversation_manager import get_conversation_manager
            _cm = get_conversation_manager()
            if _cm._brain is not None:
                self._cm_conv_id = _cm.get_or_create_session("cli", "local")
                _cm.switch_conversation(self._cm_conv_id, surface="cli")
        except ImportError:
            logger.warning("ConversationManager unavailable — cross-surface sync disabled")
        except Exception:
            logger.debug("conversation_manager_init_failed", exc_info=True)

        # ── Resume session if requested ──
        resume_id = cli_ctx.resume_session_id
        if resume_id:
            if self.agentic.load_session(resume_id):
                self.agentic_session.load(resume_id)
                show_info(f"Session restored ({len(self.agentic._conversation_history)} messages)")
            if hasattr(agent, '_resume_session_id'):
                delattr(agent, '_resume_session_id')

    # ── Permission setup ──────────────────────────────────────────────────

    def apply_model_override(self, model: Optional[str]) -> None:
        """Set the active model across every surface that tracks it.

        Three mirrors of 'current model' used to live in the codebase:
          - session.current_model (display / status bar)
          - agent.brain._model_override (routing)
          - agentic.model_override (per-loop override)
        Updating only one of them silently desynced the others — most
        commonly seen after `/model` when the status bar showed one model
        while the brain routed to another. This is now the single helper
        that updates all three at once; callers (signal handlers, /model
        command, --model flag apply) should use it instead of touching the
        mirrors directly.
        """
        resolved = None if (model is None or model == "auto") else model
        from_model = self.current_model
        try:
            self.agent.brain.set_model_override(resolved)
        except Exception:
            logger.debug("set_model_override_failed", exc_info=True)
        self.agentic.model_override = resolved
        self.current_model = resolved or "auto"

        # Log to JSONL so a future learned-routing classifier has training
        # data. Best-effort, non-blocking — never blocks the UI.
        try:
            from aura.core.event_log import log_model_override
            session_id = getattr(self.agentic_session, "session_id", "") or ""
            log_model_override(
                session_id=session_id,
                from_model=from_model or "",
                to_model=self.current_model or "auto",
                prompt_context=(self.last_user_input or "")[:500],
            )
        except Exception:
            logger.debug("log_model_override_failed", exc_info=True)

    def _cli_confirm(self, tool_name: str, description: str) -> str:
        """Interactive permission prompt — delegates to the shared Rich dialog.

        Returns one of ``'allow_once' | 'allow_session' | 'allow_always' | 'deny'``.
        ``PermissionManager.check`` understands this vocabulary
        (see aura/core/permissions.py:180-192) and records session-scope or
        always-scope approvals accordingly.
        """
        from .permissions_dialog import request_permission
        from .permissions_ui import should_auto_approve_command, should_auto_approve_edit

        if should_auto_approve_edit(self.perm_mode) and tool_name in ("edit_file", "write_file"):
            return "allow_once"
        if should_auto_approve_command(self.perm_mode):
            return "allow_once"
        return request_permission(self.console, tool_name, description)

    # ── Permission banner ─────────────────────────────────────────────────

    def _show_perm_banner(self, mode: str) -> None:
        self._status_controller.show_permission_banner(mode)

    # ── Status bar ────────────────────────────────────────────────────────

    def _show_bar(self, **kwargs: Any) -> None:
        self._status_controller.show_bar(**kwargs)

    # ── Signal handling ───────────────────────────────────────────────────

    def _handle_signal(self, user_input: str) -> bool:
        """Handle keyboard signal pseudo-inputs. Returns True if the main loop should continue."""
        result = self._signal_controller.handle(user_input)
        if result is None:
            return False
        self._injected_input = result.injected_input
        return result.should_continue_loop

    # ── Slash command dispatch ────────────────────────────────────────────

    def _dispatch_command(self, user_input: str) -> None:
        """Route slash commands and update status bar afterward."""
        from .context_bar import estimate_messages_tokens, get_context_limit
        from .display import show_error
        try:
            from .commands import handle_command
            handle_command(self.agent, user_input, speak=self.speak)
        except Exception as exc:
            show_error(f"Command failed: {exc}")
        # current_model is kept authoritative by apply_model_override; only
        # re-sync from brain if something bypassed the helper (legacy guard).
        if not self.current_model or self.current_model == "auto":
            brain_override = getattr(self.agent.brain, "_model_override", None)
            if brain_override:
                self.current_model = brain_override
        self.token_used = estimate_messages_tokens(self.agentic._conversation_history)
        self.token_limit = get_context_limit(self.current_model)
        self._show_bar(
            model=self.current_model, project_type=self._project_type,
            session_title=self.session_title, message_count=self.msg_count,
            token_used=self.token_used, token_limit=self.token_limit,
            permission_mode=self.perm_mode,
        )

    # ── Shared abort handler ────────────────────────────────────────────

    def _handle_ctrl_c_abort(self, streamer: Any) -> bool:
        """Handle double Ctrl+C abort pattern. Returns True if aborted."""
        import os
        import time as _time
        now = _time.time()

        def _do_abort():
            streamer.pause()
            self.steering.clear()
            self.agentic.cancel()
            self.console.print("\n  [red]Aborted.[/red]")
            self._last_ctrl_c_time = 0.0

        if now - self._last_ctrl_c_time < 5.0:
            _do_abort()
            return True

        self._last_ctrl_c_time = now
        self.console.print("\n  [dim yellow]Press Ctrl+C again within 5s to abort[/dim yellow]")

        _deadline = _time.time() + 5.0
        if os.name == "nt":
            # On Windows, time.sleep() does not deliver KeyboardInterrupt
            # reliably while a background thread (esc watchdog) is also
            # reading from the console. Poll msvcrt for the Ctrl+C byte
            # (\x03) directly — short cadence + explicit check beats
            # relying on SIGINT to punch through sleep.
            try:
                import msvcrt  # type: ignore[import]
            except ImportError:
                msvcrt = None  # type: ignore[assignment]
            while _time.time() < _deadline:
                if msvcrt is not None and msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == "\x03":
                        _do_abort()
                        return True
                _time.sleep(0.05)
        else:
            # POSIX: SIGINT reliably interrupts time.sleep. Use a single
            # sleep call so the second Ctrl+C raises KeyboardInterrupt
            # instead of landing between 100ms wake cycles.
            try:
                _time.sleep(max(0.0, _deadline - _time.time()))
            except KeyboardInterrupt:
                _do_abort()
                return True

        self._last_ctrl_c_time = 0.0
        return True  # Single Ctrl+C — cancel this run but leave session alive

    # ── Normal agentic execution ──────────────────────────────────────────

    def _run_agent(self, user_input: str) -> Optional[dict]:
        """Run the agentic loop for a user prompt. Returns result dict or None."""
        return self._execution_controller.run_agent(user_input)

    def _process_normal_result(self, user_input: str, result: Optional[dict]) -> bool:
        """Render and track a successful normal execution result."""
        return self._execution_controller.process_normal_result(user_input, result)

    # ── Plan-approve-execute flow ─────────────────────────────────────────

    def _run_plan_mode(self, user_input: str) -> Optional[dict]:
        """Generate plan, get approval, execute. Returns result dict or None."""
        from .display import show_error, show_info, show_tool_call, show_tool_result_inline
        from .plan_mode import (
            edit_plan_text,
            parse_plan_from_llm,
            show_plan_approval,
        )

        show_info("Generating plan...")
        plan_result = self.agentic.plan_first(user_input)

        if plan_result.get("error"):
            show_error(f"Plan generation failed: {plan_result['error']}")
            return None

        plan = plan_result.get("plan")
        plan_text = plan_result.get("plan_text", "")
        if not plan or not plan.steps:
            show_error("Could not generate a plan. Try rephrasing.")
            return None

        result = None
        while True:
            approval = show_plan_approval(self.console, plan)
            if approval == "y":
                show_info("Executing plan...")
                _prev_trust = self.permissions.trust_mode
                self.permissions.set_trust_mode(True)
                try:
                    from .display import StreamingResponse as _PlanStreamResp
                    streamer = _PlanStreamResp(model=self.current_model)
                    streamer.start()

                    def _plan_on_chunk(text: str, _s=streamer) -> None:
                        _s.chunk(text)

                    def _plan_on_tool_start(name: str, args: dict[str, Any], _s=streamer) -> None:
                        _s.pause()
                        step = getattr(self.agentic, 'iteration', 0)
                        max_iter = getattr(self.agentic, 'max_iterations', 0)
                        desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                        if not desc and "command" in args:
                            desc = args["command"][:60]
                        show_tool_call(name, str(desc), step=step, max_steps=max_iter, status="running")

                    def _plan_on_tool_call(name: str, args: dict[str, Any], _result: Any, _s=streamer) -> None:
                        show_tool_result_inline(name, _result)
                        _s.resume()

                    result = self.agentic.run(
                        user_input,
                        on_tool_call=_plan_on_tool_call,
                        on_tool_start=_plan_on_tool_start,
                        on_chunk=_plan_on_chunk,
                        steering_queue=self.steering,
                    )
                    streamer.finish()
                except KeyboardInterrupt:
                    self._handle_ctrl_c_abort(streamer)
                    result = None
                except Exception as exc:
                    streamer.pause()
                    logger.exception("plan_run_failed")
                    show_error(exc)
                    result = None
                finally:
                    self.permissions.set_trust_mode(_prev_trust)
                break
            elif approval == "e":
                edited_text = edit_plan_text(self.console, plan_text)
                if edited_text != plan_text:
                    plan_text = edited_text
                    plan = parse_plan_from_llm(edited_text)
                continue
            else:
                show_info("Plan cancelled.")
                result = None
                break

        return result

    # ── Channel bridge message processing ─────────────────────────────────

    def _drain_channels(self) -> None:
        self._runtime_controller.drain_channels()

    # ── Background task submission ────────────────────────────────────────

    def _submit_background(self, user_input: str) -> None:
        self._runtime_controller.submit_background(user_input)

    # ── atexit handler ────────────────────────────────────────────────────

    def _save_session_if_initialized(self) -> None:
        self._runtime_controller.save_session_if_initialized()

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the interactive chat session."""
        self._runtime_controller.run()
