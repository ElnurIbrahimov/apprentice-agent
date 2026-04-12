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

logger = logging.getLogger(__name__)

_ERROR_SENTINELS = ["I'm having trouble processing", "[LLM Error]"]

from .chat_session_execution import SessionExecutionController
from .chat_session_runtime import SessionRuntimeController
from .chat_session_signals import SessionSignalController
from .chat_session_status import SessionStatusController, show_startup_diagnostics


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
        from aura.core.agentic_loop import AgenticLoop
        from aura.core.context import gather_context, get_aura_md_config
        from aura.core.permissions import PermissionManager
        from aura.core.session import AgenticSession

        from .checkpoint import CheckpointManager
        from .context import CLIContext, set_ctx
        from .context_bar import get_context_limit
        from .display import (
            console,
            show_banner,
            show_info,
            show_welcome_info,
        )
        from .input import create_session
        from .model_picker import update_model_roles_from_config

        # ── Store references to display helpers for use elsewhere ──
        self.agent = agent
        self.console = console
        self.speak = speak
        self.verbose = verbose
        self.bridge = bridge

        # ── Theme ──
        from .themes import load_theme_preference, set_theme
        saved_theme = load_theme_preference()
        set_theme(saved_theme)

        show_banner()
        show_welcome_info(agent)

        # ── Quick startup diagnostics (non-blocking, <2s) ──
        show_startup_diagnostics(console)

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

        project_root = os.getcwd()
        project_context = ""
        aura_config: dict[str, Any] = {}
        try:
            project_context = gather_context(project_root)
            aura_config = get_aura_md_config(project_root)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("gather_project_context_failed", exc_info=True)

        # ── Permissions ──
        self.permissions = PermissionManager()
        self.perm_mode = "auto_edit"
        if trust:
            self.perm_mode = "full_auto"

        self._build_permissions(trust, aura_config)

        # ── Session ──
        self.agentic_session = AgenticSession()
        self._session_initialized = False

        atexit.register(self._save_session_if_initialized)

        # ── Router & AgenticLoop ──
        from aura.core.router import ModelRouter
        explicit_model = model or agent.brain._model_override or aura_config.get("model") or None
        chat_tier = tier or aura_config.get("tier", "balanced")
        chat_router = ModelRouter(tier=chat_tier, budget_usd=aura_config.get("budget"))
        self.agentic = AgenticLoop(
            brain=agent.brain,
            project_root=project_root,
            permissions=self.permissions,
            model_override=explicit_model,
            max_iterations=aura_config.get("max_iterations", 25),
            budget_usd=aura_config.get("budget"),
            context=project_context,
            session=self.agentic_session,
            aura_config=aura_config,
            router=chat_router,
        )

        # ── Neural routing preference ──
        _PREF_MAP = {"fast": "prefer-fast", "balanced": "balanced", "quality": "prefer-quality"}
        self._routing_preference = _PREF_MAP.get(preference or "balanced", "balanced")
        agent.brain._routing_preference = self._routing_preference

        # ── Checkpoint ──
        try:
            self.checkpoint_mgr: Optional[Any] = CheckpointManager()
        except (OSError, PermissionError):
            self.checkpoint_mgr = None
        if self.checkpoint_mgr:
            self.agentic._checkpoint_mgr = self.checkpoint_mgr
            self.agentic.executor._checkpoint_mgr = self.checkpoint_mgr

        self._show_perm_banner(self.perm_mode)

        # ── Optional subsystems ──
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
            self._create_background_indicator = lambda *a, **k: ""

        try:
            from .research_mode import ResearchContext, create_research_indicator
            self._ResearchContext = ResearchContext
            self._create_research_indicator = create_research_indicator
        except ImportError:
            logger.warning("ResearchContext unavailable — /research mode disabled")
            self._ResearchContext = None
            self._create_research_indicator = lambda *a, **k: ""

        try:
            from .hooks import HookEvent, HookManager
            self._HookManager = HookManager
            self._HookEvent = HookEvent
        except ImportError:
            logger.warning("HookManager unavailable — hooks from AURA.md will not fire")
            console.print("  [dim yellow]Hooks unavailable[/dim yellow]")
            self._HookManager = None
            self._HookEvent = None

        try:
            from .mood_display import create_mood_indicator
            self._create_mood_indicator = create_mood_indicator
        except ImportError:
            logger.warning("mood_display unavailable — mood indicator disabled")
            self._create_mood_indicator = lambda *a, **k: ""

        self.bg_manager = self._BackgroundManager() if self._BackgroundManager else None
        if self.bg_manager and self._notify_completion:
            self.bg_manager.set_completion_callback(self._notify_completion)

        self.research_ctx = self._ResearchContext() if self._ResearchContext else None

        self.hook_mgr = self._HookManager() if self._HookManager else None
        if self.hook_mgr and aura_config:
            self.hook_mgr.load_from_config(aura_config)
            self.hook_mgr.load_builtin_hooks(aura_config)

        if verbose:
            from . import display as _display_mod
            from .disclosure import DisclosureManager
            _display_mod._disclosure = DisclosureManager(default_expanded=True)

        if self.hook_mgr:
            self.hook_mgr.fire(self._HookEvent.SESSION_START, {"project_root": project_root})

        # ── Steering ──
        from .steering import SteeringQueue
        self.steering = SteeringQueue()

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

        # ── Build CLIContext and publish it ──
        cli_ctx = CLIContext(
            agent=agent,
            agentic_loop=self.agentic,
            permissions=self.permissions,
            session=self.agentic_session,
            bg_manager=self.bg_manager,
            research_ctx=self.research_ctx,
            hook_manager=self.hook_mgr,
            speak=speak,
            verbose=verbose,
            resume_session_id=getattr(agent, '_resume_session_id', None),
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

        # ── State variables ──
        self.current_model = explicit_model or "auto"
        self.session_title = ""
        self.msg_count = 0
        self.token_used = 0
        self.token_limit = get_context_limit(self.current_model)

        self._last_ctrl_c_time = 0.0
        self._last_ipc_heartbeat = 0.0

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

        # ── Initial status bar ──
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
                _display_channel_message(console, msg)
            bridge.set_on_message_callback(_channel_notify)

        # ── Store aura_config for plan-approve check ──
        self._aura_config = aura_config
        self._project_root = project_root

        # H3: Auto-test after each edit
        self._auto_test_enabled = bool(aura_config.get("auto_test", False)) if aura_config else False

    # ── Permission setup ──────────────────────────────────────────────────

    def _build_permissions(self, trust: bool, aura_config: dict[str, Any]) -> None:
        """Configure the PermissionManager with CLI confirm callback and overrides."""
        self.permissions.set_confirm_callback(self._cli_confirm)
        self.permissions.set_mode(self.perm_mode)
        if aura_config:
            self.permissions.apply_aura_md_overrides(aura_config)

    def _cli_confirm(self, tool_name: str, description: str) -> bool:
        """Interactive permission prompt — references self.perm_mode."""
        from .permissions_ui import should_auto_approve_command, should_auto_approve_edit

        if should_auto_approve_edit(self.perm_mode) and tool_name in ("edit_file", "write_file"):
            return True
        if should_auto_approve_command(self.perm_mode):
            return True

        try:
            from aura.cli.themes import get_theme
            theme = get_theme()
            warn = theme.warning
            accent = theme.permission_accent
        except (ImportError, AttributeError):
            warn = "#FFC107"
            accent = "#B1B9F9"

        self.console.print()
        self.console.print(f"  [{warn}]\u25b3[/{warn}] [{accent}]Allow {tool_name}?[/{accent}]", end="")
        if description:
            self.console.print(f" [dim]{description[:80]}[/dim]")
        else:
            self.console.print()
        self.console.print(f"    [{accent}]> 1. Yes[/{accent}]")
        self.console.print("      2. Yes, always (trust mode)", style="dim")
        self.console.print("      3. No", style="dim")
        try:
            response = input("  Choose (1-3, Enter=yes): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response in ("2", "always"):
            self.permissions.set_trust_mode(True)
            return True
        if response in ("3", "n", "no"):
            return False
        return response in ("", "1", "y", "yes")

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
        self.current_model = self.agent.brain._model_override or "auto"
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
        import time as _time
        now = _time.time()
        if now - self._last_ctrl_c_time < 5.0:
            streamer.pause()
            self.steering.clear()
            self.agentic.cancel()
            self.console.print("\n  [red]Aborted.[/red]")
            self._last_ctrl_c_time = 0.0
            self.agentic._cancel_event.clear()
            return True
        else:
            self._last_ctrl_c_time = now
            self.console.print("\n  [dim yellow]Press Ctrl+C again within 5s to abort[/dim yellow]")
            try:
                _deadline = _time.time() + 5.0
                while _time.time() < _deadline:
                    _time.sleep(0.1)
            except KeyboardInterrupt:
                streamer.pause()
                self.steering.clear()
                self.agentic.cancel()
                self.console.print("\n  [red]Aborted.[/red]")
                self._last_ctrl_c_time = 0.0
                self.agentic._cancel_event.clear()
                return True
            self._last_ctrl_c_time = 0.0
            return True  # Interrupted but not aborted — still cancel this run

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
                    show_error(str(exc))
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
