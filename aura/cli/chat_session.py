"""ChatSession — extracted from run_chat_mode() god-function.

Encapsulates all CLI chat session state and the main input loop
as a proper class with named methods instead of 900+ lines of
nested closures and local variables.
"""
from __future__ import annotations

import os
import logging
import atexit
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ERROR_SENTINELS = ["I'm having trouble processing", "[LLM Error]"]


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
        from .context import CLIContext, set_ctx
        from .display import (
            console, show_banner, show_response,
            show_error, show_info, show_status_bar, show_help,
            show_welcome_info, show_tool_call,
        )
        from .input import create_session, SIGNAL_MODEL_PICK
        from .model_picker import pick_model, update_model_roles_from_config
        from .context_bar import estimate_messages_tokens, get_context_limit
        from .permissions_ui import cycle_permission_mode, get_mode_description
        from .checkpoint import CheckpointManager
        from aura.core.agentic_loop import AgenticLoop
        from aura.core.session import AgenticSession
        from aura.core.permissions import PermissionManager
        from aura.core.context import gather_context, get_aura_md_config

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
        self._show_startup_diagnostics(console)

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
            from .background import BackgroundManager, notify_completion, create_background_indicator
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
            from .hooks import HookManager, HookEvent
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
            from .disclosure import DisclosureManager
            from . import display as _display_mod
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

        self._mood_cache: dict[str, Any] = {"state": {}, "ts": 0.0}
        self._last_ctrl_c_time = 0.0
        self._last_ipc_heartbeat = 0.0

        import threading
        self._channel_lock = threading.Lock()

        self._pending_follow_up: Optional[str] = None
        self._follow_up_depth = 0
        self._MAX_FOLLOW_UP_DEPTH = 3
        self._streamer_displayed = False
        self.last_user_input = ""

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
        if trust:
            self.permissions.set_trust_mode(True)
        if aura_config:
            self.permissions.apply_aura_md_overrides(aura_config)

    def _cli_confirm(self, tool_name: str, description: str) -> bool:
        """Interactive permission prompt — references self.perm_mode."""
        from .permissions_ui import should_auto_approve_edit, should_auto_approve_command

        if should_auto_approve_edit(self.perm_mode) and tool_name in ("edit_file", "write_file"):
            return True
        if should_auto_approve_command(self.perm_mode):
            return True

        try:
            from aura.cli.themes import get_theme
            theme = get_theme()
            warn = theme.warning
            accent = theme.permission_accent
            muted = theme.text_muted
        except (ImportError, AttributeError):
            warn = "#FFC107"
            accent = "#B1B9F9"
            muted = "#555555"

        self.console.print()
        self.console.print(f"  [{warn}]\u25b3[/{warn}] [{accent}]Allow {tool_name}?[/{accent}]", end="")
        if description:
            self.console.print(f" [dim]{description[:80]}[/dim]")
        else:
            self.console.print()
        self.console.print(f"    [{accent}]> 1. Yes[/{accent}]")
        self.console.print(f"      2. Yes, always (trust mode)", style="dim")
        self.console.print(f"      3. No", style="dim")
        try:
            response = input(f"  Choose (1-3, Enter=yes): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response in ("2", "always"):
            self.permissions.set_trust_mode(True)
            return True
        if response in ("3", "n", "no"):
            return False
        return response in ("", "1", "y", "yes")

    # ── Startup diagnostics ────────────────────────────────────────────────

    @staticmethod
    def _show_startup_diagnostics(console: Any) -> None:
        """Show quick warnings if Ollama or cloud key are missing."""
        import os as _os

        if not _os.environ.get("OLLAMA_API_KEY"):
            console.print(
                "  [yellow]\u26a0 OLLAMA_API_KEY not set \u2014 cloud models unavailable. "
                "Set it in .env[/yellow]"
            )

        # Quick Ollama reachability check (2s timeout)
        try:
            import urllib.request
            host = _os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            req = urllib.request.Request(host, method="HEAD")
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            console.print(
                "  [yellow]\u26a0 Ollama not running \u2014 start with: "
                "ollama serve[/yellow]"
            )

    # ── Permission banner ─────────────────────────────────────────────────

    def _show_perm_banner(self, mode: str) -> None:
        from .permissions_ui import get_mode_indicator
        self.console.print(f"  {get_mode_indicator(mode)}")
        self.console.print()

    # ── Status bar ────────────────────────────────────────────────────────

    def _show_bar(self, **kwargs: Any) -> None:
        from .display import show_status_bar
        bg_ind, res_ind, mood_ind, watch_ind = self._phase3_indicators()
        show_status_bar(
            bg_indicator=bg_ind,
            research_indicator=res_ind,
            mood_indicator=mood_ind,
            watch_indicator=watch_ind,
            steering_queue=self.steering,
            **kwargs,
        )

    def _phase3_indicators(self) -> tuple[str, str, str, str]:
        import time as _t
        background_indicator = self._create_background_indicator(self._cli_ctx.bg_manager) if self._cli_ctx.bg_manager else ""
        research_indicator = self._create_research_indicator(self._cli_ctx.research_ctx) if self._cli_ctx.research_ctx else ""
        mood_indicator = ""
        now = _t.time()
        if now - self._mood_cache["ts"] > 5.0:
            try:
                from aura.emotion.alma_engine import get_alma_engine
                engine = get_alma_engine()
                emotional_state = engine.get_emotional_state() if engine else {}
                self._mood_cache["state"] = emotional_state
                self._mood_cache["ts"] = now
            except Exception:
                logger.debug("mood_cache_update_failed", exc_info=True)
        if self._mood_cache["state"]:
            mood_indicator = self._create_mood_indicator(self._mood_cache["state"])
        watch_indicator = ""
        if self._cli_ctx.file_watcher:
            from .watch_mode import create_watch_indicator
            watch_indicator = create_watch_indicator(self._cli_ctx.file_watcher)
        return background_indicator, research_indicator, mood_indicator, watch_indicator

    # ── Signal handling ───────────────────────────────────────────────────

    def _handle_signal(self, user_input: str) -> bool:
        """Handle keyboard signal pseudo-inputs. Returns True if the main loop should continue."""
        from .input import (
            SIGNAL_CLEAR_SCREEN, SIGNAL_NEW_SESSION,
            SIGNAL_COMMAND_PALETTE, SIGNAL_OPEN_EDITOR,
            SIGNAL_REWIND, SIGNAL_CYCLE_PERMS, SIGNAL_MODEL_PICK,
        )
        from .display import show_info
        from .permissions_ui import cycle_permission_mode, get_mode_description
        from .context_bar import get_context_limit
        from .chat_loop import _rewind_picker

        if user_input == SIGNAL_CLEAR_SCREEN:
            self.console.clear()
            self._show_bar(
                model=self.current_model, project_type=self._project_type,
                session_title=self.session_title, message_count=self.msg_count,
                token_used=self.token_used, token_limit=self.token_limit,
                permission_mode=self.perm_mode,
            )
            return True

        if user_input == SIGNAL_NEW_SESSION:
            if hasattr(self.agentic, 'session') and self.agentic.session:
                self.agentic.session.save()
            self.agentic._conversation_history.clear()
            if self.checkpoint_mgr:
                self.checkpoint_mgr.clear()
            self.msg_count = 0
            self.token_used = 0
            self.console.print("[dim]New session started[/dim]")
            self._show_bar(
                model=self.current_model, project_type=self._project_type,
                session_title=self.session_title, message_count=self.msg_count,
                token_used=self.token_used, token_limit=self.token_limit,
                permission_mode=self.perm_mode,
            )
            return True

        if user_input == SIGNAL_COMMAND_PALETTE:
            from .command_palette import open_palette, build_palette, record_usage
            from .input import SLASH_COMMANDS as _palette_cmds
            items = build_palette(_palette_cmds)
            selected = open_palette(items, self.console)
            if selected:
                record_usage(selected)
                # Re-inject selected command as user_input — caller handles this
                self._injected_input = selected
                return False  # Don't continue — caller must process selected command
            return True

        if user_input == SIGNAL_OPEN_EDITOR:
            import tempfile, subprocess as _sp
            editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
                f.write("")
                tmp_path = f.name
            try:
                _sp.call([editor, tmp_path])
                edited = Path(tmp_path).read_text().strip()
            except (FileNotFoundError, OSError) as e:
                self.console.print(f"[red]Editor failed: {e}[/red]")
                edited = ""
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if not edited:
                return True
            self._injected_input = edited
            return False  # Caller must process the edited text

        if user_input == SIGNAL_CYCLE_PERMS:
            self.perm_mode = cycle_permission_mode(self.perm_mode)
            self.console.print(f"[dim]{get_mode_description(self.perm_mode)}[/dim]")
            self._show_perm_banner(self.perm_mode)
            if self.perm_mode == "full_auto":
                self.permissions.set_trust_mode(True)
            else:
                self.permissions.set_trust_mode(False)
            self._show_bar(
                model=self.current_model, project_type=self._project_type,
                session_title=self.session_title, message_count=self.msg_count,
                token_used=self.token_used, token_limit=self.token_limit,
                permission_mode=self.perm_mode,
            )
            return True

        if user_input == SIGNAL_REWIND:
            if self.checkpoint_mgr:
                _rewind_picker(self.checkpoint_mgr, self.console)
            else:
                self.console.print("[dim]No checkpoint manager available[/dim]")
            return True

        if user_input == SIGNAL_MODEL_PICK:
            from .model_picker import pick_model
            self.current_model = self.agent.brain._model_override or "auto"
            choice = pick_model(self.console, self.current_model)
            if choice:
                if choice == "auto":
                    self.agent.brain.set_model_override(None)
                    self.current_model = "auto"
                    self.agentic.model_override = None
                    show_info("Model set to auto-routing")
                else:
                    self.agent.brain.set_model_override(choice)
                    self.current_model = choice
                    self.agentic.model_override = choice
                    show_info(f"Model set to {choice}")
            self.token_limit = get_context_limit(self.current_model)
            self._show_bar(
                model=self.current_model, project_type=self._project_type,
                session_title=self.session_title, message_count=self.msg_count,
                token_used=self.token_used, token_limit=self.token_limit,
                permission_mode=self.perm_mode,
            )
            return True

        return False  # Not a signal we handle

    # ── Slash command dispatch ────────────────────────────────────────────

    def _dispatch_command(self, user_input: str) -> None:
        """Route slash commands and update status bar afterward."""
        from .display import show_error
        from .context_bar import estimate_messages_tokens, get_context_limit
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
        import time as _exec_time
        from .display import StreamingResponse, show_tool_call, show_tool_result_inline

        streamer = StreamingResponse(model=self.current_model)
        streamer.start()
        _tool_call_count = 0
        _exec_start = _exec_time.monotonic()
        try:
            def _on_chunk(text: str) -> None:
                streamer.chunk(text)

            def _on_tool_start(name: str, args: dict[str, Any]) -> None:
                """Show tool call immediately when it starts (before execution)."""
                nonlocal _tool_call_count
                _tool_call_count += 1
                streamer.pause()

                step = getattr(self.agentic, 'iteration', 0)
                max_iter = getattr(self.agentic, 'max_iterations', 0)

                if self.hook_mgr:
                    self.hook_mgr.fire(self._HookEvent.PRE_TOOL_CALL, {
                        "tool_name": name,
                        "tool_args": str(args)[:500],
                    })

                desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                if not desc and "command" in args:
                    desc = args["command"][:60]

                show_tool_call(name, str(desc), step=step, max_steps=max_iter, status="running")

            def _on_tool_call(name: str, args: dict[str, Any], _result: Any) -> None:
                """Show tool result as soon as execution finishes."""
                show_tool_result_inline(name, _result)

                if self.hook_mgr:
                    self.hook_mgr.fire(self._HookEvent.POST_TOOL_CALL, {
                        "tool_name": name,
                        "tool_args": str(args)[:500],
                    })
                if self.hook_mgr and name in ("edit_file", "write_file"):
                    self.hook_mgr.fire(self._HookEvent.POST_EDIT, {
                        "tool_name": name,
                        "file_path": args.get("path", args.get("file_path", "")),
                    })

                # H3: Auto-test after each edit
                if name in ("edit_file", "write_file") and getattr(self, '_auto_test_enabled', False):
                    try:
                        test_result = self.agentic._run_auto_test()
                        if test_result:
                            self.agentic._conversation_history.append({
                                "role": "user",
                                "content": f"[Auto-test failed after editing] {test_result}",
                            })
                    except Exception:
                        pass

                streamer.resume()

            result = self.agentic.run(
                user_input,
                on_tool_call=_on_tool_call,
                on_tool_start=_on_tool_start,
                on_chunk=_on_chunk,
                steering_queue=self.steering,
            )
        except KeyboardInterrupt:
            self._handle_ctrl_c_abort(streamer)
            return None
        except Exception as exc:
            streamer.pause()
            from .display import show_error
            show_error(str(exc))
            return None

        streamer.finish()
        self._streamer_displayed = True

        # Execution summary
        _elapsed = _exec_time.monotonic() - _exec_start
        if _tool_call_count > 0 or _elapsed > 2.0:
            from .display import show_response_attribution
            _iter_count = getattr(self.agentic, 'iteration', 0)
            _summary_parts = []
            if _iter_count > 1:
                _summary_parts.append(f"{_iter_count} steps")
            if _tool_call_count > 0:
                _summary_parts.append(f"{_tool_call_count} tool calls")
            show_response_attribution(
                model=self.current_model,
                elapsed=_elapsed,
                tokens=result.get("tokens", 0) if result else 0,
            )
            if _summary_parts:
                # Build changes summary line: files + iterations + tool calls
                try:
                    edited_files = [f for f in getattr(self.agentic, '_hot_files', []) if f]
                    if edited_files:
                        import os as _os
                        files_display = ", ".join(
                            _os.path.basename(f) for f in edited_files[:8]
                        )
                        extra = f" (+{len(edited_files) - 8} more)" if len(edited_files) > 8 else ""
                        parts_str = " \u00b7 ".join(_summary_parts)
                        self.console.print(
                            f"  [dim]Files touched: {files_display}{extra} | {parts_str}[/dim]"
                        )
                    else:
                        self.console.print(f"  [dim]{' \u00b7 '.join(_summary_parts)}[/dim]")
                except Exception:
                    self.console.print(f"  [dim]{' \u00b7 '.join(_summary_parts)}[/dim]")

        return result

    # ── Plan-approve-execute flow ─────────────────────────────────────────

    def _run_plan_mode(self, user_input: str) -> Optional[dict]:
        """Generate plan, get approval, execute. Returns result dict or None."""
        from .display import show_info, show_error, show_tool_call, show_tool_result_inline
        from .plan_mode import (
            render_plan, show_plan_approval, edit_plan_text,
            parse_plan_from_llm,
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

                    def _plan_on_chunk(text: str) -> None:
                        streamer.chunk(text)

                    def _plan_on_tool_start(name: str, args: dict[str, Any]) -> None:
                        streamer.pause()
                        step = getattr(self.agentic, 'iteration', 0)
                        max_iter = getattr(self.agentic, 'max_iterations', 0)
                        desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                        if not desc and "command" in args:
                            desc = args["command"][:60]
                        show_tool_call(name, str(desc), step=step, max_steps=max_iter, status="running")

                    def _plan_on_tool_call(name: str, args: dict[str, Any], _result: Any) -> None:
                        show_tool_result_inline(name, _result)
                        streamer.resume()

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
        """Process pending channel messages in a background thread (non-blocking).

        Processes at most 1 message per call to avoid stalling the main loop.
        The agent run happens in a daemon thread so the CLI stays responsive.
        """
        if not self.bridge or not self.bridge.has_pending():
            return
        # Skip if a previous channel message is still being processed
        if not self._channel_lock.acquire(blocking=False):
            return

        ch_msg = self.bridge.get_pending_message(timeout=0)
        if ch_msg is None:
            self._channel_lock.release()
            return

        def _process():
            try:
                from .chat_loop import _display_channel_response
                result = self.agentic.run(ch_msg.text)
                response_text = result.get("response", "") if result else ""
            except Exception as _e:
                logger.debug("channel_agent_run_failed", exc_info=True)
                response_text = f"Error processing message: {_e}"
            try:
                if response_text:
                    _display_channel_response(self.console, ch_msg, response_text)
                    self.bridge.send_response(ch_msg, response_text)
            except Exception:
                logger.debug("channel_response_display_failed", exc_info=True)
            finally:
                self._channel_lock.release()

        import threading
        threading.Thread(target=_process, daemon=True, name="channel-drain").start()

    # ── Background task submission ────────────────────────────────────────

    def _submit_background(self, user_input: str) -> None:
        """Handle the '& ' prefix for background task submission."""
        bg_prompt = user_input[2:].strip() if user_input.startswith("& ") else user_input[1:].strip()
        if not bg_prompt:
            self.console.print("[dim]Usage: & <prompt>[/dim]")
            return

        def _bg_task_fn(prompt: str) -> dict[str, Any]:
            try:
                response = self.agent.brain.think(prompt)
                if isinstance(response, dict):
                    response = response.get("response", response.get("content", str(response)))
                return {"success": True, "response": response or "", "iterations": 1}
            except Exception as e:
                return {"success": False, "error": str(e)}

        if not self.bg_manager:
            self.console.print("[red]Background tasks are not available.[/red]")
            return
        task = self.bg_manager.submit(bg_prompt, _bg_task_fn)
        if task:
            self.console.print(f"[cyan]Background task started: {task.id}[/cyan]")
        else:
            self.console.print("[red]Too many background tasks running.[/red]")

    # ── atexit handler ────────────────────────────────────────────────────

    def _save_session_if_initialized(self) -> None:
        if self._session_initialized:
            self.agentic_session.save()

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """The clean main loop — replaces the body of run_chat_mode()."""
        from .input import (
            get_input,
            SIGNAL_CLEAR_SCREEN, SIGNAL_NEW_SESSION,
            SIGNAL_COMMAND_PALETTE, SIGNAL_OPEN_EDITOR,
            SIGNAL_REWIND, SIGNAL_CYCLE_PERMS, SIGNAL_MODEL_PICK,
        )
        from .display import (
            show_response, show_error, show_info, show_help,
        )
        from .permissions_ui import is_plan_approve_mode
        from .context_bar import estimate_messages_tokens, get_context_limit

        _ALL_SIGNALS = {
            SIGNAL_CLEAR_SCREEN, SIGNAL_NEW_SESSION,
            SIGNAL_COMMAND_PALETTE, SIGNAL_OPEN_EDITOR,
            SIGNAL_REWIND, SIGNAL_CYCLE_PERMS, SIGNAL_MODEL_PICK,
        }

        while True:
            # Drain any pending channel messages before waiting for CLI input
            self._drain_channels()

            if self._pending_follow_up:
                user_input = self._pending_follow_up
                self._pending_follow_up = None
                show_info(f"Follow-up: {user_input[:60]}...")
            else:
                self._follow_up_depth = 0
                user_input = get_input(self._pt_session)

            # After CLI input, drain channel messages that arrived while typing
            self._drain_channels()

            # ── Exit ──
            if user_input is None:
                if self.bridge:
                    self.bridge.stop()
                if self.hook_mgr:
                    self.hook_mgr.fire(self._HookEvent.SESSION_END, {"reason": "user_exit"})
                self.console.print("\n[dim]Goodbye.[/dim]\n")
                break

            # ── Signal handling ──
            if user_input in _ALL_SIGNALS:
                self._injected_input = None
                should_continue = self._handle_signal(user_input)
                if should_continue:
                    continue
                # COMMAND_PALETTE and OPEN_EDITOR may inject new input
                if self._injected_input is not None:
                    user_input = self._injected_input
                    self._injected_input = None
                else:
                    continue

            # ── Empty input ──
            if not user_input:
                continue

            # ── Background tasks (& prefix) ──
            if user_input.startswith("& ") or (user_input.startswith("&") and len(user_input) > 1 and user_input[1] != " "):
                self._submit_background(user_input)
                continue

            # ── IPC heartbeat ──
            import time as _t_ipc
            if _t_ipc.time() - self._last_ipc_heartbeat > 30.0:
                self._last_ipc_heartbeat = _t_ipc.time()
                try:
                    import socket, json as _json
                    _ipc_token = ""
                    _token_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "ipc_token")
                    if os.path.isfile(_token_path):
                        with open(_token_path) as _tf:
                            _ipc_token = _tf.read().strip()
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.1)
                        s.connect(("127.0.0.1", 19733))
                        s.send((_json.dumps({"type": "activity", "token": _ipc_token}) + "\n").encode())
                except (OSError, ValueError):
                    pass

            # ── Help shortcut ──
            if user_input.strip() == "?":
                show_help()
                continue

            # ── /retry ──
            if user_input.strip() == "/retry":
                if self.last_user_input:
                    show_info(f"Retrying: {self.last_user_input[:60]}...")
                    self._pending_follow_up = self.last_user_input
                    continue
                else:
                    show_error("Nothing to retry — no previous prompt.")
                    continue

            # ── /channels ──
            if user_input.strip() == "/channels":
                if not self.bridge:
                    show_info("No channel bridge active. Start with --channels flag.")
                else:
                    from rich.table import Table
                    ch_table = Table(
                        show_header=True, header_style="bold cyan",
                        border_style="dim", padding=(0, 2),
                        title="[bold]Active Channels[/bold]",
                    )
                    ch_table.add_column("Channel", style="cyan", width=16)
                    ch_table.add_column("Status", style="white", width=12)
                    ch_table.add_column("Pending", style="dim", width=10)
                    for st in self.bridge.status():
                        status_str = "[green]running[/green]" if st["running"] else "[red]stopped[/red]"
                        ch_table.add_row(st["channel"], status_str, str(st["pending"]))
                    self.console.print()
                    self.console.print(ch_table)
                    self.console.print()
                continue

            # ── Slash commands ──
            if user_input.startswith("/"):
                self._dispatch_command(user_input)
                continue

            # ── Initialize session on first real prompt ──
            if not self._session_initialized:
                self.agentic_session.new(project_root=self._project_root, model=self.agent.brain._model_override or "auto")
                self._session_initialized = True

            self.last_user_input = user_input

            # ── Plan-Approve-Execute mode ──
            if is_plan_approve_mode(self.perm_mode):
                result = self._run_plan_mode(user_input)
                if result is None:
                    continue
                # Plan mode: handle result with streamer info
                response_text = result.get("response", "")
                model_used = result.get("model", self.current_model)
                is_error = result.get("success") is False or any(response_text.startswith(s) for s in _ERROR_SENTINELS)
                if is_error:
                    show_error(response_text)
                else:
                    # In plan mode, streamer handles display; use show_response as fallback
                    if response_text:
                        show_response(response_text, model=model_used, stream=False)

                # Track in ConversationManager (plan-approve path)
                if self._cm_conv_id:
                    try:
                        from aura.core.conversation_manager import get_conversation_manager
                        _cm = get_conversation_manager()
                        _cm.on_message_added(self._cm_conv_id, "user", user_input, "cli", "local")
                        _cm.on_message_added(self._cm_conv_id, "assistant", response_text, "cli", "local")
                    except Exception:
                        pass

                self.msg_count += 1
                if self.msg_count == 1 and user_input:
                    self.session_title = user_input[:50].strip()
                self.current_model = self.agent.brain._model_override or "auto"
                self.token_used = estimate_messages_tokens(self.agentic._conversation_history)
                self.token_limit = get_context_limit(self.current_model)
                self._show_bar(
                    model=self.current_model, project_type=self._project_type,
                    session_title=self.session_title, message_count=self.msg_count,
                    token_used=self.token_used, token_limit=self.token_limit,
                    permission_mode=self.perm_mode,
                )
                continue

            # ── Normal execution path ──
            self._streamer_displayed = False
            result = self._run_agent(user_input)

            if result is None:
                show_error("No response received.")
                continue

            response_text = result.get("response", "")
            model_used = result.get("model", self.current_model)

            is_error = result.get("success") is False or any(response_text.startswith(s) for s in _ERROR_SENTINELS)
            if is_error:
                show_error(response_text)
                continue

            # Context summary
            _ctx_memory_count = 0
            _ctx_mood = ""
            _ctx_tool_count = 0
            try:
                if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'memories'):
                    _ctx_memory_count = len(self.agent.memory.memories)
                elif hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'count'):
                    _ctx_memory_count = self.agent.memory.count()
            except (TypeError, AttributeError):
                logger.debug("ctx_memory_count_failed", exc_info=True)
            try:
                if hasattr(self.agent, 'mood') and self.agent.mood:
                    _ctx_mood = str(self.agent.mood.get("mood", "")) if isinstance(self.agent.mood, dict) else str(self.agent.mood)
            except (TypeError, AttributeError):
                logger.debug("ctx_mood_read_failed", exc_info=True)
            try:
                _ctx_tool_count = result.get("tool_calls", 0)
            except (TypeError, AttributeError):
                logger.debug("ctx_tool_count_failed", exc_info=True)
            from .display import show_context_summary
            show_context_summary(
                memory_count=_ctx_memory_count,
                mood=_ctx_mood,
                model=model_used,
                tool_count=_ctx_tool_count,
            )

            # _run_agent already calls streamer.finish() which displays the response.
            # Only show here for plan-mode results which don't use a streamer.
            if response_text and not self._streamer_displayed:
                show_response(response_text, model=model_used, stream=False)

            # Activity log
            if self.activity_log:
                try:
                    self.activity_log.log(
                        prompt=user_input,
                        response=response_text[:20000] if response_text else "",
                        model=result.get("model", ""),
                        session_id=getattr(self.agentic_session, 'session_id', ''),
                        tool_calls=result.get("tool_calls", 0),
                    )
                except (OSError, TypeError, ValueError):
                    logger.debug("activity_log_write_failed", exc_info=True)

            # ConversationManager tracking (normal path)
            if self._cm_conv_id:
                try:
                    from aura.core.conversation_manager import get_conversation_manager
                    _cm = get_conversation_manager()
                    _cm.on_message_added(self._cm_conv_id, "user", user_input, "cli", "local")
                    _cm.on_message_added(self._cm_conv_id, "assistant", response_text, "cli", "local")
                except Exception:
                    pass

            # Follow-up from steering
            follow_up = self.steering.pop_follow_up()
            if follow_up and self._follow_up_depth < self._MAX_FOLLOW_UP_DEPTH:
                self._pending_follow_up = follow_up
                self._follow_up_depth += 1
            elif follow_up:
                show_info("Max auto-follow-up depth reached, dropping follow-up.")

            # Update counters and status bar
            self.msg_count += 1
            if self.msg_count == 1 and user_input:
                self.session_title = user_input[:50].strip()
            self.current_model = self.agent.brain._model_override or "auto"
            self.token_used = estimate_messages_tokens(self.agentic._conversation_history)
            self.token_limit = get_context_limit(self.current_model)
            cost_usd = 0.0
            try:
                stats = self.agent.brain.get_session_stats()
                cost_usd = stats.get("cost_usd", 0.0)
            except (AttributeError, TypeError, KeyError):
                logger.debug("session_stats_read_failed", exc_info=True)
            self._show_bar(
                model=self.current_model, project_type=self._project_type,
                session_title=self.session_title, message_count=self.msg_count,
                cost_usd=cost_usd,
                token_used=self.token_used, token_limit=self.token_limit,
                permission_mode=self.perm_mode,
            )

            # Hooks
            if self.hook_mgr:
                self.hook_mgr.fire(self._HookEvent.POST_RESPONSE, {
                    "response": response_text[:500] if response_text else "",
                    "model": model_used,
                })

            # TTS
            if self.speak and response_text:
                try:
                    self.agent._speak(response_text)
                except (OSError, RuntimeError, AttributeError):
                    logger.warning("tts_speak_failed", exc_info=True)
