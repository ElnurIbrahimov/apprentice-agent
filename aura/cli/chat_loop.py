from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _rewind_picker(cp_mgr: Any, console: Any) -> bool:
    import time as _time
    cps = cp_mgr.list_checkpoints()
    if not cps:
        console.print("[dim]No checkpoints available[/dim]")
        return False
    console.print("\n[bold]Rewind to checkpoint:[/bold]")
    for i, cp in enumerate(cps[:10]):
        ts = _time.strftime("%H:%M:%S", _time.localtime(cp["timestamp"]))
        files = ", ".join(
            f["original_path"].split("/")[-1].split("\\")[-1]
            for f in cp["files"]
        )
        console.print(f"  {i+1}. [{ts}] {cp['label']} ({files})")
    console.print("  0. Cancel")
    try:
        choice = input("\nSelect checkpoint: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if choice.isdigit() and 0 < int(choice) <= min(10, len(cps)):
        selected = cps[int(choice) - 1]
        if cp_mgr.restore(selected["id"]):
            files = ", ".join(
                f["original_path"].split("/")[-1].split("\\")[-1]
                for f in selected["files"]
            )
            console.print(f"[green]Restored: {files}[/green]")
            return True
        else:
            console.print("[red]Restore failed.[/red]")
            return False
    return False


def _display_channel_message(console: Any, msg: Any) -> None:
    """Display an incoming channel message with a styled box."""
    from rich.text import Text
    source_name = msg.source.value.capitalize()
    width = min(console.size.width - 4, 60)
    header = f" {source_name} "
    line_rest = "\u2500" * max(width - len(header) - 2, 4)
    top = f"\u250c\u2500{header}{line_rest}"

    body = Text()
    body.append(f"\u2502 ", style="dim cyan")
    body.append(f"{msg.user_name}: ", style="bold")
    body.append(msg.text[:500])

    bottom = "\u2514" + "\u2500" * (width)

    console.print()
    console.print(f"  [cyan]{top}[/cyan]")
    console.print(f"  ", end="")
    console.print(body)
    console.print(f"  [cyan]{bottom}[/cyan]")


def _display_channel_response(console: Any, msg: Any, response_text: str) -> None:
    """Display an outgoing response to a channel with a styled box."""
    from rich.text import Text
    source_name = msg.source.value.capitalize()
    width = min(console.size.width - 4, 60)
    header = f" \u2192 {source_name} "
    line_rest = "\u2500" * max(width - len(header) - 2, 4)
    top = f"\u250c\u2500{header}{line_rest}"

    # Truncate long responses for display
    display_text = response_text[:300]
    if len(response_text) > 300:
        display_text += "..."

    body = Text()
    body.append(f"\u2502 ", style="dim green")
    body.append("AURA: ", style="bold green")
    body.append(display_text)

    bottom = "\u2514" + "\u2500" * (width)

    console.print(f"  [green]{top}[/green]")
    console.print(f"  ", end="")
    console.print(body)
    console.print(f"  [green]{bottom}[/green]")


def run_chat_mode(agent: Any, speak: bool = False, trust: bool = False, model: Optional[str] = None, verbose: bool = False, tier: Optional[str] = None, bridge: Any = None) -> None:
    from .context import CLIContext, set_ctx
    from .display import (
        console, show_banner, show_response,
        show_error, show_info, show_status_bar, show_help,
        show_welcome_info, show_tool_call,
    )
    from .input import (
        create_session, get_input,
        SIGNAL_MODEL_PICK, SIGNAL_CLEAR_SCREEN, SIGNAL_NEW_SESSION,
        SIGNAL_COMMAND_PALETTE, SIGNAL_OPEN_EDITOR, SIGNAL_REWIND, SIGNAL_CYCLE_PERMS,
    )
    from .model_picker import pick_model, update_model_roles_from_config
    from .context_bar import estimate_messages_tokens, get_context_limit
    from .permissions_ui import cycle_permission_mode, get_mode_description
    from .checkpoint import CheckpointManager
    from aura.core.agentic_loop import AgenticLoop
    from aura.core.session import AgenticSession
    from aura.core.permissions import PermissionManager
    from aura.core.context import gather_context, get_aura_md_config

    from .themes import load_theme_preference, set_theme
    saved_theme = load_theme_preference()
    set_theme(saved_theme)

    show_banner()
    show_welcome_info(agent)

    # Show permission mode banner (like Claude Code's "Allowed tools: ...")
    def _show_perm_banner(mode: str) -> None:
        from .permissions_ui import get_mode_description
        allowed = "Read, Glob, Grep, ListDir, Git"
        if mode in ("auto_edit", "full_auto"):
            allowed += ", Edit, Write"
        if mode == "full_auto":
            allowed += ", Shell(all)"
        else:
            allowed += ", Shell(safe)"
        console.print(f"  [dim]Allowed tools:[/dim] [bold]{allowed}[/bold]  [dim]|[/dim]  [dim]Mode:[/dim] [bold]{get_mode_description(mode).split(' — ')[0]}[/bold]")
        console.print()

    _project_type = ""
    try:
        from aura.tools.project_context import detect_and_load_context
        ctx = detect_and_load_context(".")
        _project_type = ctx.get("project_type", "") if isinstance(ctx, dict) else ""
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        logger.debug("project_type_detection_failed", exc_info=True)

    project_root = os.getcwd()
    project_context = ""
    aura_config = {}
    try:
        project_context = gather_context(project_root)
        aura_config = get_aura_md_config(project_root)
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("gather_project_context_failed", exc_info=True)

    permissions = PermissionManager()

    def _cli_confirm(tool_name: str, description: str) -> bool:
        from rich.prompt import Prompt
        from .permissions_ui import should_auto_approve_edit, should_auto_approve_command

        # Auto-edit mode: auto-approve file edits/writes, still prompt for shell
        if should_auto_approve_edit(current_perm_mode) and tool_name in ("edit_file", "write_file"):
            return True
        if should_auto_approve_command(current_perm_mode):
            return True

        console.print(f"\n  [yellow]Permission required:[/yellow] {tool_name}")
        if description:
            for line in description.split("\n")[:10]:
                console.print(f"    {line}", highlight=False)
        try:
            response = Prompt.ask(
                "    [yellow]Allow?[/yellow]",
                choices=["y", "n", "always"],
                default="y",
                console=console,
            )
        except (EOFError, KeyboardInterrupt):
            return False
        if response == "always":
            permissions.set_trust_mode(True)
            return True
        return response in ("y", "yes")

    permissions.set_confirm_callback(_cli_confirm)
    if trust:
        permissions.set_trust_mode(True)

    if aura_config:
        permissions.apply_aura_md_overrides(aura_config)

    agentic_session = AgenticSession()
    _session_initialized = False
    import atexit
    def _save_session_if_initialized() -> None:
        if _session_initialized:
            agentic_session.save()
    atexit.register(_save_session_if_initialized)

    from aura.core.router import ModelRouter
    explicit_model = model or agent.brain._model_override or aura_config.get("model") or None
    chat_tier = tier or aura_config.get("tier", "balanced")
    chat_router = ModelRouter(tier=chat_tier, budget_usd=aura_config.get("budget"))
    agentic = AgenticLoop(
        brain=agent.brain,
        project_root=project_root,
        permissions=permissions,
        model_override=explicit_model,
        max_iterations=aura_config.get("max_iterations", 25),
        budget_usd=aura_config.get("budget"),
        context=project_context,
        session=agentic_session,
        aura_config=aura_config,
        router=chat_router,
    )
    try:
        checkpoint_mgr = CheckpointManager()
    except (OSError, PermissionError):
        checkpoint_mgr = None
    if checkpoint_mgr:
        agentic._checkpoint_mgr = checkpoint_mgr
        agentic.executor._checkpoint_mgr = checkpoint_mgr

    current_perm_mode = "auto_edit"
    if trust:
        current_perm_mode = "full_auto"

    _show_perm_banner(current_perm_mode)

    try:
        from .background import BackgroundManager, notify_completion, create_background_indicator
    except ImportError:
        BackgroundManager = None
        notify_completion = None
        create_background_indicator = lambda *a, **k: ""

    try:
        from .research_mode import ResearchContext, create_research_indicator
    except ImportError:
        ResearchContext = None
        create_research_indicator = lambda *a, **k: ""

    try:
        from .hooks import HookManager, HookEvent
    except ImportError:
        HookManager = None
        HookEvent = None

    try:
        from .mood_display import create_mood_indicator
    except ImportError:
        create_mood_indicator = lambda *a, **k: ""

    bg_manager = BackgroundManager() if BackgroundManager else None
    if bg_manager and notify_completion:
        bg_manager.set_completion_callback(notify_completion)

    research_ctx = ResearchContext() if ResearchContext else None

    hook_mgr = HookManager() if HookManager else None
    if hook_mgr and aura_config:
        hook_mgr.load_from_config(aura_config)
        hook_mgr.load_builtin_hooks(aura_config)

    if verbose:
        from .disclosure import DisclosureManager
        from . import display as _display_mod
        _display_mod._disclosure = DisclosureManager(default_expanded=True)

    if hook_mgr:
        hook_mgr.fire(HookEvent.SESSION_START, {"project_root": project_root})

    from .steering import SteeringQueue
    steering = SteeringQueue()

    try:
        from .activity_log import ActivityLog
        activity_log = ActivityLog()
    except (ImportError, OSError):
        logger.debug("activity_log_init_failed", exc_info=True)
        activity_log = None

    # ── Build CLIContext and publish it ──────────────────────────────────
    cli_ctx = CLIContext(
        agent=agent,
        agentic_loop=agentic,
        permissions=permissions,
        session=agentic_session,
        bg_manager=bg_manager,
        research_ctx=research_ctx,
        hook_manager=hook_mgr,
        speak=speak,
        verbose=verbose,
        resume_session_id=getattr(agent, '_resume_session_id', None),
    )
    set_ctx(cli_ctx)

    # Legacy aliases so handlers that still use agent._ keep working
    # during incremental migration.  Each one will be removed as the
    # handler is ported to use get_ctx().
    agent._agentic_loop = agentic
    agent._agentic_permissions = permissions
    agent._agentic_session = agentic_session
    agent._bg_manager = bg_manager
    agent._research_ctx = research_ctx
    agent._hook_manager = hook_mgr

    # ── Cross-surface sync via ConversationManager ─────────────────
    _cm_conv_id = None
    try:
        from aura.core.conversation_manager import get_conversation_manager
        _cm = get_conversation_manager()
        if _cm._brain is not None:
            _cm_conv_id = _cm.get_or_create_session("cli", "local")
            _cm.switch_conversation(_cm_conv_id, surface="cli")
    except Exception:
        pass

    resume_id = cli_ctx.resume_session_id
    if resume_id:
        if agentic.load_session(resume_id):
            agentic_session.load(resume_id)
            show_info(f"Session restored ({len(agentic._conversation_history)} messages)")
        if hasattr(agent, '_resume_session_id'):
            delattr(agent, '_resume_session_id')

    _current_model = explicit_model or "auto"
    _session_title = ""
    _msg_count = 0
    _token_used = 0
    _token_limit = get_context_limit(_current_model)

    _mood_cache = {"state": {}, "ts": 0.0}

    def _phase3_indicators() -> tuple[str, str, str, str]:
        import time as _t
        background_indicator = create_background_indicator(cli_ctx.bg_manager) if cli_ctx.bg_manager else ""
        research_indicator = create_research_indicator(cli_ctx.research_ctx) if cli_ctx.research_ctx else ""
        mood_indicator = ""
        now = _t.time()
        if now - _mood_cache["ts"] > 5.0:
            try:
                from aura.emotion.alma_engine import get_alma_engine
                engine = get_alma_engine()
                emotional_state = engine.get_emotional_state() if engine else {}
                _mood_cache["state"] = emotional_state
                _mood_cache["ts"] = now
            except Exception:  # Catch-all: mood engine is cosmetic, must not crash status bar
                logger.debug("mood_cache_update_failed", exc_info=True)
        if _mood_cache["state"]:
            mood_indicator = create_mood_indicator(_mood_cache["state"])
        watch_indicator = ""
        if cli_ctx.file_watcher:
            from .watch_mode import create_watch_indicator
            watch_indicator = create_watch_indicator(cli_ctx.file_watcher)
        return background_indicator, research_indicator, mood_indicator, watch_indicator

    def _show_bar(**kwargs: Any) -> None:
        bg_ind, res_ind, mood_ind, watch_ind = _phase3_indicators()
        show_status_bar(
            bg_indicator=bg_ind,
            research_indicator=res_ind,
            mood_indicator=mood_ind,
            watch_indicator=watch_ind,
            steering_queue=steering,
            **kwargs,
        )

    _show_bar(
        model=_current_model, project_type=_project_type,
        session_title=_session_title, message_count=_msg_count,
        token_used=_token_used, token_limit=_token_limit,
        permission_mode=current_perm_mode,
    )

    if speak:
        show_info("Voice output enabled")

    update_model_roles_from_config()

    session = create_session()

    # -- Channel bridge setup --
    if bridge:
        show_info(f"Channel bridge active: {', '.join(s['channel'] for s in bridge.status())}")
        # Register a live-notification callback that prints above the prompt
        def _channel_notify(msg: Any) -> None:
            # This fires from the adapter thread; Rich console.print is thread-safe
            _display_channel_message(console, msg)
        bridge.set_on_message_callback(_channel_notify)

    def _drain_channel_messages() -> None:
        """Process all pending channel messages through the agent."""
        if not bridge:
            return
        while bridge.has_pending():
            ch_msg = bridge.get_pending_message(timeout=0)
            if ch_msg is None:
                break
            # Process through the agent
            try:
                result = agentic.run(ch_msg.text)
                response_text = result.get("response", "") if result else ""
            except Exception as _e:
                logger.debug("channel_agent_run_failed", exc_info=True)
                response_text = f"Error processing message: {_e}"

            if response_text:
                _display_channel_response(console, ch_msg, response_text)
                bridge.send_response(ch_msg, response_text)

    _pending_follow_up = None
    _follow_up_depth = 0
    _MAX_FOLLOW_UP_DEPTH = 3
    _last_user_input = ""

    while True:
        # Drain any pending channel messages before waiting for CLI input
        _drain_channel_messages()

        if _pending_follow_up:
            user_input = _pending_follow_up
            _pending_follow_up = None
            show_info(f"Follow-up: {user_input[:60]}...")
        else:
            _follow_up_depth = 0
            user_input = get_input(session)

        # After CLI input, drain channel messages that arrived while typing
        _drain_channel_messages()

        if user_input is None:
            if bridge:
                bridge.stop()
            if hook_mgr:
                hook_mgr.fire(HookEvent.SESSION_END, {"reason": "user_exit"})
            console.print("\n[dim]Goodbye.[/dim]\n")
            break

        if user_input == SIGNAL_CLEAR_SCREEN:
            console.clear()
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue
        elif user_input == SIGNAL_NEW_SESSION:
            if hasattr(agentic, 'session') and agentic.session:
                agentic.session.save()
            agentic._conversation_history.clear()
            if checkpoint_mgr:
                checkpoint_mgr.clear()
            _msg_count = 0
            _token_used = 0
            console.print("[dim]New session started[/dim]")
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue
        elif user_input == SIGNAL_COMMAND_PALETTE:
            from .command_palette import open_palette, build_palette, record_usage
            from .input import SLASH_COMMANDS as _palette_cmds
            items = build_palette(_palette_cmds)
            selected = open_palette(items, console)
            if selected:
                record_usage(selected)
                user_input = selected
            else:
                continue
        elif user_input == SIGNAL_OPEN_EDITOR:
            import tempfile, subprocess as _sp
            editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
                f.write("")
                tmp_path = f.name
            try:
                _sp.call([editor, tmp_path])
                user_input = Path(tmp_path).read_text().strip()
            except (FileNotFoundError, OSError) as e:
                console.print(f"[red]Editor failed: {e}[/red]")
                user_input = ""
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if not user_input:
                continue
        elif user_input == SIGNAL_CYCLE_PERMS:
            current_perm_mode = cycle_permission_mode(current_perm_mode)
            console.print(f"[dim]{get_mode_description(current_perm_mode)}[/dim]")
            _show_perm_banner(current_perm_mode)
            if current_perm_mode == "full_auto":
                permissions.set_trust_mode(True)
            else:
                permissions.set_trust_mode(False)
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue
        elif user_input == SIGNAL_REWIND:
            if checkpoint_mgr:
                _rewind_picker(checkpoint_mgr, console)
            else:
                console.print("[dim]No checkpoint manager available[/dim]")
            continue

        if user_input == SIGNAL_MODEL_PICK:
            _current_model = agent.brain._model_override or "auto"
            choice = pick_model(console, _current_model)
            if choice:
                if choice == "auto":
                    agent.brain.set_model_override(None)
                    _current_model = "auto"
                    agentic.model_override = None
                    show_info("Model set to auto-routing")
                else:
                    agent.brain.set_model_override(choice)
                    _current_model = choice
                    agentic.model_override = choice
                    show_info(f"Model set to {choice}")
            _token_limit = get_context_limit(_current_model)
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue

        if not user_input:
            continue

        if user_input.startswith("& ") or (user_input.startswith("&") and len(user_input) > 1 and user_input[1] != " "):
            bg_prompt = user_input[2:].strip() if user_input.startswith("& ") else user_input[1:].strip()
            if not bg_prompt:
                console.print("[dim]Usage: & <prompt>[/dim]")
                continue
            def _bg_task_fn(prompt: str) -> dict[str, Any]:
                try:
                    response = agent.brain.think(prompt)
                    if isinstance(response, dict):
                        response = response.get("response", response.get("content", str(response)))
                    return {"success": True, "response": response or "", "iterations": 1}
                except Exception as e:  # Catch-all: runs in thread pool, must not propagate
                    return {"success": False, "error": str(e)}
            task = bg_manager.submit(bg_prompt, _bg_task_fn)
            if task:
                console.print(f"[cyan]Background task started: {task.id}[/cyan]")
            else:
                console.print("[red]Too many background tasks running.[/red]")
            continue

        # IPC heartbeat — best-effort, failures are expected and harmless
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

        if user_input.strip() == "?":
            show_help()
            continue

        if user_input.strip() == "/retry":
            if _last_user_input:
                show_info(f"Retrying: {_last_user_input[:60]}...")
                _pending_follow_up = _last_user_input
                continue
            else:
                show_error("Nothing to retry — no previous prompt.")
                continue

        if user_input.strip() == "/channels":
            if not bridge:
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
                for st in bridge.status():
                    status_str = "[green]running[/green]" if st["running"] else "[red]stopped[/red]"
                    ch_table.add_row(st["channel"], status_str, str(st["pending"]))
                console.print()
                console.print(ch_table)
                console.print()
            continue

        if user_input.startswith("/"):
            try:
                from .commands import handle_command
                handle_command(agent, user_input, speak=speak)
            except Exception as exc:  # Catch-all: protect main loop from command crashes
                show_error(f"Command failed: {exc}")
            _current_model = agent.brain._model_override or "auto"
            _token_used = estimate_messages_tokens(agentic._conversation_history)
            _token_limit = get_context_limit(_current_model)
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue

        if not _session_initialized:
            agentic_session.new(project_root=project_root, model=agent.brain._model_override or "auto")
            _session_initialized = True

        _last_user_input = user_input

        # ── Plan-Approve-Execute: generate plan first, then execute on approval ──
        from .permissions_ui import is_plan_approve_mode
        if is_plan_approve_mode(current_perm_mode):
            from .plan_mode import (
                render_plan, show_plan_approval, edit_plan_text,
                parse_plan_from_llm,
            )
            show_info("Generating plan...")
            plan_result = agentic.plan_first(user_input)

            if plan_result.get("error"):
                show_error(f"Plan generation failed: {plan_result['error']}")
                continue

            plan = plan_result.get("plan")
            plan_text = plan_result.get("plan_text", "")
            if not plan or not plan.steps:
                show_error("Could not generate a plan. Try rephrasing.")
                continue

            # Approval loop: user can edit and re-approve
            while True:
                approval = show_plan_approval(console, plan)
                if approval == "y":
                    # Execute with trust mode (auto-approve tools during plan execution)
                    show_info("Executing plan...")
                    _prev_trust = permissions.trust_mode
                    permissions.set_trust_mode(True)
                    try:
                        from .display import StreamingResponse as _PlanStreamResp
                        streamer = _PlanStreamResp(model=_current_model)
                        streamer.start()

                        def _plan_on_chunk(text: str) -> None:
                            streamer.chunk(text)

                        def _plan_on_tool_call(name: str, args: dict[str, Any], _result: Any) -> None:
                            streamer.pause()
                            desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                            if not desc and "command" in args:
                                desc = args["command"][:60]
                            show_tool_call(name, str(desc))
                            streamer.resume()

                        result = agentic.run(
                            user_input,
                            on_tool_call=_plan_on_tool_call,
                            on_chunk=_plan_on_chunk,
                            steering_queue=steering,
                        )
                        streamer.finish()
                    except KeyboardInterrupt:
                        streamer.pause()
                        agentic.cancel()
                        show_info("Cancelled.")
                        agentic._cancel_event.clear()
                        result = None
                    except Exception as exc:
                        streamer.pause()
                        show_error(str(exc))
                        result = None
                    finally:
                        permissions.set_trust_mode(_prev_trust)
                    break
                elif approval == "e":
                    edited_text = edit_plan_text(console, plan_text)
                    if edited_text != plan_text:
                        plan_text = edited_text
                        plan = parse_plan_from_llm(edited_text)
                    continue  # Show the edited plan for re-approval
                else:
                    show_info("Plan cancelled.")
                    result = None
                    break

            # Skip normal execution path — jump to result handling
            if result is None:
                continue

            response_text = result.get("response", "")
            model_used = result.get("model", _current_model)

            _ERROR_SENTINELS = ["I'm having trouble processing", "[LLM Error]"]
            is_error = result.get("success") is False or any(response_text.startswith(s) for s in _ERROR_SENTINELS)
            if is_error:
                show_error(response_text)
            else:
                if not streamer.displayed and response_text:
                    show_response(response_text, model=model_used, stream=False)

            # Track in ConversationManager (plan-approve path)
            if _cm_conv_id:
                try:
                    _cm = get_conversation_manager()
                    _cm.on_message_added(_cm_conv_id, "user", user_input, "cli", "local")
                    _cm.on_message_added(_cm_conv_id, "assistant", response_text, "cli", "local")
                except Exception:
                    pass

            _msg_count += 1
            if _msg_count == 1 and user_input:
                _session_title = user_input[:50].strip()
            _current_model = agent.brain._model_override or "auto"
            _token_used = estimate_messages_tokens(agentic._conversation_history)
            _token_limit = get_context_limit(_current_model)
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue

        # ── Normal execution path ──
        from .display import StreamingResponse
        streamer = StreamingResponse(model=_current_model)
        streamer.start()
        try:
            def _on_chunk(text: str) -> None:
                streamer.chunk(text)

            def _on_tool_call(name: str, args: dict[str, Any], _result: Any) -> None:
                streamer.pause()
                if hook_mgr:
                    hook_mgr.fire(HookEvent.PRE_TOOL_CALL, {
                        "tool_name": name,
                        "tool_args": str(args)[:500],
                    })
                desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                if not desc and "command" in args:
                    desc = args["command"][:60]
                show_tool_call(name, str(desc))
                if hook_mgr:
                    hook_mgr.fire(HookEvent.POST_TOOL_CALL, {
                        "tool_name": name,
                        "tool_args": str(args)[:500],
                    })
                if hook_mgr and name in ("edit_file", "write_file"):
                    hook_mgr.fire(HookEvent.POST_EDIT, {
                        "tool_name": name,
                        "file_path": args.get("path", args.get("file_path", "")),
                    })
                streamer.resume()

            result = agentic.run(
                user_input,
                on_tool_call=_on_tool_call,
                on_chunk=_on_chunk,
                steering_queue=steering,
            )
        except KeyboardInterrupt:
            # First Ctrl+C: graceful cancel -- signal the loop and wait
            streamer.pause()
            steering.clear()
            agentic.cancel()
            show_info("Cancelling... (press Ctrl+C again to force stop)")
            try:
                import time as _cancel_time
                _cancel_deadline = _cancel_time.time() + 2.0
                while _cancel_time.time() < _cancel_deadline:
                    _cancel_time.sleep(0.1)
            except KeyboardInterrupt:
                # Second Ctrl+C: force break immediately
                show_info("Force stopped.")
            agentic._cancel_event.clear()
            continue
        except Exception as exc:  # Catch-all: protect main loop from crash
            streamer.pause()
            show_error(str(exc))
            continue

        streamer.finish()

        if result is None:
            show_error("No response received.")
            continue

        response_text = result.get("response", "")
        model_used = result.get("model", _current_model)

        _ERROR_SENTINELS = ["I'm having trouble processing", "[LLM Error]"]
        is_error = result.get("success") is False or any(response_text.startswith(s) for s in _ERROR_SENTINELS)
        if is_error:
            show_error(response_text)
            continue

        _ctx_memory_count = 0
        _ctx_mood = ""
        _ctx_tool_count = 0
        try:
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'memories'):
                _ctx_memory_count = len(agent.memory.memories)
            elif hasattr(agent, 'memory') and hasattr(agent.memory, 'count'):
                _ctx_memory_count = agent.memory.count()
        except (TypeError, AttributeError):
            logger.debug("ctx_memory_count_failed", exc_info=True)
        try:
            if hasattr(agent, 'mood') and agent.mood:
                _ctx_mood = str(agent.mood.get("mood", "")) if isinstance(agent.mood, dict) else str(agent.mood)
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

        if not streamer.displayed and response_text:
            show_response(response_text, model=model_used, stream=False)

        if activity_log:
            try:
                activity_log.log(
                    prompt=user_input,
                    response=response_text[:20000] if response_text else "",
                    model=result.get("model", ""),
                    session_id=getattr(agentic_session, 'session_id', ''),
                    tool_calls=result.get("tool_calls", 0),
                )
            except (OSError, TypeError, ValueError):
                logger.debug("activity_log_write_failed", exc_info=True)

        # Track in ConversationManager (normal path)
        if _cm_conv_id:
            try:
                _cm = get_conversation_manager()
                _cm.on_message_added(_cm_conv_id, "user", user_input, "cli", "local")
                _cm.on_message_added(_cm_conv_id, "assistant", response_text, "cli", "local")
            except Exception:
                pass

        follow_up = steering.pop_follow_up()
        if follow_up and _follow_up_depth < _MAX_FOLLOW_UP_DEPTH:
            _pending_follow_up = follow_up
            _follow_up_depth += 1
        elif follow_up:
            show_info("Max auto-follow-up depth reached, dropping follow-up.")

        _msg_count += 1
        if _msg_count == 1 and user_input:
            _session_title = user_input[:50].strip()
        _current_model = agent.brain._model_override or "auto"
        _token_used = estimate_messages_tokens(agentic._conversation_history)
        _token_limit = get_context_limit(_current_model)
        cost_usd = 0.0
        try:
            stats = agent.brain.get_session_stats()
            cost_usd = stats.get("cost_usd", 0.0)
        except (AttributeError, TypeError, KeyError):
            logger.debug("session_stats_read_failed", exc_info=True)
        _show_bar(
            model=_current_model, project_type=_project_type,
            session_title=_session_title, message_count=_msg_count,
            cost_usd=cost_usd,
            token_used=_token_used, token_limit=_token_limit,
            permission_mode=current_perm_mode,
        )

        if hook_mgr:
            hook_mgr.fire(HookEvent.POST_RESPONSE, {
                "response": response_text[:500] if response_text else "",
                "model": model_used,
            })

        if speak and response_text:
            try:
                agent._speak(response_text)
            except (OSError, RuntimeError, AttributeError):
                logger.warning("tts_speak_failed", exc_info=True)

        # Drain channel messages that arrived during agent execution
        _drain_channel_messages()
