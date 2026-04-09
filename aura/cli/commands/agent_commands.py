import logging
from typing import Optional

from ..context import get_ctx

logger = logging.getLogger(__name__)


def print_result(result, is_fastpath: bool = False):
    from ..display import console, show_response

    response = result.get("response", "")
    if response:
        show_response(response)
    else:
        mode = "Fast-path" if is_fastpath else f"{result.get('iterations', '?')} iterations"
        console.print(f"[dim]Completed ({mode})[/dim]")


def handle_goal(agent, arg, context) -> Optional[str]:
    if arg:
        from ..display import show_error as _goal_err
        from ..display import show_info as _goal_info
        from ..display import show_response as _goal_resp
        from ..display import show_tool_call as _goal_tool
        _goal_info("Running goal...")
        try:
            _goal_agentic = get_ctx().agentic_loop if get_ctx() else None
            if _goal_agentic:
                def _goal_on_tool_call(name, args, _result):
                    desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                    if not desc and "command" in args:
                        desc = args["command"][:60]
                    _goal_tool(name, str(desc))
                result = _goal_agentic.run(arg, on_tool_call=_goal_on_tool_call)
                _goal_resp(result.get("response", ""), model=result.get("model", ""))
            else:
                result = agent.run(arg)
                print_result(result)
        except Exception as e:  # Catch-all: protect CLI from goal execution crash
            _goal_err(str(e))
    else:
        from ..display import console
        console.print("Usage: /goal <your goal>")


def handle_plan(agent, arg, context) -> Optional[str]:
    if arg:
        from ..display import console as _plan_console
        from ..display import show_info as _plan_info
        from ..display import show_tool_call as _plan_tool
        from ..plan_mode import PLAN_GENERATION_PROMPT, StepStatus, parse_plan_from_llm, render_plan
        _plan_info("Generating plan...")
        prompt = PLAN_GENERATION_PROMPT.format(task=arg)
        try:
            response = agent.brain.think(prompt)
            if isinstance(response, dict):
                response = response.get("response", response.get("content", str(response)))
        except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            _plan_console.print(f"  Error generating plan: {e}")
            return

        plan = parse_plan_from_llm(response)
        render_plan(_plan_console, plan)

        _plan_console.print("\n[dim]Execute this plan? (y/n)[/dim]")
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if choice in ("y", "yes"):
            agentic_loop = get_ctx().agentic_loop if get_ctx() else None
            for step in plan.steps:
                step.status = StepStatus.RUNNING
                render_plan(_plan_console, plan)
                try:
                    if agentic_loop:
                        def _plan_on_tool_call(name, args, _result):
                            desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                            if not desc and "command" in args:
                                desc = args["command"][:60]
                            _plan_tool(name, str(desc))
                        result = agentic_loop.run(step.description, on_tool_call=_plan_on_tool_call)
                    else:
                        result = agent.run(step.description)
                    if result.get("success"):
                        step.status = StepStatus.DONE
                        step.result = result.get("response", "")[:200]
                    else:
                        step.status = StepStatus.FAILED
                        step.error = result.get("error", result.get("response", "Step failed"))[:200]
                except Exception as e:  # Catch-all: step failure must not abort entire plan
                    step.status = StepStatus.FAILED
                    step.error = str(e)[:200]

                if step.status == StepStatus.FAILED:
                    render_plan(_plan_console, plan)
                    _plan_console.print("[yellow]Step failed. Continue? (y/n)[/yellow]")
                    try:
                        cont = input("> ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if cont not in ("y", "yes"):
                        break

            render_plan(_plan_console, plan)
            _plan_console.print("[green]Plan execution complete.[/green]")
        else:
            _plan_console.print("  Plan cancelled.")
        return
    else:
        from ..display import console as _plan_console
        _plan_console.print("Usage: /plan <task description>")


def handle_fleet(agent, arg, context) -> Optional[str]:
    task = arg.strip()
    if not task:
        from ..display import console as _fleet_console
        _fleet_console.print("[dim]Usage: /fleet <task description>[/dim]")
        return
    from ..display import console as _fleet_console
    from ..fleet import (
        DECOMPOSITION_PROMPT,
        FleetExecutor,
        FleetRun,
        parse_decomposition,
        render_fleet_dashboard,
        run_fleet_live,
    )
    prompt = DECOMPOSITION_PROMPT.format(task=task)
    response = agent.brain.think(prompt)
    if isinstance(response, dict):
        response = response.get("response", response.get("content", str(response)))
    subtasks = parse_decomposition(response)
    if not subtasks:
        from ..display import show_error as _fleet_err
        _fleet_err("Could not decompose task into sub-tasks.")
        return
    fleet = FleetRun(goal=task, tasks=subtasks)
    render_fleet_dashboard(_fleet_console, fleet)
    _fleet_console.print(f"\n[dim]Execute {len(subtasks)} tasks in parallel? (y/n)[/dim]")
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if choice not in ("y", "yes"):
        return
    def _fleet_task_fn(prompt):
        try:
            response = agent.brain.think(prompt)
            if isinstance(response, dict):
                response = response.get("response", response.get("content", str(response)))
            return {"success": True, "response": response or "", "iterations": 1}
        except Exception as e:  # Catch-all: runs in thread pool, must not propagate
            return {"success": False, "error": str(e)}
    executor = FleetExecutor(max_workers=3)
    run_fleet_live(_fleet_console, fleet, executor, _fleet_task_fn)


def handle_agent(agent, arg, context) -> Optional[str]:
    _handle_agent_command(agent, arg)


def handle_hand(agent, arg, context) -> Optional[str]:
    _handle_hand_command(agent, arg)


def _handle_agent_command(agent, arg: str):
    from ..display import console as _agent_console
    try:
        from aura.core.permissions import PermissionTier
        pm = getattr(agent, "permissions", None)
        if pm and pm.current_mode != PermissionTier.FULL_AUTO:
            if arg:
                confirm = input(f"  Spawn agent for: {arg[:60]}\n  Confirm? (y/n): ").strip().lower()
                if confirm not in ("y", "yes"):
                    _agent_console.print("  Cancelled.")
                    return
    except (ImportError, AttributeError, EOFError, KeyboardInterrupt):
        logger.debug("agent_permission_check_skipped", exc_info=True)

    if not hasattr(agent, 'orchestrator') or agent.orchestrator is None:
        try:
            from aura.multi_agent.orchestrator import MultiAgentOrchestrator

            def llm_func(system_prompt, user_message):
                return agent.brain.think(user_message, system_prompt=system_prompt, use_history=False)

            agent.orchestrator = MultiAgentOrchestrator(
                tool_registry=agent.tools,
                llm_func=llm_func
            )
        except (ImportError, AttributeError, TypeError) as e:
            _agent_console.print(f"Multi-agent system not available: {e}")
            return

    specialists = list(agent.orchestrator.specialists.keys())

    if not arg:
        _agent_console.print("\n  Available specialists:")
        for name in specialists:
            spec = agent.orchestrator.specialists[name]
            desc = getattr(spec, 'description', name.capitalize())
            _agent_console.print(f"    {name:12s} - {desc}")
        _agent_console.print("\n  Usage: /agent <specialist> <task>")
        _agent_console.print("         /agent parallel <task>")
        return

    parts = arg.split(maxsplit=1)
    specialist = parts[0].lower()
    task = parts[1] if len(parts) > 1 else ""

    if not task:
        _agent_console.print(f"Usage: /agent {specialist} <task>")
        return

    if specialist == "parallel":
        _agent_console.print(f"  Running all specialists in parallel on: {task[:60]}...")
        from aura.multi_agent.protocol import AgentMessage
        message = AgentMessage(content=task, sender="user")
        results = agent.orchestrator._execute_parallel(specialists, message)
        for result in results:
            status = "OK" if result.success else "FAIL"
            _agent_console.print(f"\n  [{result.agent.upper()}] ({status}):")
            _agent_console.print(f"  {result.response[:500]}")
    elif specialist in specialists:
        _agent_console.print(f"  [{specialist.upper()}] Working on: {task[:60]}...")
        from aura.multi_agent.protocol import AgentMessage
        message = AgentMessage(content=task, sender="user")
        result = agent.orchestrator._execute_single(specialist, message)
        if result.success:
            _agent_console.print(f"\n  [{specialist.upper()}]:")
            _agent_console.print(f"  {result.response}")
        else:
            _agent_console.print(f"  Error: {result.response}")
    else:
        _agent_console.print(f"  Unknown specialist: {specialist}")
        _agent_console.print(f"  Available: {', '.join(specialists)}, parallel")


def handle_debate(agent, arg, context) -> Optional[str]:
    from ..debate_mode import parse_debate_args, run_debate
    from ..display import console as _debate_console
    if not arg.strip():
        _debate_console.print("[dim]Usage: /debate <question>[/dim]")
        _debate_console.print("[dim]       /debate --models kimi,deepseek,chatgpt <question>[/dim]")
        return
    question, user_models = parse_debate_args(arg)
    if not question:
        from ..display import show_error as _debate_err
        _debate_err("No question provided.")
        return
    try:
        run_debate(agent.brain, question, user_models=user_models)
    except Exception as e:
        from ..display import show_error as _debate_err2
        _debate_err2(f"Debate failed: {e}")


def handle_chain(agent, arg, context) -> Optional[str]:
    """Run, save, load, or list prompt chains."""
    from aura.cli.chain_mode import (
        delete_chain,
        list_chains,
        load_chain,
        parse_chain,
        run_chain,
        save_chain,
    )
    from aura.cli.display import console, show_info, show_response

    if not arg:
        console.print("[dim]  Prompt pipelines — chain prompts where each output feeds the next.[/dim]")
        console.print()
        console.print("  [bold]Usage:[/bold]")
        console.print('    /chain "research X" -> "summarize" -> "write code"')
        console.print('    /chain research X @kimi-k2.5:cloud -> summarize @nemotron-3-super:cloud')
        console.print("    /chain save <name> step1 -> step2 -> step3")
        console.print("    /chain run <name>")
        console.print("    /chain list")
        console.print("    /chain delete <name>")
        console.print()
        console.print("  [dim]Use {prev} in a step to control where previous output is injected.[/dim]")
        console.print("  [dim]Use @model after a step to override the model for that step.[/dim]")
        return

    stripped = arg.strip()

    # /chain list
    if stripped == "list":
        chains = list_chains()
        if not chains:
            console.print("  [dim]No saved chains.[/dim]")
            return
        show_info(f"{len(chains)} saved chain(s)")
        for c in chains:
            loaded = load_chain(c)
            step_count = len(loaded.steps) if loaded else "?"
            console.print(f"  [cyan]\u2022[/cyan] {c} [dim]({step_count} steps)[/dim]")
        return

    # /chain delete <name>
    if stripped.startswith("delete "):
        name = stripped[7:].strip()
        if delete_chain(name):
            console.print(f"  [green]Deleted chain '{name}'.[/green]")
        else:
            console.print(f"  [red]Chain '{name}' not found.[/red]")
        return

    # /chain save <name> step1 -> step2 -> ...
    if stripped.startswith("save "):
        parts = stripped[5:].split(maxsplit=1)
        name = parts[0]
        if len(parts) < 2:
            console.print("  [red]Usage: /chain save <name> step1 -> step2[/red]")
            return
        steps = parse_chain(parts[1])
        if not steps:
            console.print("  [red]No steps parsed. Use '->' to separate steps.[/red]")
            return
        save_chain(name, steps)
        console.print(f"  [green]Saved chain '{name}' ({len(steps)} steps)[/green]")
        return

    # /chain run <name>
    if stripped.startswith("run "):
        name = stripped[4:].strip()
        chain = load_chain(name)
        if not chain:
            console.print(f"  [red]Chain '{name}' not found. Use /chain list.[/red]")
            return
        steps = chain.steps
    else:
        # Direct: /chain step1 -> step2 -> step3
        steps = parse_chain(stripped)

    if not steps:
        console.print("  [red]No steps parsed. Use '->' to separate steps.[/red]")
        return

    # Show chain overview
    show_info(f"Chain: {len(steps)} steps")
    for i, s in enumerate(steps, 1):
        model_tag = f" [dim cyan]@{s.model}[/dim cyan]" if s.model else ""
        prompt_preview = s.prompt_template[:70]
        if len(s.prompt_template) > 70:
            prompt_preview += "..."
        console.print(f"  [dim]{i}.[/dim] {prompt_preview}{model_tag}")
    console.print()

    # Execute with progress callback
    def on_step(step_num, total, result):
        elapsed_str = f"{result['elapsed']:.1f}s"
        model_str = result["model"]
        show_info(f"Step {step_num}/{total} complete ({elapsed_str}, model: {model_str})")
        show_response(result["response"], model=model_str, stream=False)

    result = run_chain(agent.brain, steps, on_step=on_step)

    if result.success:
        show_info(f"Chain complete: {len(steps)} steps in {result.total_elapsed:.1f}s")
    else:
        console.print("  [yellow]Chain finished with errors. Check output above.[/yellow]")


def _handle_hand_command(agent, arg: str):
    from aura.hands.collector import CollectorHand
    from aura.hands.guardian import GuardianHand
    from aura.hands.manager import get_hand_manager
    from aura.hands.memory_hand import MemoryHand
    from aura.hands.researcher import ResearcherHand

    from ..display import console as _hand_console

    manager = get_hand_manager()

    if not manager.list_hands():
        manager.register(ResearcherHand())
        manager.register(GuardianHand())
        manager.register(MemoryHand())
        manager.register(CollectorHand())

    parts = arg.strip().split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "list"
    subarg = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "list" or subcmd == "ls":
        hands = manager.list_hands()
        if not hands:
            _hand_console.print("No hands registered.")
            return
        _hand_console.print(f"\n  {'Name':<15} {'State':<12} {'Runs':<6} {'Cost':<10} {'Failures':<10}")
        _hand_console.print(f"  {'─'*15} {'─'*12} {'─'*6} {'─'*10} {'─'*10}")
        for h in hands:
            _hand_console.print(f"  {h['name']:<15} {h['state']:<12} {h['total_runs']:<6} ${h['total_cost']:<9.4f} {h['consecutive_failures']:<10}")
        _hand_console.print()

    elif subcmd == "activate" and subarg:
        if manager.activate(subarg):
            _hand_console.print(f"  Hand '{subarg}' activated.")
        else:
            _hand_console.print(f"  Unknown hand: {subarg}")

    elif subcmd == "deactivate" and subarg:
        if manager.deactivate(subarg):
            _hand_console.print(f"  Hand '{subarg}' deactivated.")
        else:
            _hand_console.print(f"  Unknown hand: {subarg}")

    elif subcmd == "run" and subarg:
        import asyncio
        hand = manager.get_hand(subarg)
        if not hand:
            _hand_console.print(f"  Unknown hand: {subarg}")
            return
        _hand_console.print(f"  Running hand '{subarg}'...")
        try:
            result = asyncio.run(manager.run_hand(
                subarg,
                brain=getattr(agent, 'brain', None),
                tools=getattr(agent, 'tools', {}),
            ))
            status = "SUCCESS" if result.success else "FAILED"
            _hand_console.print(f"  [{status}] {result.summary}")
            if result.error:
                _hand_console.print(f"  Error: {result.error}")
            _hand_console.print(f"  ({result.iterations} iterations, {result.duration_seconds:.1f}s, ${result.cost_usd:.4f})")
        except Exception as e:  # Catch-all: hand execution may fail in many ways
            _hand_console.print(f"  Hand execution failed: {e}")

    elif subcmd == "status" and subarg:
        hand = manager.get_hand(subarg)
        if not hand:
            _hand_console.print(f"  Unknown hand: {subarg}")
            return
        stats = hand.get_stats()
        _hand_console.print(f"\n  Hand: {stats['name']}")
        _hand_console.print(f"  State: {stats['state']}")
        _hand_console.print(f"  Total runs: {stats['total_runs']}")
        _hand_console.print(f"  Total cost: ${stats['total_cost']:.4f}")
        _hand_console.print(f"  Failures (consecutive): {stats['consecutive_failures']}")
        if stats['last_error']:
            _hand_console.print(f"  Last error: {stats['last_error']}")
        _hand_console.print()

    else:
        _hand_console.print("Usage: /hand <list|activate|deactivate|run|status> [name]")
        _hand_console.print("  /hand list              — Show all hands and their status")
        _hand_console.print("  /hand activate <name>   — Activate a hand for scheduling")
        _hand_console.print("  /hand deactivate <name> — Deactivate a hand")
        _hand_console.print("  /hand run <name>        — Run a hand immediately")
        _hand_console.print("  /hand status <name>     — Show detailed hand status")
