import argparse
import logging

logger = logging.getLogger(__name__)

"""Subcommand handlers for Aura Dev CLI.

Handles: aura init, aura doctor, aura config, aura models, aura commit, aura cost
"""

import json
import os
import subprocess
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

try:
    from aura.cli.display import console
except ImportError:
    from rich.console import Console
    console = Console()


def _create_subcommand_permission_manager():
    """Create a permission manager for top-level subcommands.

    Subcommands don't have a live CLIContext, so they need a lightweight
    confirm callback to preserve the same policy model used in chat mode.
    """
    from aura.core.permissions import PermissionManager

    permissions = PermissionManager()
    permissions.set_mode("careful")

    def _confirm(tool_name: str, description: str) -> bool | str:
        console.print()
        console.print("  [bold yellow]Permission required:[/]")
        console.print(f"    {tool_name}")
        if description:
            for line in description.split("\n"):
                console.print(f"    [dim]{line}[/]")
        try:
            response = input("    Allow? [y/n/always]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response == "always":
            return "always"
        return response in ("y", "yes")

    permissions.set_confirm_callback(_confirm)
    return permissions


def handle_subcommand(command: str, args: argparse.Namespace) -> int:
    """Dispatch subcommand. Returns exit code."""
    handlers = {
        "init": cmd_init,
        "setup": cmd_setup,
        "doctor": cmd_doctor,
        "config": cmd_config,
        "models": cmd_models,
        "commit": cmd_commit,
        "cost": cmd_cost,
        "ide": cmd_ide_setup,
    }
    handler = handlers.get(command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            return 130
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            return 1
    console.print(f"[red]Unknown command:[/] {command}")
    return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Create AURA.md in the current project."""
    from aura.tools.code_search import CodeSearchTool

    cwd = os.getcwd()
    aura_md = os.path.join(cwd, "AURA.md")

    if os.path.exists(aura_md):
        console.print(f"  AURA.md already exists at [cyan]{aura_md}[/]")
        return 0

    # Detect project type for template customization
    searcher = CodeSearchTool()
    project_info = searcher.detect_project_type(cwd)

    # Build template with frontmatter
    project_type = project_info.get("project_type", "unknown")
    frameworks = project_info.get("frameworks", [])
    stack = project_info.get("stack", [])

    # Detect test command
    test_cmd = _detect_test_cmd(cwd)

    frontmatter_lines = [
        "---",
        "tier: balanced",
        "# model: qwen3.5:397b-cloud",
    ]
    if test_cmd:
        frontmatter_lines.append(f"test_cmd: {test_cmd}")
        frontmatter_lines.append("auto_test: true")
    frontmatter_lines.extend([
        "# permissions:",
        "#   shell: auto",
        "#   edit_file: auto",
        "# max_iterations: 50",
        "# budget: 5.0",
        "---",
        "",
    ])

    body_lines = [
        f"# {Path(cwd).name}",
        "",
    ]
    if stack:
        body_lines.append(f"Stack: {', '.join(stack)}")
    if frameworks:
        body_lines.append(f"Frameworks: {', '.join(frameworks)}")
    body_lines.extend([
        "",
        "## Instructions",
        "",
        "<!-- Add project-specific instructions for Aura here -->",
        "",
    ])

    content = "\n".join(frontmatter_lines + body_lines)

    with open(aura_md, "w", encoding="utf-8") as f:
        f.write(content)

    console.print(f"  [green]Created[/] {aura_md}")
    if project_type != "unknown":
        console.print(f"  Detected: [cyan]{project_type}[/] project ({', '.join(stack)})")
    if test_cmd:
        console.print(f"  Test command: [cyan]{test_cmd}[/]")
    console.print("\n  Edit AURA.md to customize Aura's behavior for this project.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check Ollama, models, dependencies."""
    console.print("\n[bold]Aura Doctor[/]\n")
    all_ok = True

    # 1. Check Ollama
    console.print("  [bold]Ollama:[/]")
    try:
        import ollama
        models = ollama.list()
        model_names = [m.get("name", m.get("model", "?")) for m in models.get("models", [])]
        console.print(f"    [green]Running[/], {len(model_names)} models loaded")
        for name in sorted(model_names)[:15]:
            console.print(f"      [dim]{name}[/]")
        if len(model_names) > 15:
            console.print(f"      [dim]... and {len(model_names) - 15} more[/]")
    except Exception as e:
        console.print(f"    [red]ERROR[/] Not reachable: {e}")
        console.print("    Run: [cyan]ollama serve[/]")
        all_ok = False

    # 2. Check key dependencies
    dep_table = Table(show_header=False, box=None, padding=(0, 2))
    dep_table.add_column("Package", style="bold")
    dep_table.add_column("Status")
    deps = [
        ("rich", "rich"),
        ("prompt_toolkit", "prompt_toolkit"),
        ("yaml", "PyYAML"),
        ("ollama", "ollama"),
    ]
    for module, pkg in deps:
        try:
            __import__(module)
            dep_table.add_row(pkg, "[green]OK[/]")
        except ImportError:
            dep_table.add_row(pkg, f"[red]MISSING[/] (pip install {pkg})")
            all_ok = False
    console.print("\n  [bold]Dependencies:[/]")
    console.print(dep_table)

    # 3. Check optional tools
    console.print("\n  [bold]Optional tools:[/]")
    optionals = [
        ("aura.tools.brave_search", "BraveSearchTool", "BRAVE_API_KEY"),
        ("aura.tools.tavily_tool", "TavilyTool", "TAVILY_API_KEY"),
    ]
    for module, cls, env_var in optionals:
        try:
            __import__(module)
            has_key = bool(os.environ.get(env_var))
            status = "[green]OK[/]" if has_key else f"[yellow]no {env_var}[/]"
            console.print(f"    {cls}: {status}")
        except ImportError:
            console.print(f"    {cls}: [dim]not installed[/]")

    # 4. Check AURA.md
    console.print("\n  [bold]Project:[/]")
    aura_md = os.path.join(os.getcwd(), "AURA.md")
    if os.path.exists(aura_md):
        console.print("    AURA.md: [green]found[/]")
    else:
        console.print("    AURA.md: [yellow]not found[/] (run: [cyan]aura init[/])")

    # 5. Check git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5, cwd=os.getcwd(),
        )
        if result.returncode == 0:
            console.print("    Git repo: [green]yes[/]")
        else:
            console.print("    Git repo: [yellow]no[/]")
    except Exception:
        console.print("    Git: [dim]not available[/]")

    if all_ok:
        console.print("\n  [bold green]All checks passed![/]\n")
    else:
        console.print("\n  [bold yellow]Some issues found.[/]\n")
    return 0 if all_ok else 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show current configuration including AURA.md overrides."""
    from aura.config import Config
    from aura.core.context import get_aura_md_config

    console.print("\n[bold]Aura Configuration[/]\n")

    # Global config
    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column("Setting", style="bold")
    tbl.add_column("Value", style="cyan")
    tbl.add_row("Model (fast)", str(Config.MODEL_FAST))
    tbl.add_row("Model (reason)", str(Config.MODEL_REASON))
    tbl.add_row("Model (code)", str(Config.MODEL_CODE))
    tbl.add_row("Ollama host", str(getattr(Config, "OLLAMA_HOST", "http://localhost:11434")))
    console.print("  [bold]Global:[/]")
    console.print(tbl)

    # Model chains
    chains = {
        "Fast chain": getattr(Config, "MODEL_FAST_CHAIN", []),
        "Reason chain": getattr(Config, "MODEL_REASON_CHAIN", []),
        "Code chain": getattr(Config, "MODEL_CODE_CHAIN", []),
    }
    has_chains = any(chains.values())
    if has_chains:
        console.print("\n  [bold]Model chains:[/]")
        for label, chain in chains.items():
            if chain:
                console.print(f"    {label}: [cyan]{' -> '.join(chain)}[/]")

    # Project-level AURA.md overrides
    aura_config = get_aura_md_config(os.getcwd())
    if aura_config:
        proj_tbl = Table(show_header=False, box=None, padding=(0, 2))
        proj_tbl.add_column("Key", style="bold")
        proj_tbl.add_column("Value", style="cyan")
        for key in ["tier", "model", "test_cmd", "auto_test", "max_iterations", "budget"]:
            val = aura_config.get(key)
            if val is not None:
                proj_tbl.add_row(key, str(val))
        perms = aura_config.get("permissions")
        if perms:
            proj_tbl.add_row("permissions", str(perms))
        console.print("\n  [bold]Project (AURA.md):[/]")
        console.print(proj_tbl)
    else:
        console.print(f"\n  No AURA.md found in {os.getcwd()} (run: [cyan]aura init[/])")

    console.print()
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List available models with routing roles."""
    from aura.core.router import ROUTING_TABLE

    console.print("\n[bold]Aura Model Routing[/]\n")

    tbl = Table(box=None, padding=(0, 2))
    tbl.add_column("Category", style="bold")
    tbl.add_column("local", style="dim")
    tbl.add_column("balanced", style="cyan")
    tbl.add_column("max", style="green")

    for category, tiers in ROUTING_TABLE.items():
        tbl.add_row(
            category,
            tiers.get("local", "-"),
            tiers.get("balanced", "-"),
            tiers.get("max", "-"),
        )
    console.print(tbl)

    # Show which models are actually available
    console.print()
    try:
        import ollama
        models = ollama.list()
        available = {m.get("name", m.get("model", "")) for m in models.get("models", [])}
        console.print(f"  [green]{len(available)}[/] models available locally")
    except Exception:
        console.print("  [dim](Could not check available models — is Ollama running?)[/]")

    console.print()
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    """Smart commit with AI-generated message."""
    from aura import ApprenticeAgent
    from aura.tools.git_tool import GitTool

    git = GitTool()
    cwd = os.getcwd()
    permissions = _create_subcommand_permission_manager()

    # Check for changes
    status = git.status(cwd)
    if not status.get("success"):
        console.print("[red]Not in a git repository or git error.[/]")
        return 1

    diff_result = git.diff(cwd)
    diff_text = diff_result.get("diff", "")

    if not diff_text and not status.get("dirty_count", 0):
        console.print("No changes to commit.")
        return 0

    # Stage all if --all flag
    if getattr(args, 'all', False):
        if not permissions.check("git", {"action": "add", "files": "."}):
            console.print("  Cancelled.")
            return 0
        add_result = git.add(cwd, files=".")
        if not add_result.get("success"):
            console.print(f"[red]Stage failed:[/] {add_result.get('error', 'unknown error')}")
            return 1

    # Get diff of staged changes
    try:
        subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )
        staged_diff_proc = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )
        diff_text = staged_diff_proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        diff_text = diff_result.get("diff", "")

    if not diff_text:
        console.print("No staged changes. Use [cyan]git add[/] first or pass [cyan]--all[/].")
        return 1

    # Generate commit message
    console.print("[dim]Generating commit message...[/]")
    try:
        agent = ApprenticeAgent()
        # Use more diff context for better messages
        max_diff = 8000
        truncated = f"\n... (truncated {len(diff_text) - max_diff} chars)" if len(diff_text) > max_diff else ""
        prompt = f"""Generate a concise git commit message for these changes.
Return ONLY the commit message (1-2 lines), no explanation.

Diff:
{diff_text[:max_diff]}{truncated}"""

        result = agent.brain.think(prompt, use_history=False)
        message = result.strip().strip('"').strip("'").strip("`")

        # Clean up common LLM artifacts
        for prefix in ["commit message:", "here's the commit message:", "here's",
                        "here is the commit message:", "here is", "message:"]:
            if message.lower().startswith(prefix):
                message = message[len(prefix):].strip().strip('"').strip("'")
                break

        if not message:
            console.print("[red]Error:[/] LLM returned empty commit message.")
            return 1

    except Exception as e:
        console.print(f"[red]Error generating message:[/] {e}")
        return 1

    console.print(f"\n  Commit message: [bold]{message}[/]\n")
    try:
        confirm = input("  Edit commit message before approval? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 1

    if confirm == "edit" or confirm == "e":
        try:
            message = input("  Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    elif confirm in ("y", "yes"):
        try:
            message = input("  Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    elif confirm not in ("n", "no", ""):
        console.print("  Cancelled.")
        return 0

    if not permissions.check("git", {"action": "commit", "message": message}):
        console.print("  Cancelled.")
        return 0

    result = git.commit(cwd, message=message)
    if result.get("success"):
        console.print(f"  [green]Committed:[/] {message}")
        return 0
    else:
        console.print(f"  [red]Commit failed:[/] {result.get('error', 'unknown error')}")
        return 1


def cmd_cost(args: argparse.Namespace) -> int:
    """Show session cost breakdown from activity log."""
    try:
        from aura.cli.activity_log import ActivityLog
        log = ActivityLog()
        stats = log.get_stats()
    except (ImportError, OSError) as e:
        console.print(f"\n[red]Could not read activity log:[/] {e}")
        console.print("Cost data is tracked during interactive sessions.\n")
        return 1

    console.print("\n[bold]Aura Cost Summary[/]\n")
    total_cost = stats.get("total_cost", 0.0)
    total_interactions = stats.get("total_interactions", 0)
    tokens_in = stats.get("total_tokens_in", 0)
    tokens_out = stats.get("total_tokens_out", 0)
    total_tokens = tokens_in + tokens_out
    total_tool_calls = stats.get("total_tool_calls", 0)

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", style="cyan")
    tbl.add_row("Total cost", f"${total_cost:.4f}")
    tbl.add_row("Interactions", str(total_interactions))
    tbl.add_row("Tokens", f"{total_tokens:,} (in: {tokens_in:,} / out: {tokens_out:,})")
    tbl.add_row("Tool calls", str(total_tool_calls))
    console.print(tbl)

    console.print()
    return 0


def cmd_ide_setup(args: argparse.Namespace) -> int:
    """Generate VS Code tasks.json and print MCP config snippet."""
    import sys as _sys

    cwd = os.getcwd()
    vscode_dir = os.path.join(cwd, ".vscode")
    tasks_path = os.path.join(vscode_dir, "tasks.json")

    # Resolve the current Python interpreter and absolute path to Aura's main.py.
    # `python -m main` would only work when cwd contains a top-level main module,
    # which is not the case for arbitrary user projects.
    main_py = str(Path(__file__).resolve().parents[2] / "main.py")
    aura_cmd = f'"{_sys.executable}" "{main_py}"'

    # Aura tasks for VS Code
    aura_tasks = [
        {
            "label": "Aura: Chat",
            "type": "shell",
            "command": aura_cmd,
            "presentation": {"reveal": "always", "panel": "dedicated"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Run Prompt",
            "type": "shell",
            "command": f'{aura_cmd} -p "${{input:auraPrompt}}"',
            "presentation": {"reveal": "always"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Init Project",
            "type": "shell",
            "command": f"{aura_cmd} init",
            "presentation": {"reveal": "always"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Smart Commit",
            "type": "shell",
            "command": f"{aura_cmd} commit --all",
            "presentation": {"reveal": "always"},
            "problemMatcher": [],
        },
    ]

    inputs = [
        {
            "id": "auraPrompt",
            "description": "What should Aura do?",
            "type": "promptString",
        },
    ]

    # Merge with existing tasks.json if present
    if os.path.exists(tasks_path):
        try:
            with open(tasks_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {"version": "2.0.0", "tasks": []}

        # Remove old Aura tasks
        existing_tasks = [
            t for t in existing.get("tasks", [])
            if not t.get("label", "").startswith("Aura:")
        ]
        existing_tasks.extend(aura_tasks)
        existing["tasks"] = existing_tasks

        # Add inputs if not present
        existing_inputs = existing.get("inputs", [])
        existing_input_ids = {i.get("id") for i in existing_inputs}
        for inp in inputs:
            if inp["id"] not in existing_input_ids:
                existing_inputs.append(inp)
        existing["inputs"] = existing_inputs

        tasks_data = existing
    else:
        tasks_data = {
            "version": "2.0.0",
            "tasks": aura_tasks,
            "inputs": inputs,
        }

    # Write tasks.json
    os.makedirs(vscode_dir, exist_ok=True)
    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks_data, f, indent=2)

    console.print(f"\n  [green]Created[/] {tasks_path}")
    console.print("    - Aura: Chat (interactive mode)")
    console.print("    - Aura: Run Prompt (one-shot)")
    console.print("    - Aura: Init Project")
    console.print("    - Aura: Smart Commit")

    # Print MCP config snippet
    # Point cwd at Aura's install dir so `python -m aura.core.mcp_server` can
    # resolve the package without requiring a site-packages install.
    aura_root = str(Path(__file__).resolve().parents[2])
    console.print("\n  [bold]MCP Server config for VS Code settings.json:[/]\n")
    mcp_config = {
        "mcp.servers": {
            "aura": {
                "command": _sys.executable,
                "args": ["-m", "aura.core.mcp_server"],
                "cwd": aura_root,
            }
        }
    }
    console.print(Panel(json.dumps(mcp_config, indent=4), border_style="dim"))
    console.print()
    return 0


def _detect_test_cmd(project_root: str) -> str:
    """Try to detect the project's test command."""
    root = Path(project_root)

    # Python
    if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
        return "pytest"
    if (root / "setup.py").exists():
        return "python -m pytest"

    # Node
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text())
            scripts = pkg.get("scripts", {})
            if "test" in scripts:
                test_script = scripts["test"]
                if "vitest" in test_script:
                    return "npx vitest run"
                elif "jest" in test_script:
                    return "npx jest"
                return "npm test"
        except Exception as e:
            logger.debug(f"[Commands] non-critical: {e}")
    # Rust
    if (root / "Cargo.toml").exists():
        return "cargo test"

    # Go
    if (root / "go.mod").exists():
        return "go test ./..."

    return ""


def cmd_setup(args: argparse.Namespace) -> int:
    """Interactive setup wizard for configuring Aura in a project."""
    cwd = os.getcwd()
    project_name = Path(cwd).name

    console.print(f"\n  [bold]Aura Setup Wizard[/] for [cyan]'{project_name}'[/]\n")

    # Step 1: Detect project type
    console.print("  [bold]Step 1:[/] Detecting project...")
    try:
        from aura.tools.code_search import CodeSearchTool
        searcher = CodeSearchTool()
        info = searcher.detect_project_type(cwd)
        project_type = info.get("project_type", "unknown")
        frameworks = info.get("frameworks", [])
        stack = info.get("stack", [])
        if project_type != "unknown":
            console.print(f"    Detected: [cyan]{project_type}[/] ({', '.join(stack)})")
        else:
            console.print("    [yellow]Could not auto-detect project type.[/]")
    except Exception:
        project_type = "unknown"
        frameworks = []
        stack = []
        console.print("    [yellow]Could not auto-detect (code_search unavailable).[/]")

    # Step 2: Choose tier
    console.print("\n  [bold]Step 2:[/] Choose model tier")
    console.print("    [cyan]fast[/]     — Quick responses, lower cost")
    console.print("    [cyan]balanced[/] — Good balance of speed and quality (recommended)")
    console.print("    [cyan]max[/]      — Best quality, higher cost")
    tier = _prompt("    Tier", "balanced")
    if tier not in ("fast", "balanced", "max"):
        console.print(f"    [yellow]Invalid tier '{tier}', using 'balanced'.[/]")
        tier = "balanced"

    # Step 3: Model
    model = _prompt("\n  Step 3: Model (or 'auto' for smart routing)", "auto")

    # Step 4: Test command
    detected_test = _detect_test_cmd(cwd)
    default_test = detected_test or "pytest"
    test_cmd = _prompt("\n  Step 4: Test command", default_test)
    auto_test_str = _prompt("    Auto-run tests after edits? [y/n]", "y")
    auto_test = auto_test_str.lower() in ("y", "yes")

    # Step 5: API keys
    console.print("\n  [bold]Step 5:[/] Checking API keys...")
    for key_name in ["OLLAMA_API_KEY", "BRAVE_API_KEY", "TAVILY_API_KEY"]:
        has = bool(os.environ.get(key_name))
        status = "[green]found[/]" if has else "[yellow]not set[/]"
        console.print(f"    {key_name}: {status}")

    # Step 6: Generate AURA.md
    console.print("\n  [bold]Step 6:[/] Creating AURA.md...")
    aura_md_path = os.path.join(cwd, "AURA.md")

    if os.path.exists(aura_md_path):
        overwrite = _prompt("    AURA.md already exists. Overwrite? [y/n]", "n")
        if overwrite.lower() not in ("y", "yes"):
            console.print("    Kept existing AURA.md.")
            console.print("\n  [bold green]Setup complete![/] Run [cyan]aura[/] to start.\n")
            return 0

    # Build content
    lines = ["---", f"tier: {tier}"]
    if model and model != "auto":
        lines.append(f"model: {model}")
    if test_cmd:
        lines.append(f"test_cmd: {test_cmd}")
    if auto_test:
        lines.append("auto_test: true")
    lines.extend([
        "# permissions:",
        "#   shell: auto",
        "#   edit_file: auto",
        "---",
        "",
        f"# {project_name}",
        "",
    ])
    if stack:
        lines.append(f"Stack: {', '.join(stack)}")
    if frameworks:
        lines.append(f"Frameworks: {', '.join(frameworks)}")
    lines.extend([
        "",
        "## Instructions",
        "",
        "<!-- Add project-specific instructions for Aura here -->",
        "",
    ])

    content = "\n".join(lines)
    with open(aura_md_path, "w", encoding="utf-8") as f:
        f.write(content)

    console.print(f"    [green]Created[/] {aura_md_path}")
    console.print("\n  [bold green]Setup complete![/] Run [cyan]aura[/] to start.\n")
    return 0


def _prompt(text: str, default: str) -> str:
    """Prompt with a default value shown in brackets."""
    try:
        value = input(f"{text} [{default}]: ").strip()
        return value if value else default
    except (EOFError, KeyboardInterrupt):
        return default
