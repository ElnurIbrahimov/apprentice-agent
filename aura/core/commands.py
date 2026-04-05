import logging

logger = logging.getLogger(__name__)

"""Subcommand handlers for Aura Dev CLI.

Handles: aura init, aura doctor, aura config, aura models, aura commit, aura cost
"""

import json
import os
import subprocess
from pathlib import Path


def handle_subcommand(command: str, args) -> int:
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
            print(f"Error: {e}")
            return 1
    print(f"Unknown command: {command}")
    return 1


def cmd_init(args) -> int:
    """Create AURA.md in the current project."""
    from aura.tools.project_context import init_project
    from aura.tools.code_search import CodeSearchTool

    cwd = os.getcwd()
    aura_md = os.path.join(cwd, "AURA.md")

    if os.path.exists(aura_md):
        print(f"AURA.md already exists at {aura_md}")
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
        f"# model: qwen3.5:397b-cloud",
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

    print(f"Created {aura_md}")
    if project_type != "unknown":
        print(f"  Detected: {project_type} project ({', '.join(stack)})")
    if test_cmd:
        print(f"  Test command: {test_cmd}")
    print(f"\n  Edit AURA.md to customize Aura's behavior for this project.")
    return 0


def cmd_doctor(args) -> int:
    """Check Ollama, models, dependencies."""
    print("\nAura Doctor\n")
    all_ok = True

    # 1. Check Ollama
    print("  Ollama:")
    try:
        import ollama
        models = ollama.list()
        model_names = [m.get("name", m.get("model", "?")) for m in models.get("models", [])]
        print(f"    Running, {len(model_names)} models loaded")
        for name in sorted(model_names)[:15]:
            print(f"      {name}")
        if len(model_names) > 15:
            print(f"      ... and {len(model_names) - 15} more")
    except Exception as e:
        print(f"    [ERROR] Not reachable: {e}")
        print(f"    Run: ollama serve")
        all_ok = False

    # 2. Check key dependencies
    print("\n  Dependencies:")
    deps = [
        ("rich", "rich"),
        ("prompt_toolkit", "prompt_toolkit"),
        ("yaml", "PyYAML"),
        ("ollama", "ollama"),
    ]
    for module, pkg in deps:
        try:
            __import__(module)
            print(f"    {pkg}: OK")
        except ImportError:
            print(f"    {pkg}: MISSING (pip install {pkg})")
            all_ok = False

    # 3. Check optional tools
    print("\n  Optional tools:")
    optionals = [
        ("aura.tools.brave_search", "BraveSearchTool", "BRAVE_API_KEY"),
        ("aura.tools.tavily_tool", "TavilyTool", "TAVILY_API_KEY"),
    ]
    for module, cls, env_var in optionals:
        try:
            __import__(module)
            has_key = bool(os.environ.get(env_var))
            status = "OK" if has_key else f"no {env_var}"
            print(f"    {cls}: {status}")
        except ImportError:
            print(f"    {cls}: not installed")

    # 4. Check AURA.md
    print("\n  Project:")
    aura_md = os.path.join(os.getcwd(), "AURA.md")
    if os.path.exists(aura_md):
        print(f"    AURA.md: found")
    else:
        print(f"    AURA.md: not found (run: aura init)")

    # 5. Check git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5, cwd=os.getcwd(),
        )
        if result.returncode == 0:
            print(f"    Git repo: yes")
        else:
            print(f"    Git repo: no")
    except Exception:
        print(f"    Git: not available")

    print(f"\n  {'All checks passed!' if all_ok else 'Some issues found.'}\n")
    return 0 if all_ok else 1


def cmd_config(args) -> int:
    """Show current configuration including AURA.md overrides."""
    from aura.config import Config
    from aura.core.context import get_aura_md_config

    print("\nAura Configuration\n")

    # Global config
    print("  Global:")
    config_items = [
        ("Model (fast)", Config.MODEL_FAST),
        ("Model (reason)", Config.MODEL_REASON),
        ("Model (code)", Config.MODEL_CODE),
        ("Ollama host", getattr(Config, "OLLAMA_HOST", "http://localhost:11434")),
    ]
    for label, value in config_items:
        print(f"    {label:20s}: {value}")

    # Model chains
    chains = {
        "Fast chain": getattr(Config, "MODEL_FAST_CHAIN", []),
        "Reason chain": getattr(Config, "MODEL_REASON_CHAIN", []),
        "Code chain": getattr(Config, "MODEL_CODE_CHAIN", []),
    }
    has_chains = any(chains.values())
    if has_chains:
        print()
        print("  Model chains:")
        for label, chain in chains.items():
            if chain:
                print(f"    {label}: {' -> '.join(chain)}")

    # Project-level AURA.md overrides
    aura_config = get_aura_md_config(os.getcwd())
    if aura_config:
        print()
        print("  Project (AURA.md):")
        for key in ["tier", "model", "test_cmd", "auto_test", "max_iterations", "budget"]:
            val = aura_config.get(key)
            if val is not None:
                print(f"    {key:20s}: {val}")
        perms = aura_config.get("permissions")
        if perms:
            print(f"    {'permissions':20s}: {perms}")
    else:
        print(f"\n  No AURA.md found in {os.getcwd()} (run: aura init)")

    print()
    return 0


def cmd_models(args) -> int:
    """List available models with routing roles."""
    from aura.core.router import ROUTING_TABLE, VALID_TIERS

    print("\nAura Model Routing\n")
    print(f"  {'Category':<16} {'local':<22} {'balanced':<28} {'max'}")
    print(f"  {'─' * 16} {'─' * 22} {'─' * 28} {'─' * 28}")

    for category, tiers in ROUTING_TABLE.items():
        local = tiers.get("local", "-")
        balanced = tiers.get("balanced", "-")
        max_ = tiers.get("max", "-")
        print(f"  {category:<16} {local:<22} {balanced:<28} {max_}")

    # Show which models are actually available
    print()
    try:
        import ollama
        models = ollama.list()
        available = {m.get("name", m.get("model", "")) for m in models.get("models", [])}
        print(f"  {len(available)} models available locally")
    except Exception:
        print("  (Could not check available models — is Ollama running?)")

    print()
    return 0


def cmd_commit(args) -> int:
    """Smart commit with AI-generated message."""
    from aura.tools.git_tool import GitTool
    from aura import ApprenticeAgent

    git = GitTool()
    cwd = os.getcwd()

    # Check for changes
    status = git.status(cwd)
    if not status.get("success"):
        print("Not in a git repository or git error.")
        return 1

    diff_result = git.diff(cwd)
    diff_text = diff_result.get("diff", "")

    if not diff_text and not status.get("dirty_count", 0):
        print("No changes to commit.")
        return 0

    # Stage all if --all flag
    if getattr(args, 'all', False):
        git.add(cwd, files=".")

    # Get diff of staged changes
    try:
        staged_proc = subprocess.run(
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
        print("No staged changes. Use 'git add' first or pass --all.")
        return 1

    # Generate commit message
    print("Generating commit message...")
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
            print("Error: LLM returned empty commit message.")
            return 1

    except Exception as e:
        print(f"Error generating message: {e}")
        return 1

    print(f"\n  Commit message: {message}\n")
    try:
        confirm = input("  Commit with this message? [y/n/edit]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 1

    if confirm == "edit" or confirm == "e":
        try:
            message = input("  Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    elif confirm not in ("y", "yes", ""):
        print("  Cancelled.")
        return 0

    result = git.commit(cwd, message=message)
    if result.get("success"):
        print(f"  Committed: {message}")
        return 0
    else:
        print(f"  Commit failed: {result.get('error', 'unknown error')}")
        return 1


def cmd_cost(args) -> int:
    """Show session cost breakdown from activity log."""
    try:
        from aura.cli.activity_log import ActivityLog
        log = ActivityLog()
        stats = log.get_stats()
    except (ImportError, OSError) as e:
        print(f"\nCould not read activity log: {e}")
        print("Cost data is tracked during interactive sessions.\n")
        return 1

    print("\nAura Cost Summary\n")
    total_cost = stats.get("total_cost_usd", 0.0)
    total_sessions = stats.get("total_sessions", 0)
    total_messages = stats.get("total_messages", 0)
    total_tokens = stats.get("total_tokens", 0)

    print(f"  Total cost:     ${total_cost:.4f}")
    print(f"  Sessions:       {total_sessions}")
    print(f"  Messages:       {total_messages}")
    print(f"  Tokens:         {total_tokens:,}")

    # Per-model breakdown if available
    model_costs = stats.get("model_costs", {})
    if model_costs:
        print(f"\n  By model:")
        for model, cost in sorted(model_costs.items(), key=lambda x: x[1], reverse=True):
            print(f"    {model:30s} ${cost:.4f}")

    print()
    return 0


def cmd_ide_setup(args) -> int:
    """Generate VS Code tasks.json and print MCP config snippet."""
    cwd = os.getcwd()
    vscode_dir = os.path.join(cwd, ".vscode")
    tasks_path = os.path.join(vscode_dir, "tasks.json")

    # Aura tasks for VS Code
    aura_tasks = [
        {
            "label": "Aura: Chat",
            "type": "shell",
            "command": "python -m main --chat",
            "presentation": {"reveal": "always", "panel": "dedicated"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Run Prompt",
            "type": "shell",
            "command": "python -m main -p \"${input:auraPrompt}\"",
            "presentation": {"reveal": "always"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Init Project",
            "type": "shell",
            "command": "python -m main init",
            "presentation": {"reveal": "always"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Smart Commit",
            "type": "shell",
            "command": "python -m main commit --all",
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

    print(f"\n  Created {tasks_path}")
    print(f"    - Aura: Chat (interactive mode)")
    print(f"    - Aura: Run Prompt (one-shot)")
    print(f"    - Aura: Init Project")
    print(f"    - Aura: Smart Commit")

    # Print MCP config snippet
    print(f"\n  MCP Server config for VS Code settings.json:\n")
    mcp_config = {
        "mcp.servers": {
            "aura": {
                "command": "python",
                "args": ["-m", "aura.core.mcp_server"],
                "cwd": cwd,
            }
        }
    }
    print(f"    {json.dumps(mcp_config, indent=4)}")
    print()
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


def cmd_setup(args) -> int:
    """Interactive setup wizard for configuring Aura in a project."""
    cwd = os.getcwd()
    project_name = Path(cwd).name

    print(f"\n  Aura Setup Wizard for '{project_name}'\n")

    # Step 1: Detect project type
    print("  Step 1: Detecting project...")
    try:
        from aura.tools.code_search import CodeSearchTool
        searcher = CodeSearchTool()
        info = searcher.detect_project_type(cwd)
        project_type = info.get("project_type", "unknown")
        frameworks = info.get("frameworks", [])
        stack = info.get("stack", [])
        if project_type != "unknown":
            print(f"    Detected: {project_type} ({', '.join(stack)})")
        else:
            print("    Could not auto-detect project type.")
    except Exception:
        project_type = "unknown"
        frameworks = []
        stack = []
        print("    Could not auto-detect (code_search unavailable).")

    # Step 2: Choose tier
    print("\n  Step 2: Choose model tier")
    print("    fast     — Quick responses, lower cost")
    print("    balanced — Good balance of speed and quality (recommended)")
    print("    max      — Best quality, higher cost")
    tier = _prompt("    Tier", "balanced")
    if tier not in ("fast", "balanced", "max"):
        print(f"    Invalid tier '{tier}', using 'balanced'.")
        tier = "balanced"

    # Step 3: Model
    model = _prompt("\n  Step 3: Model (or 'auto' for smart routing)", "auto")

    # Step 4: Test command
    detected_test = _detect_test_cmd(cwd)
    default_test = detected_test or "pytest"
    test_cmd = _prompt(f"\n  Step 4: Test command", default_test)
    auto_test_str = _prompt("    Auto-run tests after edits? [y/n]", "y")
    auto_test = auto_test_str.lower() in ("y", "yes")

    # Step 5: API keys
    print("\n  Step 5: Checking API keys...")
    for key_name in ["OLLAMA_API_KEY", "BRAVE_API_KEY", "TAVILY_API_KEY"]:
        has = bool(os.environ.get(key_name))
        print(f"    {key_name}: {'found' if has else 'not set'}")

    # Step 6: Generate AURA.md
    print("\n  Step 6: Creating AURA.md...")
    aura_md_path = os.path.join(cwd, "AURA.md")

    if os.path.exists(aura_md_path):
        overwrite = _prompt("    AURA.md already exists. Overwrite? [y/n]", "n")
        if overwrite.lower() not in ("y", "yes"):
            print("    Kept existing AURA.md.")
            print(f"\n  Setup complete! Run 'aura' to start.\n")
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

    print(f"    Created {aura_md_path}")
    print(f"\n  Setup complete! Run 'aura' to start.\n")
    return 0


def _prompt(text: str, default: str) -> str:
    """Prompt with a default value shown in brackets."""
    try:
        value = input(f"{text} [{default}]: ").strip()
        return value if value else default
    except (EOFError, KeyboardInterrupt):
        return default
