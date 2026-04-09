# aura/cli/git_tools.py
"""Git power tools — /pr, /branch, /stash, /blame, auto-commit."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def git_run(args: List[str], cwd: Optional[str] = None, timeout: int = 30) -> Dict:
    """Run a git command and return result dict."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=cwd or ".",
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Git command timed out"}
    except FileNotFoundError:
        return {"success": False, "error": "git not found on PATH"}
    except (OSError, NotADirectoryError) as e:
        return {"success": False, "error": f"Git command failed: {e}"}


def get_current_branch() -> str:
    """Get current git branch name."""
    result = git_run(["branch", "--show-current"])
    return result.get("stdout", "unknown") if result["success"] else "unknown"


def get_staged_diff() -> str:
    """Get the staged diff for commit message generation."""
    result = git_run(["diff", "--cached", "--stat"])
    if not result["success"] or not result["stdout"]:
        result = git_run(["diff", "--stat"])
    return result.get("stdout", "")


def get_recent_log(n: int = 5) -> str:
    """Get recent commit log."""
    result = git_run(["log", "--oneline", f"-{n}"])
    return result.get("stdout", "")


def create_branch(name: str) -> Dict:
    """Create and switch to a new branch."""
    safe_name = re.sub(r'[^\w\-/.]', '-', name)[:60]
    result = git_run(["checkout", "-b", safe_name])
    return {
        "success": result["success"],
        "branch": safe_name,
        "message": result.get("stdout", result.get("stderr", "")),
    }


def smart_stash(description: str = "") -> Dict:
    """Stash changes with a descriptive message."""
    args = ["stash", "push"]
    if description:
        args.extend(["-m", description[:100]])
    return git_run(args)


def create_pr(title: str, body: str, base: str = "main") -> Dict:
    """Create a pull request using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "pr", "create", "--title", title[:100], "--body", body[:2000], "--base", base],
            capture_output=True, text=True, timeout=30,
        )
        return {
            "success": result.returncode == 0,
            "url": result.stdout.strip(),
            "error": result.stderr.strip() if result.returncode != 0 else "",
        }
    except FileNotFoundError:
        return {"success": False, "error": "gh CLI not installed. Install: https://cli.github.com/"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "PR creation timed out"}


def get_blame(file_path: str, line_num: int) -> Dict:
    """Get git blame for a specific line."""
    result = git_run(["blame", "-L", f"{line_num},{line_num}", "--porcelain", file_path])
    if not result["success"]:
        return result

    # Parse porcelain blame output
    lines = result["stdout"].splitlines()
    info = {}
    for line in lines:
        if line.startswith("author "):
            info["author"] = line[7:]
        elif line.startswith("author-time "):
            import time
            try:
                info["date"] = time.strftime("%Y-%m-%d", time.localtime(int(line[12:])))
            except (ValueError, OSError):
                info["date"] = "unknown"
        elif line.startswith("summary "):
            info["commit_message"] = line[8:]
        elif line.startswith("\t"):
            info["content"] = line[1:]

    info["success"] = True
    return info


def auto_commit_edit(file_path: str, description: str = "") -> Dict:
    """Auto-commit a single file edit with AI-ready message."""
    git_run(["add", file_path])
    msg = description[:72] if description else f"aura: edit {Path(file_path).name}"
    return git_run(["commit", "-m", msg])


# Prompt templates for AI-generated content
COMMIT_MSG_PROMPT = """Generate a concise git commit message for these changes:

{diff}

Rules:
- First line: type(scope): description (max 72 chars)
- Types: feat, fix, refactor, docs, test, chore
- Be specific about what changed and why
- No period at end of first line"""

PR_DESCRIPTION_PROMPT = """Generate a pull request title and description for these changes:

Branch: {branch}
Diff stats:
{diff}

Recent commits:
{log}

Respond in this format:
TITLE: <concise title, max 70 chars>
BODY:
## Summary
<2-3 bullet points>

## Changes
<list of key changes>"""
