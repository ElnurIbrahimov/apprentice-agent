"""Tool dispatching for AgenticLoop."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import ClassVar

from aura.pools import tool_pool as _tool_pool_fn

from .agentic_loop_support import _truncate

logger = logging.getLogger(__name__)
MAX_TOOL_OUTPUT_CHARS = 15000


def get_tool_pool():
    """Lazy accessor for the shared tool pool."""
    return _tool_pool_fn()


class ToolExecutor:
    """Executes tool calls by dispatching to existing Aura tool classes."""

    _TOOL_ALIASES: ClassVar[dict[str, str]] = {
        "filesystem": "list_dir",
        "code_search": "grep",
        "code_edit": "edit_file",
        "shell_executor": "shell",
        "web_search": "search_web",
        "tavily_search": "search_web",
        "brave_search": "search_web",
        "ls": "list_dir",
        "find": "glob",
        "cat": "read_file",
    }

    _COMMAND_ALTERNATIVES: ClassVar[dict[str, list[str]]] = {
        "python": ["python3", "py"],
        "python3": ["python", "py"],
        "py": ["python", "python3"],
        "pip": ["pip3", "python -m pip"],
        "pip3": ["pip", "python3 -m pip"],
        "npx": ["npm exec", "pnpm exec"],
        "npm": ["pnpm", "yarn"],
        "pnpm": ["npm", "yarn"],
        "yarn": ["npm", "pnpm"],
        "node": ["nodejs"],
        "nodejs": ["node"],
        "make": ["cmake", "nmake"],
        "gcc": ["cc", "clang"],
        "g++": ["c++", "clang++"],
        "curl": ["wget", "Invoke-WebRequest"],
        "wget": ["curl"],
        "cat": ["type", "Get-Content"],
        "ls": ["dir", "Get-ChildItem"],
        "rm": ["del", "Remove-Item"],
        "cp": ["copy", "Copy-Item"],
        "mv": ["move", "Move-Item"],
        "grep": ["findstr", "Select-String"],
    }

    def __init__(self, project_root: str, sub_agent_mgr=None, permissions=None, mcp_client=None):
        self.project_root = project_root
        self.sub_agent_mgr = sub_agent_mgr
        self.permissions = permissions
        self._mcp_client = mcp_client
        self._init_lock = threading.Lock()
        self._fs = None
        self._code_edit = None
        self._search = None
        self._shell = None
        self._git = None

    @property
    def fs(self):
        if self._fs is None:
            with self._init_lock:
                if self._fs is None:
                    from aura.tools.filesystem import FileSystemTool

                    self._fs = FileSystemTool(sandbox_enabled=False)
        return self._fs

    @property
    def code_edit(self):
        if self._code_edit is None:
            with self._init_lock:
                if self._code_edit is None:
                    from aura.tools.code_edit import CodeEditTool

                    self._code_edit = CodeEditTool()
        return self._code_edit

    @property
    def search(self):
        if self._search is None:
            with self._init_lock:
                if self._search is None:
                    from aura.tools.code_search import CodeSearchTool

                    self._search = CodeSearchTool()
        return self._search

    @property
    def shell(self):
        if self._shell is None:
            with self._init_lock:
                if self._shell is None:
                    from aura.tools.shell_executor import ShellExecutorTool

                    self._shell = ShellExecutorTool()
        return self._shell

    @property
    def git(self):
        if self._git is None:
            with self._init_lock:
                if self._git is None:
                    from aura.tools.git_tool import GitTool

                    self._git = GitTool()
        return self._git

    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths against project root with containment checks."""
        if os.path.isabs(path):
            resolved = os.path.realpath(path)
        else:
            resolved = os.path.realpath(os.path.join(self.project_root, path))

        if not os.path.exists(resolved) and not os.path.isabs(path):
            project_name = os.path.basename(self.project_root)
            if path.startswith(project_name + "/") or path.startswith(project_name + "\\"):
                stripped = path[len(project_name) + 1 :]
                alt = os.path.realpath(os.path.join(self.project_root, stripped))
                if os.path.exists(alt):
                    resolved = alt

        allowed_roots = [os.path.realpath(self.project_root)]
        home = os.path.realpath(os.path.expanduser("~"))
        if home:
            candidate = os.path.join(home, ".aura")
            if os.path.isdir(candidate):
                allowed_roots.append(os.path.realpath(candidate))

        sensitive_dirs = {
            ".ssh",
            ".gnupg",
            ".aws",
            ".azure",
            ".kube",
            ".docker",
            ".config/gcloud",
            "AppData/Roaming/1Password",
        }
        for sensitive in sensitive_dirs:
            sensitive_path = os.path.realpath(os.path.join(home, sensitive))
            if resolved.startswith(sensitive_path + os.sep) or resolved == sensitive_path:
                raise PermissionError(f"Access to sensitive directory blocked: {path}")

        for root in allowed_roots:
            if resolved.startswith(root + os.sep) or resolved == root:
                return resolved
        raise PermissionError(f"Path outside allowed directories: {path}")

    def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool call and serialize the result for the LLM."""
        try:
            result = self._dispatch(tool_name, args)
            # expand_observation intentionally returns full content — skip truncation
            # so the masker's round-trip actually works for >15k items.
            if tool_name == "expand_observation" and isinstance(result, str):
                return result
            if isinstance(result, dict):
                return _truncate(json.dumps(result, indent=2, default=str))
            return _truncate(str(result))
        except Exception as exc:
            logger.error("[ToolExecutor] %s failed: %s", tool_name, exc)
            return json.dumps({"error": str(exc)})

    def _dispatch(self, tool_name: str, args: dict) -> dict:
        tool_name = self._TOOL_ALIASES.get(tool_name, tool_name)

        if "action" in args and len(args) == 1:
            args = self._parse_action_to_args(tool_name, args["action"])
            if "error" in args:
                return args

        if tool_name == "read_file":
            return self._read_file(args)
        if tool_name == "grep":
            return self.search.grep(
                pattern=args["pattern"],
                path=self._resolve_path(args.get("path", ".")),
                file_type=args.get("file_type"),
                case_insensitive=args.get("case_insensitive", False),
                context_lines=2,
                max_results=50,
            )
        if tool_name == "glob":
            return self.search.glob(
                pattern=args["pattern"],
                path=self._resolve_path(args.get("path", ".")),
            )
        if tool_name == "list_dir":
            return self._list_dir(self._resolve_path(args.get("path", ".")))
        if tool_name == "edit_file":
            return self._edit_file(args)
        if tool_name == "write_file":
            return self._write_file(args)
        if tool_name == "shell":
            cwd = self._resolve_path(args.get("cwd", "."))
            result = self.shell.run_sandboxed(
                command=args["command"],
                cwd=cwd,
                timeout=min(args.get("timeout", 60), 300),
            )
            return self._enrich_shell_error(result, args.get("command", ""))
        if tool_name == "git":
            return self._git_dispatch(args)
        if tool_name == "search_web":
            return self._web_search(args)
        if tool_name == "project_structure":
            return self.search.project_structure(
                path=self._resolve_path(args.get("path", ".")),
                max_depth=args.get("max_depth", 3),
            )
        if tool_name == "fetch_url":
            return self._fetch_url(args)
        if tool_name == "create_directory":
            path = self._resolve_path(args.get("path", ""))
            os.makedirs(path, exist_ok=True)
            return {"success": True, "path": path}
        if tool_name == "move_file":
            src = self._resolve_path(args.get("source", ""))
            dst = self._resolve_path(args.get("destination", ""))
            import shutil

            shutil.move(src, dst)
            return {"success": True, "source": src, "destination": dst}
        if tool_name == "multi_edit":
            return self.code_edit.multi_edit(
                path=self._resolve_path(args["path"]),
                edits=args.get("edits", []),
            )
        if tool_name == "run_tests":
            return self._run_tests(args)
        if tool_name == "spawn_agent":
            if self.sub_agent_mgr is None:
                return {"error": "Sub-agents not available"}
            return self.sub_agent_mgr.spawn(task=args.get("task", ""), role=args.get("role", "reader"))
        if tool_name == "expand_observation":
            from aura.memory.observation_masker import expand_observation as _expand
            obs_id = args.get("obs_id") or args.get("id") or ""
            if not obs_id:
                return {"error": "expand_observation requires obs_id"}
            full = _expand(str(obs_id))
            if full is None:
                return {"error": f"No observation found for ID: {obs_id}"}
            return full
        if tool_name.startswith("mcp_"):
            if self._mcp_client:
                return {"result": self._mcp_client.call_tool(tool_name, args)}
            return {"error": f"MCP client not available for: {tool_name}"}
        return {"error": f"Unknown tool: {tool_name}"}

    def _parse_action_to_args(self, tool_name: str, action: str) -> dict:
        """Convert a freeform ``action`` string into structured args."""
        action = action.strip()
        if tool_name == "read_file":
            for prefix in ("read ", "cat ", "open "):
                if action.lower().startswith(prefix):
                    action = action[len(prefix) :].strip()
                    break
            return {"path": action}
        if tool_name == "list_dir":
            for prefix in ("list ", "ls ", "dir ", "list_directory "):
                if action.lower().startswith(prefix):
                    action = action[len(prefix) :].strip()
                    break
            return {"path": action or "."}
        if tool_name == "grep":
            parts = action.split(maxsplit=1)
            if len(parts) >= 2:
                return {"pattern": parts[0], "path": parts[1]}
            return {"pattern": action, "path": "."}
        if tool_name == "glob":
            return {"pattern": action or "*"}
        if tool_name == "shell":
            return {"command": action}
        if tool_name == "search_web":
            return {"query": action}
        if tool_name in {"write_file", "edit_file"}:
            try:
                parsed = json.loads(action)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:
                logger.debug("[ToolExecutor] %s JSON parse failed: %s", tool_name, exc)
            return {"error": f"Could not parse {tool_name} arguments"}
        if tool_name == "git":
            return {"action": action}
        if tool_name == "project_structure":
            return {"path": action or "."}
        return {"action": action}

    def _list_dir(self, path: str) -> dict:
        try:
            entries = []
            for entry in sorted(os.listdir(path)):
                full = os.path.join(path, entry)
                if os.path.isdir(full):
                    entries.append(f"  {entry}/")
                    continue
                try:
                    size = os.path.getsize(full)
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size // 1024}KB"
                    else:
                        size_str = f"{size // (1024 * 1024)}MB"
                    entries.append(f"  {entry} ({size_str})")
                except OSError:
                    entries.append(f"  {entry}")
            return {"success": True, "path": path, "count": len(entries), "entries": "\n".join(entries)}
        except FileNotFoundError:
            return {"error": f"Directory not found: {path}"}
        except PermissionError:
            return {"error": f"Permission denied: {path}"}

    def _read_file(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
                lines = file_obj.readlines()
        except FileNotFoundError:
            return self._read_file_not_found(path, args.get("path", ""))
        except PermissionError:
            return {"error": f"Permission denied: {path}"}

        offset = args.get("offset", 0)
        limit = args.get("limit", 0)
        selected = lines[offset : offset + limit] if limit > 0 else lines[offset:]
        numbered = [f"{index:>5}\t{line.rstrip()}" for index, line in enumerate(selected, start=offset + 1)]
        return {
            "success": True,
            "path": path,
            "total_lines": len(lines),
            "showing": f"{offset + 1}-{offset + len(selected)}",
            "content": "\n".join(numbered),
        }

    def _read_file_not_found(self, resolved_path: str, original_path: str) -> dict:
        basename = os.path.basename(original_path or resolved_path)
        if not basename:
            return {"error": f"File not found: {resolved_path}"}

        suggestions = []
        try:
            result = self.search.glob(pattern=f"**/{basename}", path=self.project_root, max_results=5)
            if result.get("success") and result.get("files"):
                suggestions = [item["path"] for item in result["files"][:5]]
        except Exception as exc:
            logger.debug("[ToolExecutor] File suggestion glob failed: %s", exc)

        if not suggestions:
            stem = os.path.splitext(basename)[0]
            if len(stem) >= 3:
                try:
                    result = self.search.glob(pattern=f"**/{stem}*", path=self.project_root, max_results=5)
                    if result.get("success") and result.get("files"):
                        suggestions = [item["path"] for item in result["files"][:5]]
                except Exception as exc:
                    logger.debug("[ToolExecutor] Partial file suggestion failed: %s", exc)

        if suggestions:
            return {"error": f"File not found: {resolved_path}", "did_you_mean": suggestions}
        return {"error": f"File not found: {resolved_path}"}

    def _enrich_shell_error(self, result: dict, command: str) -> dict:
        if result.get("success", False):
            return result

        stderr = result.get("stderr", "") or result.get("error", "") or ""
        patterns = [
            "command not found",
            "not recognized as an internal or external command",
            "is not recognized as",
            "No such file or directory",
            "The term",
        ]
        if not any(pattern.lower() in stderr.lower() for pattern in patterns):
            return result

        cmd_name = os.path.basename(command.strip().split()[0]) if command.strip() else ""
        alternatives = self._COMMAND_ALTERNATIVES.get(cmd_name, [])
        if alternatives:
            result["suggestion"] = f"'{cmd_name}' not found. Try: {', '.join(alternatives)}"
        return result

    def _edit_file(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        checkpoint_mgr = getattr(self, "_checkpoint_mgr", None)
        if checkpoint_mgr:
            try:
                checkpoint_mgr.snapshot(path, label=f"before edit: {Path(path).name}")
            except Exception as exc:
                logger.debug("[ToolExecutor] Edit checkpoint failed: %s", exc)

        preview = self.code_edit.edit(
            path=path,
            old_string=args["old_string"],
            new_string=args["new_string"],
            dry_run=True,
        )
        if not preview.get("success"):
            return preview

        from .diff_display import show_diff

        try:
            show_diff(path, args["old_string"], args["new_string"])
        except Exception as exc:
            logger.debug("[ToolExecutor] Diff display failed: %s", exc)

        if not getattr(self, "_trust_all_edits", False):
            perm_mode = getattr(self.permissions, "mode", "auto_edit") if self.permissions else "auto_edit"
            has_confirm_callback = getattr(self.permissions, "has_confirm_callback", False)
            if perm_mode == "careful" and not has_confirm_callback:
                try:
                    import sys

                    sys.stderr.write("  Apply this edit? [y/n/all]: ")
                    sys.stderr.flush()
                    choice = input().strip().lower()
                    if choice == "all":
                        self._trust_all_edits = True
                    elif choice not in ("y", "yes", ""):
                        return {"success": False, "skipped": True, "message": "Edit rejected by user"}
                except (EOFError, KeyboardInterrupt):
                    return {"success": False, "skipped": True, "message": "Edit cancelled"}

        result = self.code_edit.edit(
            path=path,
            old_string=args["old_string"],
            new_string=args["new_string"],
        )
        if result.get("success"):
            self._maybe_broadcast_artifact(path)
        return result

    def _write_file(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        checkpoint_mgr = getattr(self, "_checkpoint_mgr", None)
        if checkpoint_mgr and os.path.exists(path):
            try:
                checkpoint_mgr.snapshot(path, label=f"before write: {Path(path).name}")
            except Exception as exc:
                logger.debug("[ToolExecutor] Write checkpoint failed: %s", exc)

        result = self.fs.write_file(path=path, content=args["content"], overwrite=True)
        if result.get("success"):
            self._maybe_broadcast_artifact(path, content=args.get("content"))
        return result

    def _maybe_broadcast_artifact(self, path: str, content: str | None = None) -> None:
        try:
            from api.routes.artifacts import broadcast_artifact, is_previewable

            if not is_previewable(path):
                return
            if content is None:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
                        content = file_obj.read()
                except Exception as exc:
                    logger.debug("[ToolExecutor] Artifact file read failed: %s", exc)
                    return
            if content:
                try:
                    filename = os.path.relpath(path, self.project_root)
                except Exception as exc:
                    logger.debug("[ToolExecutor] relpath failed, using basename: %s", exc)
                    filename = Path(path).name
                broadcast_artifact(filename.replace("\\", "/"), content)
        except ImportError:
            return
        except Exception as exc:
            logger.debug("[ToolExecutor] Artifact broadcast failed: %s", exc)

    def _git_dispatch(self, args: dict) -> dict:
        action = args.get("action", "status")
        repo = self.project_root
        if action == "status":
            return self.git.status(repo)
        if action == "diff":
            return self.git.diff(repo, file=args.get("file"))
        if action == "log":
            return self.git.log(repo, count=args.get("count", 5))
        if action == "branch":
            return self.git.branch(repo)
        if action == "add":
            return self.git.add(repo, files=args.get("files", "."))
        if action == "commit":
            return self.git.commit(repo, message=args.get("message", ""))
        if action == "push":
            return self.git.push(repo)
        if action == "pull":
            return self.git.pull(repo)
        return {"error": f"Unknown git action: {action}"}

    def _web_search(self, args: dict) -> dict:
        from aura.tools.search_fallback import web_search_with_fallback

        return web_search_with_fallback(query=args["query"], max_results=args.get("max_results", 8))

    def _fetch_url(self, args: dict) -> dict:
        url = args.get("url", "")
        if not url:
            return {"error": "No URL provided"}

        try:
            from bs4 import BeautifulSoup

            from aura.security.ssrf_guard import safe_request
        except ImportError as exc:
            return {"error": f"Missing URL fetch dependency: {exc}"}

        try:
            response = safe_request(
                url,
                headers={"User-Agent": "Aura-Dev-Agent/1.0"},
                timeout=15,
            )
            response.raise_for_status()
            content = response.text
        except Exception as exc:
            return {"error": f"Failed to fetch {url}: {exc}"}

        try:
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except Exception:
            text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n... [truncated]"
        return {"success": True, "url": url, "length": len(text), "content": text}

    def _run_tests(self, args: dict) -> dict:
        target = args.get("target", "")
        cwd = self.project_root

        try:
            from aura.tools.auto_verify import AutoVerifyTool

            result = AutoVerifyTool().run(project_root=cwd, target=target)
            if result and "error" not in result:
                return result
        except Exception as exc:
            logger.debug("[ToolExecutor] auto_verify unavailable: %s", exc)

        cmd = None
        if (
            os.path.exists(os.path.join(cwd, "pytest.ini"))
            or os.path.exists(os.path.join(cwd, "pyproject.toml"))
            or os.path.exists(os.path.join(cwd, "setup.py"))
        ):
            cmd = f"python -m pytest {target} -x -q --tb=short" if target else "python -m pytest -x -q --tb=short"
        elif os.path.exists(os.path.join(cwd, "package.json")):
            try:
                with open(os.path.join(cwd, "package.json"), "r", encoding="utf-8") as file_obj:
                    package = json.load(file_obj)
                deps = {**package.get("devDependencies", {}), **package.get("dependencies", {})}
                if "vitest" in deps:
                    cmd = f"npx vitest run {target}" if target else "npx vitest run"
                else:
                    cmd = f"npx jest {target}" if target else "npx jest"
            except Exception as exc:
                logger.debug("[ToolExecutor] package.json test detection failed: %s", exc)
                cmd = f"npm test -- {target}" if target else "npm test"
        elif os.path.exists(os.path.join(cwd, "Cargo.toml")):
            cmd = f"cargo test {target}" if target else "cargo test"
        elif os.path.exists(os.path.join(cwd, "go.mod")):
            cmd = f"go test {target}" if target else "go test ./..."

        if not cmd:
            return {"error": "Could not detect test framework. Use shell tool to run tests manually."}
        return self.shell.run_sandboxed(command=cmd, cwd=cwd, timeout=120)
