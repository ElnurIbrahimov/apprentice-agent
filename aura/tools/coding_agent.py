"""Delegate coding tasks to external agent CLIs.

Aura's sub-agents are in-process; this tool is the opposite — it hands a
task off to a peer agent CLI installed on the host (Claude Code, Codex,
Aider, OpenCode, Goose). Useful for:

- Heavy refactors where Aura's own agentic loop would eat too much context
- Cross-checking a plan with a different model family
- Long-running builds where you don't want to block Aura's chat loop

Pattern borrowed from OpenClaw's `coding-agent` SKILL.md (MIT, Nous
Research / OpenClaw). Windows-first: no PTY — every supported CLI has a
non-interactive mode we can invoke via `subprocess`.

Sandbox-tier aware: refuses to run when sandbox tier is READ_ONLY
(spawning another agent is by definition a write action).

Supported agents (must be installed and on PATH):
    claude     → `claude --permission-mode bypassPermissions --print`
    codex      → `codex exec <prompt>`
    aider      → `aider --message <prompt> --yes-always --no-pretty`
    opencode   → `opencode run <prompt>`
    goose      → `goose run --text <prompt>`
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional

logger = logging.getLogger(__name__)


AGENT_NAMES = ("claude", "codex", "aider", "opencode", "goose")


@dataclass
class CodingAgentResult:
    agent: str
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    cwd: str
    command: List[str] = field(default_factory=list)

    def summary(self, max_chars: int = 4000) -> str:
        parts = [
            f"[{self.agent}] exit={self.exit_code} duration={self.duration_s:.1f}s",
            f"cwd: {self.cwd}",
        ]
        if self.stdout:
            parts.append(f"\n--- stdout ({len(self.stdout)} chars) ---\n{self.stdout[:max_chars]}")
            if len(self.stdout) > max_chars:
                parts.append(f"... [+{len(self.stdout) - max_chars} chars truncated]")
        if self.stderr and self.stderr.strip():
            parts.append(f"\n--- stderr ---\n{self.stderr[:1000]}")
        return "\n".join(parts)


class CodingAgentTool:
    """Tool for delegating coding tasks to external agent CLIs."""

    name: ClassVar[str] = "coding_agent"
    description: ClassVar[str] = (
        "Delegate a coding task to an external agent CLI (claude, codex, aider, "
        "opencode, goose). Use for heavy refactors, cross-model review, or long "
        "builds. NOT for simple edits — use edit_file for those."
    )

    @classmethod
    def available_agents(cls) -> Dict[str, bool]:
        """Report which agents are currently on PATH.

        On Windows, shutil.which honors PATHEXT so shims like claude.cmd resolve.
        """
        return {name: shutil.which(name) is not None for name in AGENT_NAMES}

    @classmethod
    def _build_command(
        cls, agent: str, prompt: str, extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        extra = list(extra_args or [])
        if agent == "claude":
            return [
                "claude",
                "--permission-mode", "bypassPermissions",
                "--print",
                *extra,
                prompt,
            ]
        if agent == "codex":
            return ["codex", "exec", *extra, prompt]
        if agent == "aider":
            return [
                "aider",
                "--message", prompt,
                "--yes-always",
                "--no-pretty",
                "--no-stream",
                *extra,
            ]
        if agent == "opencode":
            return ["opencode", "run", *extra, prompt]
        if agent == "goose":
            return ["goose", "run", "--text", prompt, *extra]
        raise ValueError(f"Unknown agent: {agent!r}. Known: {', '.join(AGENT_NAMES)}")

    @classmethod
    def delegate(
        cls,
        agent: str,
        prompt: str,
        *,
        cwd: Optional[str] = None,
        timeout: int = 600,
        extra_args: Optional[List[str]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> CodingAgentResult:
        """Run the agent on the given prompt and capture its output.

        Args:
            agent: one of AGENT_NAMES
            prompt: the task description
            cwd: working directory (defaults to current)
            timeout: seconds before hard kill
            extra_args: extra CLI flags inserted before the prompt
            env_overrides: env vars to set for the subprocess
        """
        agent = agent.lower().strip()
        if agent not in AGENT_NAMES:
            raise ValueError(f"Unknown agent: {agent!r}. Known: {', '.join(AGENT_NAMES)}")

        # Sandbox-tier gate: READ_ONLY bars external agent spawns entirely.
        try:
            from aura.core.permissions import SandboxTier, get_sandbox_tier
            if get_sandbox_tier() == SandboxTier.READ_ONLY:
                return CodingAgentResult(
                    agent=agent,
                    success=False,
                    stdout="",
                    stderr="BLOCKED by READ_ONLY sandbox tier",
                    exit_code=-1,
                    duration_s=0.0,
                    cwd=cwd or os.getcwd(),
                )
        except Exception:
            pass

        if shutil.which(agent) is None:
            return CodingAgentResult(
                agent=agent,
                success=False,
                stdout="",
                stderr=f"agent {agent!r} not found on PATH",
                exit_code=127,
                duration_s=0.0,
                cwd=cwd or os.getcwd(),
            )

        cmd = cls._build_command(agent, prompt, extra_args=extra_args)
        run_cwd = cwd or os.getcwd()
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        logger.info("[coding_agent] dispatching %s in %s: %s", agent, run_cwd, shlex.join(cmd))
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=run_cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
            duration = time.time() - start
            return CodingAgentResult(
                agent=agent,
                success=proc.returncode == 0,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
                duration_s=duration,
                cwd=run_cwd,
                command=cmd,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start
            return CodingAgentResult(
                agent=agent,
                success=False,
                stdout=(exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=f"TIMEOUT after {timeout}s",
                exit_code=-9,
                duration_s=duration,
                cwd=run_cwd,
                command=cmd,
            )
        except FileNotFoundError:
            duration = time.time() - start
            return CodingAgentResult(
                agent=agent,
                success=False,
                stdout="",
                stderr=f"agent {agent!r} not found (PATH issue)",
                exit_code=127,
                duration_s=duration,
                cwd=run_cwd,
                command=cmd,
            )
        except Exception as exc:
            duration = time.time() - start
            logger.exception("[coding_agent] unexpected failure")
            return CodingAgentResult(
                agent=agent,
                success=False,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                exit_code=-1,
                duration_s=duration,
                cwd=run_cwd,
                command=cmd,
            )

    @classmethod
    def tool_schema(cls) -> dict:
        """OpenAI-style function schema for LLM tool-calling."""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": list(AGENT_NAMES),
                            "description": "Which external CLI to invoke.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The task description to pass to the agent.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory. Defaults to current.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Hard kill timeout in seconds (default 600).",
                        },
                    },
                    "required": ["agent", "prompt"],
                },
            },
        }

    @classmethod
    def run(cls, agent: str, prompt: str, cwd: Optional[str] = None,
            timeout: int = 600) -> str:
        """LLM-facing entry point — returns a summarized string."""
        result = cls.delegate(agent=agent, prompt=prompt, cwd=cwd, timeout=timeout)
        return result.summary()
