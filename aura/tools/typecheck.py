"""Fast, changed-file-scoped type checking.

Complements aura/tools/auto_verify.py (which runs full-suite lint + typecheck
for the whole project). This module is optimized for VerificationStage — it
detects the available checker (tsc, mypy, pyright, ty) and runs it only on
files that changed this turn, so post-edit verification feels snappy.

Supported checkers, in detection order:
  - tsc (TypeScript)                 — marker: tsconfig.json
  - pyright                          — marker: pyrightconfig.json, or pyright on PATH + .py files
  - mypy                             — marker: [tool.mypy] in pyproject.toml, mypy.ini, or .mypy.ini
  - ty (Python)                      — marker: ty.toml or ty on PATH
  - (none)                           — no checker configured; returns success=True
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Diagnostic:
    """One typechecker diagnostic."""
    file: str
    line: int
    col: int
    severity: str  # "error" | "warning" | "info"
    code: str      # typechecker error code (e.g. "TS2322", "arg-type")
    message: str


@dataclass
class TypecheckResult:
    runner: str                                # "tsc" / "mypy" / "pyright" / "ty" / "none"
    success: bool
    duration_s: float
    diagnostics: list[Diagnostic] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    skipped_reason: str = ""                   # populated when runner="none"


# ── Detection ────────────────────────────────────────────────────────────

def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _pyproject_has_section(project_root: str, section: str) -> bool:
    p = Path(project_root) / "pyproject.toml"
    if not p.exists():
        return False
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        return f"[tool.{section}]" in text or f"[tool.{section}." in text
    except OSError:
        return False


def _file_exists(project_root: str, name: str) -> bool:
    return (Path(project_root) / name).is_file()


def detect_typechecker(project_root: str, changed_files: list[str]) -> str:
    """Return the best available typechecker for the given project.

    Preference order biases toward the fastest check that's already configured.
    Returns one of: "tsc", "pyright", "mypy", "ty", "none".
    """
    has_ts = any(f.endswith((".ts", ".tsx")) for f in changed_files)
    has_py = any(f.endswith(".py") for f in changed_files)

    # TypeScript first — most teams already run tsc in CI, and --noEmit is fast.
    if has_ts and _file_exists(project_root, "tsconfig.json"):
        if _has_tool("tsc") or _has_tool("npx"):
            return "tsc"

    # Python: pyright → mypy → ty. Pyright is faster than mypy when configured.
    if has_py:
        if _file_exists(project_root, "pyrightconfig.json") and _has_tool("pyright"):
            return "pyright"
        if (_pyproject_has_section(project_root, "mypy")
                or _file_exists(project_root, "mypy.ini")
                or _file_exists(project_root, ".mypy.ini")) and _has_tool("mypy"):
            return "mypy"
        if _file_exists(project_root, "ty.toml") and _has_tool("ty"):
            return "ty"

    return "none"


# ── Runner dispatchers ───────────────────────────────────────────────────

_TSC_RE = re.compile(
    r"^(?P<file>[^(]+?)\((?P<line>\d+),(?P<col>\d+)\):\s+"
    r"(?P<severity>error|warning)\s+(?P<code>TS\d+):\s+(?P<msg>.*)$"
)
_MYPY_RE = re.compile(
    r"^(?P<file>[^:]+?):(?P<line>\d+)(?::(?P<col>\d+))?:\s+"
    r"(?P<severity>error|warning|note):\s+(?P<msg>.+?)(?:\s+\[(?P<code>[\w-]+)\])?$"
)
_PYRIGHT_RE = re.compile(
    r"^\s*(?P<file>[^:]+?):(?P<line>\d+):(?P<col>\d+)\s*-\s*"
    r"(?P<severity>error|warning|information):\s+(?P<msg>.+?)\s*(?:\((?P<code>[\w-]+)\))?$"
)


def _parse_tsc(output: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for line in output.splitlines():
        m = _TSC_RE.match(line.strip())
        if m:
            diagnostics.append(Diagnostic(
                file=m.group("file").strip(),
                line=int(m.group("line")),
                col=int(m.group("col")),
                severity=m.group("severity"),
                code=m.group("code"),
                message=m.group("msg").strip(),
            ))
    return diagnostics


def _parse_mypy(output: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for line in output.splitlines():
        m = _MYPY_RE.match(line.strip())
        if m and m.group("severity") in ("error", "warning"):
            diagnostics.append(Diagnostic(
                file=m.group("file").strip(),
                line=int(m.group("line")),
                col=int(m.group("col") or 0),
                severity=m.group("severity"),
                code=m.group("code") or "",
                message=m.group("msg").strip(),
            ))
    return diagnostics


def _parse_pyright(output: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for line in output.splitlines():
        m = _PYRIGHT_RE.match(line.rstrip())
        if m and m.group("severity") in ("error", "warning"):
            diagnostics.append(Diagnostic(
                file=m.group("file").strip(),
                line=int(m.group("line")),
                col=int(m.group("col")),
                severity=m.group("severity"),
                code=m.group("code") or "",
                message=m.group("msg").strip(),
            ))
    return diagnostics


def _run(cmd: list[str], cwd: str, timeout: int) -> tuple[int, str, str, float]:
    """Run a subprocess, return (exit_code, stdout, stderr, duration_s)."""
    import time as _t
    start = _t.monotonic()
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout or "", r.stderr or "", _t.monotonic() - start
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s", _t.monotonic() - start
    except (FileNotFoundError, OSError) as e:
        return 127, "", str(e), _t.monotonic() - start


# ── Public API ───────────────────────────────────────────────────────────

def typecheck_changed_files(
    project_root: str,
    changed_files: list[str],
    timeout: int = 30,
    override_cmd: Optional[str] = None,
) -> TypecheckResult:
    """Run the project's typechecker against only the changed files.

    Returns TypecheckResult(runner="none", success=True) when no checker is
    configured — intentional: we don't punish projects without typechecking.

    *override_cmd* bypasses detection and runs the command as-is (split via shlex).
    The list of changed files is appended as additional arguments; use `{files}`
    in the override to place them at a custom position.
    """
    if not changed_files:
        return TypecheckResult(
            runner="none", success=True, duration_s=0.0,
            skipped_reason="no changed files",
        )

    # Normalize paths to be relative to project_root where possible.
    rel_files: list[str] = []
    for f in changed_files:
        try:
            rel = os.path.relpath(f, project_root)
            rel_files.append(rel)
        except ValueError:
            rel_files.append(f)

    if override_cmd and override_cmd.strip():
        import shlex as _shlex
        tokens = _shlex.split(override_cmd.strip())
        if "{files}" in tokens:
            idx = tokens.index("{files}")
            cmd = tokens[:idx] + rel_files + tokens[idx + 1:]
        else:
            cmd = tokens + rel_files
        code, out, err, dur = _run(cmd, project_root, timeout)
        return TypecheckResult(
            runner="custom", success=code == 0, duration_s=dur,
            diagnostics=[], stdout=out, stderr=err,
        )

    runner = detect_typechecker(project_root, rel_files)

    if runner == "none":
        return TypecheckResult(
            runner="none", success=True, duration_s=0.0,
            skipped_reason="no typechecker configured/detected",
        )

    if runner == "tsc":
        # tsc --noEmit runs against the whole project; filtering per-file is
        # fragile because TS resolves module graphs. Trade-off: always whole-project.
        # Still gated on changed_files containing .ts/.tsx so we don't run on
        # pure-Python changes.
        tsc_bin = "tsc" if _has_tool("tsc") else "npx"
        cmd = ["tsc", "--noEmit"] if tsc_bin == "tsc" else ["npx", "tsc", "--noEmit"]
        code, out, err, dur = _run(cmd, project_root, timeout)
        return TypecheckResult(
            runner="tsc", success=code == 0, duration_s=dur,
            diagnostics=_parse_tsc(out + "\n" + err),
            stdout=out, stderr=err,
        )

    if runner == "pyright":
        py_files = [f for f in rel_files if f.endswith(".py")]
        cmd = ["pyright", "--outputjson=false", *py_files]
        code, out, err, dur = _run(cmd, project_root, timeout)
        return TypecheckResult(
            runner="pyright", success=code == 0, duration_s=dur,
            diagnostics=_parse_pyright(out + "\n" + err),
            stdout=out, stderr=err,
        )

    if runner == "mypy":
        py_files = [f for f in rel_files if f.endswith(".py")]
        cmd = ["mypy", "--no-color-output", "--show-error-codes", *py_files]
        code, out, err, dur = _run(cmd, project_root, timeout)
        return TypecheckResult(
            runner="mypy", success=code == 0, duration_s=dur,
            diagnostics=_parse_mypy(out + "\n" + err),
            stdout=out, stderr=err,
        )

    if runner == "ty":
        py_files = [f for f in rel_files if f.endswith(".py")]
        cmd = ["ty", "check", *py_files]
        code, out, err, dur = _run(cmd, project_root, timeout)
        # ty output format isn't stable yet; keep raw stdout/stderr and leave
        # diagnostics empty until we have a parser to rely on.
        return TypecheckResult(
            runner="ty", success=code == 0, duration_s=dur,
            diagnostics=[], stdout=out, stderr=err,
        )

    return TypecheckResult(
        runner="none", success=True, duration_s=0.0,
        skipped_reason=f"unknown runner: {runner}",
    )
