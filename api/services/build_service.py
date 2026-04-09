"""Background build task service for phase-4 agent build mode."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.routes.artifacts import broadcast_event
from api.services.agent_service import agent_service
from aura.core.agentic_loop import AgenticLoop
from aura.core.permissions import PermissionManager

logger = logging.getLogger(__name__)


def _normalize_rel_path(path: str) -> str:
    normalized = path.replace("\\", "/").replace("./", "", 1).lstrip("/").strip()
    if not normalized:
        raise ValueError("Path cannot be empty")
    parts = normalized.split("/")
    if any((not part) or part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid path: {path}")
    return normalized


def _safe_result_has_error(result: Any) -> bool:
    if isinstance(result, dict):
        return "error" in result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except Exception:
            return '"error"' in result
        return isinstance(parsed, dict) and "error" in parsed
    return False


def _extract_json_array(raw: str) -> Optional[list]:
    text = (raw or "").strip()
    if not text:
        return None
    candidates = [text]
    first = text.find("[")
    last = text.rfind("]")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _parse_plan_response(raw: str) -> List[Dict[str, Any]]:
    parsed = _extract_json_array(raw)
    plan: List[Dict[str, Any]] = []
    if parsed:
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            try:
                normalized = _normalize_rel_path(path)
            except ValueError:
                continue
            purpose = item.get("purpose")
            priority = item.get("priority")
            plan.append({
                "path": normalized,
                "purpose": purpose.strip() if isinstance(purpose, str) and purpose.strip() else "Generated file",
                "priority": priority if isinstance(priority, int) else index + 1,
            })
    if plan:
        return _dedupe_and_sort_plan(plan)

    fallback: List[Dict[str, Any]] = []
    for match in re.finditer(r"(?:^|\n)\s*(?:[-*]\s*)?([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)\s*(?:[-:]\s*(.+))?", raw or ""):
        try:
            path = _normalize_rel_path(match.group(1))
        except ValueError:
            continue
        fallback.append({
            "path": path,
            "purpose": (match.group(2) or "Generated file").strip(),
            "priority": len(fallback) + 1,
        })
    return _dedupe_and_sort_plan(fallback)


def _dedupe_and_sort_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in sorted(plan, key=lambda value: (int(value.get("priority", 0)), value.get("path", ""))):
        path = item["path"]
        if path in seen:
            continue
        seen.add(path)
        deduped.append(item)
    for index, item in enumerate(deduped):
        item["priority"] = index + 1
    return deduped


def _default_plan(framework: str, existing_files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if existing_files:
        return _dedupe_and_sort_plan([
            {"path": _normalize_rel_path(item["path"]), "purpose": "Existing project file", "priority": index + 1}
            for index, item in enumerate(existing_files)
            if item.get("path")
        ])
    if framework == "react":
        return [
            {"path": "src/main.jsx", "purpose": "App bootstrap entry", "priority": 1},
            {"path": "src/App.jsx", "purpose": "Main React view", "priority": 2},
            {"path": "src/styles.css", "purpose": "Shared styling", "priority": 3},
        ]
    return [
        {"path": "index.html", "purpose": "Main HTML structure", "priority": 1},
        {"path": "styles/main.css", "purpose": "Core styling", "priority": 2},
        {"path": "scripts/main.js", "purpose": "Client-side interactivity", "priority": 3},
    ]


def _build_plan_prompt(description: str, framework: str, existing_files: List[Dict[str, str]]) -> str:
    existing = "\n".join(f"- {item['path']}" for item in existing_files if item.get("path")) or "None"
    return f"""Plan a multi-file web project.

Description:
{description}

Framework: {framework}
Existing files:
{existing}

Return JSON only as an array:
[
  {{"path": "index.html", "purpose": "Main page shell", "priority": 1}}
]

Rules:
- Use forward slashes for file paths
- Order by dependency
- Keep the plan concise
- Static projects: max 10 files
- React projects: max 15 files"""


def _build_execution_prompt(
    description: str,
    framework: str,
    plan: List[Dict[str, Any]],
    existing_files: List[Dict[str, str]],
) -> str:
    plan_lines = "\n".join(
        f"{item['priority']}. {item['path']} - {item['purpose']}"
        for item in plan
    )
    existing = "\n".join(f"- {item['path']}" for item in existing_files if item.get("path")) or "None"
    return f"""Build or refine a complete {framework} web project.

User request:
{description}

Approved build plan:
{plan_lines}

Existing files already in the workspace:
{existing}

Instructions:
- Use read_file before editing an existing file
- Use write_file for new files and edit_file for targeted changes
- Follow the approved plan order first
- Keep file paths consistent with the plan unless a small support file is truly necessary
- Build working, previewable web output"""


def _write_seed_files(workspace: str, files: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seeded: List[Dict[str, str]] = []
    for item in files:
        path = item.get("path", "")
        content = item.get("content", "")
        if not path:
            continue
        normalized = _normalize_rel_path(path)
        absolute = os.path.join(workspace, normalized)
        os.makedirs(os.path.dirname(absolute), exist_ok=True)
        with open(absolute, "w", encoding="utf-8") as handle:
            handle.write(content)
        seeded.append({"path": normalized, "content": content})
    return seeded


def _path_to_artifact_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".html", ".htm"}:
        return "html"
    if ext in {".jsx", ".tsx"}:
        return "react"
    if ext == ".css":
        return "css"
    if ext == ".svg":
        return "svg"
    if ext in {".md", ".markdown"}:
        return "markdown"
    return "html"


@dataclass
class BuildTaskRecord:
    task_id: str
    description: str
    framework: str
    workspace: str
    plan: List[Dict[str, Any]]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "queued"
    files_created: List[str] = field(default_factory=list)
    error: str = ""
    model: Optional[str] = None
    max_iterations: int = 24
    result: Optional[Dict[str, Any]] = None
    loop: Optional[AgenticLoop] = None
    cancel_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "framework": self.framework,
            "workspace": self.workspace,
            "plan": self.plan,
            "status": self.status,
            "files_created": self.files_created,
            "error": self.error or None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
        }


class BuildService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: Dict[str, BuildTaskRecord] = {}

    def create_plan(
        self,
        description: str,
        framework: str = "static",
        files: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        existing_files = files or []
        brain = agent_service.agent.brain
        raw = brain.think(
            _build_plan_prompt(description, framework, existing_files),
            system_prompt="Return JSON only.",
            use_history=False,
            model_override=model,
        )
        if isinstance(raw, dict):
            raw = raw.get("response", raw.get("content", str(raw)))
        plan = _parse_plan_response(str(raw or ""))
        return plan or _default_plan(framework, existing_files)

    def start_build(
        self,
        description: str,
        framework: str = "static",
        files: Optional[List[Dict[str, str]]] = None,
        plan: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_iterations: int = 24,
    ) -> BuildTaskRecord:
        existing_files = files or []
        approved_plan = _dedupe_and_sort_plan(plan or self.create_plan(description, framework, existing_files, model=model))
        task_id = str(uuid.uuid4())
        workspace = tempfile.mkdtemp(prefix=f"aura-build-{task_id[:8]}-")
        seeded_files = _write_seed_files(workspace, existing_files)
        record = BuildTaskRecord(
            task_id=task_id,
            description=description,
            framework=framework,
            workspace=workspace,
            plan=approved_plan,
            model=model,
            max_iterations=max_iterations,
        )
        with self._lock:
            self._tasks[task_id] = record

        broadcast_event({
            "type": "build_start",
            "task_id": task_id,
            "framework": framework,
            "plan": approved_plan,
            "workspace": workspace,
        })

        if seeded_files:
            record.files_created = [item["path"] for item in seeded_files]
            for item in seeded_files:
                broadcast_event({
                    "type": "artifact_update",
                    "filename": item["path"],
                    "code": item["content"],
                    "artifact_type": _path_to_artifact_type(item["path"]),
                    "timestamp": time.time(),
                })

        thread = threading.Thread(
            target=self._run_build_task,
            args=(record, seeded_files),
            daemon=True,
            name=f"build-task-{task_id[:8]}",
        )
        thread.start()
        return record

    def get_task(self, task_id: str) -> Optional[BuildTaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> Optional[BuildTaskRecord]:
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return None
            record.cancel_requested = True
            record.updated_at = time.time()
            loop = record.loop
        if loop is not None:
            loop.cancel()
        broadcast_event({
            "type": "build_cancel_requested",
            "task_id": task_id,
        })
        return record

    def _run_build_task(self, record: BuildTaskRecord, seeded_files: List[Dict[str, str]]) -> None:
        permissions = PermissionManager()
        permissions.set_trust_mode(True)
        loop = AgenticLoop(
            brain=agent_service.agent.brain,
            project_root=record.workspace,
            permissions=permissions,
            model_override=record.model,
            max_iterations=record.max_iterations,
        )

        with self._lock:
            record.loop = loop
            record.status = "building"
            record.updated_at = time.time()

        seeded_paths = {item["path"] for item in seeded_files}
        touched_files: set[str] = set()
        total = max(len(record.plan), 1)
        current_file = ""

        def on_tool_start(tool_name: str, args: Dict[str, Any]) -> None:
            nonlocal current_file
            if tool_name not in ("write_file", "edit_file"):
                return
            path = args.get("path", "")
            if not isinstance(path, str) or not path.strip():
                return
            try:
                current_file = _normalize_rel_path(os.path.relpath(os.path.join(record.workspace, path), record.workspace))
            except Exception:
                try:
                    current_file = _normalize_rel_path(path)
                except ValueError:
                    current_file = path.replace("\\", "/")
            broadcast_event({
                "type": "build_progress",
                "task_id": record.task_id,
                "step": len(touched_files),
                "total": total,
                "current_file": current_file,
                "message": f"Writing {current_file}",
            })

        def on_tool_call(tool_name: str, args: Dict[str, Any], result: Any) -> None:
            if tool_name not in ("write_file", "edit_file") or _safe_result_has_error(result):
                return
            path = args.get("path", "")
            if not isinstance(path, str) or not path.strip():
                return
            try:
                rel_path = _normalize_rel_path(path)
            except ValueError:
                return
            touched_files.add(rel_path)
            with self._lock:
                record.files_created = sorted(seeded_paths | touched_files)
                record.updated_at = time.time()
            broadcast_event({
                "type": "build_progress",
                "task_id": record.task_id,
                "step": len(touched_files),
                "total": total,
                "current_file": rel_path,
                "message": f"Updated {rel_path}",
            })

        try:
            result = loop.run(
                _build_execution_prompt(record.description, record.framework, record.plan, seeded_files),
                on_tool_start=on_tool_start,
                on_tool_call=on_tool_call,
            )
            with self._lock:
                record.result = result
                record.files_created = sorted(seeded_paths | touched_files)
                record.updated_at = time.time()
                cancelled = record.cancel_requested or str(result.get("response", "")).strip() == "Cancelled by user."
                if cancelled:
                    record.status = "cancelled"
                elif not result.get("success", False):
                    record.status = "error"
                    record.error = str(result.get("response", "Build failed"))
                else:
                    record.status = "completed"
                record.loop = None

            if record.status == "cancelled":
                broadcast_event({
                    "type": "build_cancelled",
                    "task_id": record.task_id,
                    "files_created": record.files_created,
                })
            elif record.status == "error":
                broadcast_event({
                    "type": "build_error",
                    "task_id": record.task_id,
                    "error": record.error or "Build failed",
                    "files_created": record.files_created,
                })
            else:
                broadcast_event({
                    "type": "build_complete",
                    "task_id": record.task_id,
                    "files_created": len(record.files_created),
                    "paths": record.files_created,
                    "workspace": record.workspace,
                })
        except Exception as exc:
            logger.exception("[BuildService] Build task failed")
            with self._lock:
                record.status = "error"
                record.error = str(exc)
                record.updated_at = time.time()
                record.loop = None
            broadcast_event({
                "type": "build_error",
                "task_id": record.task_id,
                "error": str(exc),
                "files_created": record.files_created,
            })


build_service = BuildService()
