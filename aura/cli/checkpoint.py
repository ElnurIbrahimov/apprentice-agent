"""File checkpoint system for rewind/undo support."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional


class CheckpointManager:
    """Manages file snapshots for checkpoint/rewind."""

    def __init__(self, checkpoint_dir: Optional[Path] = None, max_checkpoints: int = 50):
        self._dir = Path(checkpoint_dir or Path.cwd() / ".aura_checkpoints")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_checkpoints
        self._index_path = self._dir / "index.json"
        self._lock = threading.Lock()
        self._index: List[Dict] = self._load_index()

    def _load_index(self) -> List[Dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_index(self):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
            os.replace(tmp_path, str(self._index_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def snapshot(self, file_path: str, label: str = "") -> str:
        """Snapshot a single file. Returns checkpoint ID."""
        return self.snapshot_multi([file_path], label=label)

    def snapshot_multi(self, file_paths: List[str], label: str = "") -> str:
        """Snapshot multiple files. Returns checkpoint ID."""
        with self._lock:
            cp_id = f"cp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            if not re.match(r'^cp_\d+_[a-f0-9]{8}$', cp_id):
                raise ValueError(f"Generated checkpoint ID failed validation: {cp_id}")
            cp_dir = self._dir / cp_id
            cp_dir.mkdir(parents=True, exist_ok=True)

            files_info = []
            for fp in file_paths:
                src = Path(fp)
                if src.is_symlink():
                    continue  # don't snapshot symlinks
                if src.exists():
                    dest = cp_dir / src.name
                    counter = 0
                    while dest.exists():
                        counter += 1
                        dest = cp_dir / f"{src.stem}_{counter}{src.suffix}"
                    shutil.copy2(str(src), str(dest))
                    files_info.append({
                        "original_path": str(src.resolve()),
                        "backup_name": dest.name,
                        "original_exists": True,
                    })
                else:
                    files_info.append(
                        {
                            "original_path": str(src.resolve()),
                            "backup_name": None,
                            "original_exists": False,
                        }
                    )

            entry = {
                "id": cp_id,
                "timestamp": time.time(),
                "label": label,
                "files": files_info,
            }
            self._index.insert(0, entry)

            while len(self._index) > self._max:
                old = self._index.pop()
                old_dir = self._dir / old["id"]
                if old_dir.exists():
                    shutil.rmtree(old_dir, ignore_errors=True)

            self._save_index()
            return cp_id

    def restore(self, checkpoint_id: str) -> bool:
        """Restore files from a checkpoint."""
        # Validate checkpoint_id format to prevent path traversal
        if not re.match(r'^cp_\d+_[a-f0-9]{8}$', checkpoint_id):
            return False
        with self._lock:
            entry = next((e for e in self._index if e["id"] == checkpoint_id), None)
            if not entry:
                return False
            cp_dir = self._dir / checkpoint_id
            if not cp_dir.exists():
                return False
            for f_info in entry["files"]:
                dest = Path(f_info["original_path"]).resolve()
                backup_name = f_info.get("backup_name")
                if not f_info.get("original_exists", True):
                    try:
                        if dest.is_file() or dest.is_symlink():
                            dest.unlink()
                        elif dest.is_dir():
                            shutil.rmtree(dest)
                    except OSError:
                        continue
                    continue

                if not backup_name:
                    continue

                backup_name = os.path.basename(backup_name)  # strip path components
                src = cp_dir / backup_name
                if src.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dest))
            return True

    def list_checkpoints(self) -> List[Dict]:
        """Return all checkpoints, most recent first."""
        with self._lock:
            return list(self._index)

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Get a specific checkpoint's metadata (returns a copy for thread safety)."""
        with self._lock:
            entry = next((e for e in self._index if e["id"] == checkpoint_id), None)
            return dict(entry) if entry is not None else None

    def clear(self):
        """Remove all checkpoints."""
        with self._lock:
            for entry in self._index:
                cp_dir = self._dir / entry["id"]
                if cp_dir.exists():
                    shutil.rmtree(cp_dir, ignore_errors=True)
            self._index.clear()
            self._save_index()
