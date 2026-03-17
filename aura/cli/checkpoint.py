"""File checkpoint system for rewind/undo support."""
from __future__ import annotations
import json
import shutil
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
        self._index: List[Dict] = self._load_index()

    def _load_index(self) -> List[Dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def snapshot(self, file_path: str, label: str = "") -> str:
        """Snapshot a single file. Returns checkpoint ID."""
        return self.snapshot_multi([file_path], label=label)

    def snapshot_multi(self, file_paths: List[str], label: str = "") -> str:
        """Snapshot multiple files. Returns checkpoint ID."""
        cp_id = f"cp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        cp_dir = self._dir / cp_id
        cp_dir.mkdir(parents=True, exist_ok=True)

        files_info = []
        for fp in file_paths:
            src = Path(fp)
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
                })

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
        entry = next((e for e in self._index if e["id"] == checkpoint_id), None)
        if not entry:
            return False
        cp_dir = self._dir / checkpoint_id
        if not cp_dir.exists():
            return False
        for f_info in entry["files"]:
            src = cp_dir / f_info["backup_name"]
            dest = Path(f_info["original_path"])
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dest))
        return True

    def list_checkpoints(self) -> List[Dict]:
        """Return all checkpoints, most recent first."""
        return list(self._index)

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Get a specific checkpoint's metadata."""
        return next((e for e in self._index if e["id"] == checkpoint_id), None)

    def clear(self):
        """Remove all checkpoints."""
        for entry in self._index:
            cp_dir = self._dir / entry["id"]
            if cp_dir.exists():
                shutil.rmtree(cp_dir, ignore_errors=True)
        self._index.clear()
        self._save_index()
