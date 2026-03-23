"""Git-style conversation branching.

Branches are named conversation histories. The main branch always exists.
Forking deep-copies history. Merging appends new messages back to parent.
State persists as JSON in the session directory.
"""

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Branch:
    id: str                        # e.g. "main", "fork-1", "fork-2"
    parent_id: Optional[str]       # None for main branch
    name: str                      # User-readable name
    created_at: float
    history: list                  # Conversation messages (list[dict])
    fork_point: int                # Index in parent's history where fork happened
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "name": self.name,
            "created_at": self.created_at,
            "history": self.history,
            "fork_point": self.fork_point,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Branch":
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            name=data.get("name", data["id"]),
            created_at=data.get("created_at", 0.0),
            history=data.get("history", []),
            fork_point=data.get("fork_point", 0),
            metadata=data.get("metadata", {}),
        )


class ConversationTree:
    """Manages branching conversations within a session."""

    def __init__(self, session_dir: Optional[Path] = None):
        self.session_dir = session_dir
        self.branches: dict[str, Branch] = {}
        self.current_branch: str = "main"
        self._fork_counter: int = 0
        self._init_main()

    def _init_main(self) -> None:
        """Create main branch if it doesn't exist."""
        if "main" not in self.branches:
            self.branches["main"] = Branch(
                id="main",
                parent_id=None,
                name="main",
                created_at=time.time(),
                history=[],
                fork_point=0,
            )

    def fork(self, name: Optional[str] = None) -> Branch:
        """Fork from current branch at current position. Returns new branch."""
        current = self.branches[self.current_branch]
        self._fork_counter += 1
        fork_id = f"fork-{self._fork_counter}"
        branch_name = name or f"Branch {self._fork_counter}"

        # Shallow copy: messages are dicts that are never mutated after creation,
        # so sharing references is safe and avoids expensive deepcopy on large histories.
        new_branch = Branch(
            id=fork_id,
            parent_id=current.id,
            name=branch_name,
            created_at=time.time(),
            history=list(current.history),
            fork_point=len(current.history),
        )
        self.branches[fork_id] = new_branch
        self.current_branch = fork_id
        return new_branch

    def switch(self, branch_id: str) -> Branch:
        """Switch to a different branch by id or numeric shorthand."""
        # Allow numeric shorthand: "1" -> "fork-1", "2" -> "fork-2", etc.
        if branch_id.isdigit():
            numeric_id = f"fork-{branch_id}"
            if numeric_id in self.branches:
                branch_id = numeric_id
        # Also allow "main"
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")
        self.current_branch = branch_id
        return self.branches[branch_id]

    def list_branches(self) -> list[Branch]:
        """List all branches, main first."""
        result = []
        if "main" in self.branches:
            result.append(self.branches["main"])
        for bid, branch in self.branches.items():
            if bid != "main":
                result.append(branch)
        return result

    def get_current(self) -> Branch:
        """Get current branch."""
        return self.branches[self.current_branch]

    def get_children(self, branch_id: str) -> list[Branch]:
        """Get direct children of a branch."""
        return [b for b in self.branches.values() if b.parent_id == branch_id]

    def merge_to_parent(self) -> dict:
        """Merge current branch's new messages into parent.

        Only messages added *after* the fork point are merged.
        Switches back to parent branch on success.
        """
        current = self.branches[self.current_branch]
        if not current.parent_id:
            return {"error": "Cannot merge main branch — it has no parent"}

        parent = self.branches[current.parent_id]
        new_messages = current.history[current.fork_point:]

        if not new_messages:
            # Switch back even if nothing to merge
            self.current_branch = current.parent_id
            return {"merged": 0, "target": parent.name, "from": current.name}

        parent.history.extend(copy.deepcopy(new_messages))
        old_branch = current.name
        self.current_branch = current.parent_id
        return {"merged": len(new_messages), "target": parent.name, "from": old_branch}

    def sync_history(self, history: list) -> None:
        """Sync external history list into the current branch.

        Called after the agentic loop appends messages so the branch
        stays in sync without the loop needing to know about branches.
        """
        current = self.branches[self.current_branch]
        current.history = history

    def save(self) -> None:
        """Persist branch tree to disk as JSON."""
        if not self.session_dir:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        target = self.session_dir / "branches.json"
        tmp = self.session_dir / "branches.tmp"

        data = {
            "current_branch": self.current_branch,
            "fork_counter": self._fork_counter,
            "branches": {bid: b.to_dict() for bid, b in self.branches.items()},
        }

        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            import os
            os.replace(str(tmp), str(target))
        except Exception as e:
            logger.error(f"[ConversationTree] Save failed: {e}")
            if tmp.exists():
                tmp.unlink()

    def load(self) -> bool:
        """Load branch tree from disk. Returns True if loaded successfully."""
        if not self.session_dir:
            return False
        branch_file = self.session_dir / "branches.json"
        if not branch_file.exists():
            return False

        try:
            with open(branch_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"[ConversationTree] Load failed: {e}")
            return False

        self.current_branch = data.get("current_branch", "main")
        self._fork_counter = data.get("fork_counter", 0)
        self.branches = {}
        for bid, bdata in data.get("branches", {}).items():
            self.branches[bid] = Branch.from_dict(bdata)

        # Ensure main always exists
        self._init_main()
        return True
