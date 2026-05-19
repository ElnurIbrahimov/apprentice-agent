# Aura CLI — Block-Based Output + AGENTS.md Support Plan

> **For Hermes:** Implement task-by-task. Commit at each task boundary.

**Goal:** Elevate Aura CLI from linear scroll to block-based navigable output, and add AGENTS.md/CLAUDE.md project persona support.

**Architecture:** Build on existing `DisclosureManager` (already has collapsible sections) to create a `BlockManager` that wraps every tool call and agent response as a discrete, numbered block. Each block has an ID, can be expanded/collapsed, copied to clipboard, and referenced by number. For AGENTS.md, add a lightweight file read in `session_bootstrap.py` that appends to project context.

**Tech Stack:** Python 3.12, Rich (Live, Panel, Text), prompt_toolkit

---

## Phase 1: Block Registry Infrastructure

### Task 1: Create BlockManager in aura/cli/blocks.py

**Objective:** A registry that tracks every output block as it's rendered.

**Files:**
- Create: `aura/cli/blocks.py`

**Implementation:**

```python
"""Block-based output manager for navigable conversation history."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class OutputBlock:
    """A single output block in the conversation."""
    id: int
    block_type: str  # "response", "tool_call", "tool_result", "error", "diff", "info"
    title: str       # one-line summary
    content: str     # full content (markdown)
    expanded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BlockManager:
    """Tracks output blocks for navigation, copy, and export."""

    _MAX_BLOCKS = 200

    def __init__(self):
        self._blocks: List[OutputBlock] = []
        self._counter: int = 0
        self._lock = threading.Lock()
        self._on_block_added: Optional[Callable] = None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._blocks)

    def add(self, block_type: str, title: str, content: str,
            metadata: Optional[Dict] = None, expanded: bool = False) -> int:
        """Add a block and return its ID."""
        with self._lock:
            self._counter += 1
            block = OutputBlock(
                id=self._counter,
                block_type=block_type,
                title=title,
                content=content,
                expanded=expanded,
                metadata=metadata or {},
            )
            self._blocks.append(block)
            # Prune old blocks
            while len(self._blocks) > self._MAX_BLOCKS:
                self._blocks.pop(0)
        if self._on_block_added:
            try:
                self._on_block_added(block)
            except Exception:
                pass
        return block.id

    def get(self, block_id: int) -> Optional[OutputBlock]:
        with self._lock:
            return next((b for b in self._blocks if b.id == block_id), None)

    def get_recent(self, n: int = 10) -> List[OutputBlock]:
        with self._lock:
            return list(self._blocks[-n:])

    def get_last(self) -> Optional[OutputBlock]:
        with self._lock:
            return self._blocks[-1] if self._blocks else None

    def set_on_block_added(self, callback: Callable) -> None:
        self._on_block_added = callback

    def clear(self) -> None:
        with self._lock:
            self._blocks.clear()
            self._counter = 0


# Process-wide singleton — chat_session.py wires this up
_block_manager: Optional[BlockManager] = None


def get_block_manager() -> BlockManager:
    global _block_manager
    if _block_manager is None:
        _block_manager = BlockManager()
    return _block_manager
```

**Verification:**
```bash
python -c "from aura.cli.blocks import BlockManager; bm = BlockManager(); bm.add('response', 'test', 'hello'); assert bm.count == 1; assert bm.get_last().title == 'test'; print('PASS')"
```

### Task 2: Wire BlockManager into ChatSession

**Objective:** Create the BlockManager singleton during session init and expose it via the CLI context.

**Files:**
- Modify: `aura/cli/chat_session.py` (in `_init_ui_and_state`)
- Modify: `aura/cli/context.py` (add `blocks` to CLIContext)
- Modify: `aura/cli/chat_session_runtime.py` (wire `/blocks` command)

---

## Phase 2: Block Rendering for Tool Calls

### Task 3: Render tool calls as blocks with block IDs

**Objective:** Every `show_tool_call()` and `show_tool_result_inline()` produces a numbered block.

**Files:**
- Modify: `aura/cli/display/__init__.py` (show_tool_call, show_tool_result_inline)

**Design:** After rendering the tool call/result inline (existing behavior), also register it as a block. Show the block ID as `[#12]` in dim text at the end of the tool call line.

### Task 4: Render streaming responses as blocks

**Objective:** When StreamingResponse.finish() is called, register the accumulated markdown as a numbered block and show a block footer.

**Files:**
- Modify: `aura/cli/display/streaming.py` (StreamingResponse.finish)

### Task 5: Render errors and info messages as blocks

**Objective:** `show_error()`, `show_info()` also produce numbered blocks.

**Files:**
- Modify: `aura/cli/display/__init__.py` (show_error, show_info, show_response, show_context_summary)

---

## Phase 3: Block Navigation Commands

### Task 6: Add `/blocks` command to list recent blocks

**Objective:** `/blocks` shows a scrollable list of recent blocks. `/blocks 5` shows block #5 expanded.

**Files:**
- Create/modify: `aura/cli/commands/session_commands.py` (add handle_blocks)
- Modify: `aura/cli/commands/__init__.py` (register /blocks command)

### Task 7: Add `/copy` command to copy last response or block N to clipboard

**Objective:** `/copy` copies last response. `/copy 12` copies block #12.

**Files:**
- Modify: `aura/cli/commands/copy_command.py` (extend existing handler)

---

## Phase 4: AGENTS.md / CLAUDE.md Support

### Task 8: Read AGENTS.md and CLAUDE.md in session bootstrap

**Objective:** In addition to AURA.md, read the industry-standard AGENTS.md and CLAUDE.md files from project root and include them in the project context.

**Files:**
- Modify: `aura/cli/session_bootstrap.py` (in gather_context or equivalent)

**Implementation:** After reading AURA.md, also attempt to read `AGENTS.md` and `CLAUDE.md` from project_root. Merge into project context with a clear header indicating source.

### Task 9: Add `/agents` command to show loaded project persona

**Objective:** `/agents` prints the loaded AGENTS.md / CLAUDE.md / AURA.md content.

**Files:**
- Modify: `aura/cli/commands/ui_commands.py` (add handle_agents)
- Modify: `aura/cli/commands/__init__.py` (register /agents)

---

## Phase 5: Polish & Tests

### Task 10: Run full test suite and fix any regressions

**Objective:** 558 tests must pass. Add tests for new BlockManager.

**Files:**
- Create: `tests/cli/test_blocks.py`

### Task 11: Block output in bottom toolbar

**Objective:** Status bar shows "12 blocks" when blocks exist.

**Files:**
- Modify: `aura/cli/status_bar.py` (add block count to P2 tier)

---

## Risks & Tradeoffs

- **Memory**: BlockManager caps at 200 blocks to prevent unbounded growth
- **Performance**: Block registration is O(1) append; lookup is O(n) linear scan (fine for 200 entries)
- **Backward compat**: All existing display functions continue to work; blocks are an additive layer
- **AGENTS.md format**: The format varies across tools; we just inject the raw text as context

## Out of Scope (for this phase)

- Agent View / dashboard (needs full TUI split-pane — bigger architectural change)
- MCP server mode (needs async server + protocol work)
- Worktree isolation for /fleet (needs git worktree integration)
- Block-based file export (`.aura-blocks.json` — save full history)
