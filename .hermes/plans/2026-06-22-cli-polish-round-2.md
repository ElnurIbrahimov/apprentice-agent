# CLI Polish Plan — Round 2

**Scope:** Address the 4 polish items I deliberately deferred in commit `dbb0a6e`.

**Plan file:** `.hermes/plans/2026-06-22-cli-polish-round-2.md`

---

## Pillar 1: Unthemed status messages → themed helpers

### 1a. Add 2 new helpers to `aura/cli/display/__init__.py`

After `show_warning` (line 722), add:

- `show_success(message: str) -> None` — green ✓ icon + message. Use for "switched to X", "plan complete", "model override cleared". Maps to the 7 `[green]…[/green]` sites.
- `show_progress(message: str) -> None` — cyan · icon + message. Use for "Updating Aura…", "Recalling memories…". Maps to the 2 `[cyan]…[/cyan]` sites.

Existing helpers:
- `show_info` already handles `[dim]…[/dim]` — 0 callers currently use it. I will use it for the ~30 `[dim]` sites (hints, "cancelled", "no records", "usage: …").
- `show_warning` adds a ⚠ triangle — that changes UX for plain `[yellow]` status lines ("Step failed. Continue?", "Interrupting…"). I will NOT convert yellow status lines to `show_warning` (it would be a UX regression). Yellow usage hints ("Usage: /git <cmd>") likewise stay raw. Only the 5 pure-status yellow lines that aren't user-prompted will be considered.

### 1b. Conversion rules

| Raw | Replacement | Why |
|---|---|---|
| `console.print("[dim]foo[/dim]")` | `show_info("foo")` | Already-themed helper. |
| `console.print("[green]foo[/green]")` | `show_success("foo")` | New themed helper. |
| `console.print("[cyan]foo[/cyan]")` | `show_progress("foo")` | New themed helper. |
| `console.print("[yellow]foo[/yellow]")` for user prompts ("Continue? (y/n)") | **Keep raw** | These are interactive prompts, not warnings. |
| `console.print("[yellow]foo[/yellow]")` for status ("Interrupting…") | **Keep raw** | Status text, not warning. |
| `console.print("[yellow]Usage: …[/yellow]")` | **Keep raw** | Usage hint, not warning. |

Net: 30+ sites converted to themed helpers. The yellow status/usage sites stay as-is to preserve UX. The change is purely about consistency, not visual redesign.

### 1c. Files affected (17 files have `[dim]/[green]/[cyan]` raw prints)

All 17 files enumerated by the grep. Each will get:
1. Import added: `from ..display import console, show_info, show_success, show_progress` (only the ones it uses).
2. Raw prints replaced.

Skip pattern: `_int_console.print("[dim]…")` becomes `_int_console = console` (drop the alias) → `show_info("…")`. This naturally addresses **Pillar 2** (over-defensive aliases).

---

## Pillar 2: Over-defensive aliases in `agent_commands.py`

The 22 aliases (`_int_console`, `_goal_err`, `_fleet_console`, etc.) are unnecessary. The imports are already function-local; no name conflict exists. Drop the aliases — they obscure which is the same `console` everywhere.

**Before:** `from ..display import console as _int_console`
**After:** `from ..display import console`

Then update call sites: `_int_console.print(...)` → `console.print(...)`.

This is a 22-line mechanical change. No behavior change.

---

## Pillar 3: `/help` examples for the rest of the commands

### 3a. Approach: move examples into the command's own module

The current `EXAMPLES` dict in `ui_commands.py:140-158` is hardcoded for 9 commands. This pattern doesn't scale. Better: each command module can optionally export an `EXAMPLES` list (or the `@command` decorator can accept an `examples` kwarg).

I'll go with: add `examples: list[str] | None = None` to the `command()` decorator in `common.py`. Each example is a one-line string. The handler module declares them at the decorator site.

```python
@command("/model", "View/set model (auto, <name>)",
          tier=TIER_STABLE,
          examples=[
              "/model              -- open interactive picker",
              "/model qwen3:8b     -- lock to specific model",
              "/model auto         -- return to auto-routing",
          ])
def handle_model(...): ...
```

This way:
- The example lives next to the command it documents.
- New commands can add examples without editing `ui_commands.py`.
- `/help <cmd>` can pull examples from the registry without a separate dict.

### 3b. Add examples to all 84 commands

For each command, write 1-3 examples. Some commands already have implicit examples in their docstring (e.g. `/chain: "Run prompt pipelines (step1 -> step2 -> step3)"` — that's already an example). I'll add `examples=[…]` to the decorator and prefer the explicit form.

Trade-off: 84 commands × ~2 examples each = ~170 lines. The tedium is the price; the payoff is that `/help <any cmd>` is useful for every command, not just 9.

### 3c. Update `handle_help` in `ui_commands.py`

Replace the hardcoded `EXAMPLES` dict with: read examples from the registry:

```python
# When the command is found in COMMAND_REGISTRY, the registry entry
# is the handler function. Examples are stashed as `func.__aura_examples__`
# (or a module-level dict lookup) so we don't need a separate map.
```

Implementation: the `command()` decorator will attach the examples list as `func.__aura_examples__`. `handle_help` reads that attribute.

### 3d. Migration of the 9 existing hardcoded examples

The 9 in `ui_commands.py:140-158` move to their respective decorator sites:
- `/model` → `tool_commands.py` or wherever `handle_model` lives
- `/copy` → `copy_command.py`
- `/fleet` → `agent_commands.py` (`handle_fleet`)
- `/chain` → `agent_commands.py` (`handle_chain`)
- `/test` → `tool_commands.py` (`handle_test`)
- `/shell` → `tool_commands.py` (`handle_shell`)
- `/grep` → `tool_commands.py` (`handle_grep`)
- `/research` → `research_commands.py`
- `/debate` → `agent_commands.py` (`handle_debate`)

I'll move them as I add examples to all 84 commands.

---

## Pillar 4: `ChatSession` smoke test

### 4a. What's actually testable

`ChatSession.__init__` does:
- Phase 1: `_init_display(agent)` — calls `show_banner`, `show_welcome_info`, `show_startup_diagnostics`
- Phase 2: `_init_session_core(...)` — calls `build_session_bootstrap`, `build_permission_manager`
- Phase 3: `_init_subsystems_and_steering(...)` — sets up steering, activity log, checkpoints
- Phase 4: `_init_ui_and_state(...)` — wires controllers, channels

To test this without a real agent, brain, or auramd, I'd need to mock 8-10 dependencies. That's an integration test, not a smoke test.

A genuine **unit test** for `ChatSession` is hard because the class is tightly coupled to session bootstrap, the agent, the brain, the bridge, etc. The existing 3-controller tests (test_chat_session_execution/runtime/signals) cover the parts that have been refactored into controllers.

**The right move:** instead of a full `ChatSession` test, add tests for the **stateless helpers** that the constructors call. Specifically:
- `apply_model_override(model)` — pure logic that updates 3 mirrors
- `_dispatch_command(user_input)` — wraps `handle_command` in a try/except

Both are testable in isolation. `apply_model_override` is the one most likely to have bugs (the docstring says it updates 3 mirrors; if a 4th mirror is added without updating, the test catches it).

### 4b. Plan

Add `tests/cli/test_chat_session_apply_model.py`:
- Mock `agent.brain.set_model_override`, `agentic.model_override = ...`
- Test that `apply_model_override("gpt-4")`:
  - Calls `brain.set_model_override("gpt-4")`
  - Sets `agentic.model_override = "gpt-4"`
  - Sets `self.current_model = "gpt-4"`
- Test that `apply_model_override("auto")`:
  - Calls `brain.set_model_override(None)` (None is the auto signal)
  - Sets `self.current_model = "auto"`
- Test that `apply_model_override(None)`:
  - Same as auto

This locks in the 3-mirror contract.

Add `tests/cli/test_chat_session_dispatch.py`:
- Mock `handle_command` to raise
- Test that `_dispatch_command` calls `show_error` with the exception
- Test that the status bar gets re-synced (current_model fallback)

---

## Order of operations (single commit per pillar)

1. **Pillar 1** (helpers + conversions) — adds 2 helpers, converts 30+ sites, drops 22 aliases. Net: ~30 lines added, ~30 lines removed.
2. **Pillar 3** (decorator examples) — adds `examples` to `command()`, adds 1-3 examples to each of 84 commands, rewrites `handle_help`. Net: +200 lines, more readable `/help`.
3. **Pillar 4** (smoke test) — adds 2 test files, ~60 lines.
4. **Pillar 2** (aliases) — folded into Pillar 1 because the alias changes happen naturally when imports are unified.

Each pillar: ruff clean + tests pass + commit. Push at the end.

---

## Verification

- `pytest tests/cli/ -q --timeout=10` → 519+ pass (no regressions).
- `ruff check aura/cli/` → clean.
- `aura --help` shows all 84 commands.
- `aura <any cmd> --help` shows at least one example for every command.
- `aura /help /bench` shows the `/bench` examples.
- `aura /help /unknown` still suggests via difflib.

---

## Risks

- **Pillar 3 example authoring**: writing 1-3 good examples for 84 commands is tedious and might be rushed. Mitigation: copy from each command's docstring (most have an example section), fallback to a single line of the form `/<cmd> <what it does>`.
- **Pillar 4 testability**: `ChatSession.__init__` calls into `build_session_bootstrap` which reads the filesystem. The test needs `tmp_path` and `monkeypatch` to be hermetic. Mitigation: skip the constructor test entirely; test `apply_model_override` and `_dispatch_command` directly via `unittest.mock`.
- **Pillar 1 conversion of yellow status**: I will NOT convert yellow status messages — they would get a ⚠ icon that doesn't fit a status line. Document this in code comments.

---

## Out of scope

- Adding new helpers `show_progress`, `show_success` to a different module (they belong with `show_info`/`show_warning`).
- Backwards-compat shims for the old raw-print pattern.
- Refactoring `handle_help` to use a Markdown renderer (out of scope; the current `console.print` calls are fine).
