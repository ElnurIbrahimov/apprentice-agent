"""StreamingResponse + splitter helpers.

Extracted from the old monolithic ``display.py``. Handles live token streaming
to the terminal via Rich's ``Live`` display, including partial code-fence
recovery so unclosed blocks render as proper code while the stream continues.
"""
from __future__ import annotations

from typing import Optional

from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text

# Late attribute access via the package so tests can patch
# `aura.cli.display.console` and have the mock take effect here.
from aura.cli import display as _display


class StreamingResponse:
    """Manages live token streaming to terminal via Rich.

    Clean output — no panels, no borders. Just markdown flowing in.
    When pause() is called (for tool-call display), accumulated text is
    printed permanently. When resume() is called, a fresh Live starts.

    Partial code blocks: if the stream is inside an unclosed code fence,
    the block is temporarily closed for rendering so the user sees code
    as it arrives instead of a blank gap.

    Token stats: after finish(), shows (N tokens . X.X tok/s) dimmed.
    """

    def __init__(self, model: str = "") -> None:
        self._accumulated: str = ""
        self._live: Optional[Live] = None
        self._model: str = model
        self._displayed: bool = False
        self._permanent_len: int = 0
        self._spinner_active: bool = False
        self._start_time: float = 0.0
        self._first_chunk_time: float = 0.0
        self._fade_frames_remaining: int = 0
        # Fence-state machine: tracks whether we're currently inside an
        # unclosed ``` block. Carried across pause/resume so a tool call in
        # the middle of a code block doesn't break the render when the block
        # continues after resume. A pure fence-count parity over `new_content`
        # would drop this context as soon as `_permanent_len` advanced past
        # the opener.
        self._in_fence: bool = False
        # Parallel tool tracking: each resume() before the next pause()
        # means another tool is running concurrently. The spinner swaps
        # to "N tools running" when >1 so the user isn't staring at a
        # single label while several tools chug.
        self._parallel_active: int = 0

    def start(self) -> None:
        """Begin live rendering context with a themed thinking spinner."""
        # Mirror the resume() re-entrancy guard. If an outer try/finally
        # failed to release the previous Live (exception in chunk/pause
        # before stop()), starting a second Live would raise LiveError.
        if self._live is not None:
            return
        import time as _t
        _display.console.print()
        self._spinner_active = True
        self._start_time = _t.monotonic()

        from ..spinner import AuraSpinner
        spinner = AuraSpinner()
        self._live = Live(
            spinner,
            console=_display.console, refresh_per_second=12, transient=True,
        )
        self._live._aura_spinner = spinner
        self._live.start()

    @staticmethod
    def _close_partial_fences(text: str) -> str:
        """Legacy parity-based helper. Kept for callers that don't carry fence
        state (e.g. `finish()` — stream is ending anyway). Prefer
        `_wrap_for_render` which accounts for state carried across pause/resume."""
        fence_count = sum(
            1 for line in text.split("\n") if line.lstrip().startswith("```")
        )
        if fence_count % 2 == 1:
            return text + "\n```"
        return text

    def _wrap_for_render(self, new_content: str) -> str:
        """Return *new_content* wrapped with synthetic fences so Rich
        renders it correctly even when permanent_len is mid-block.

        - If we started this render INSIDE a fence, prepend an opener so
          Rich recognizes the content as code.
        - Walk new_content toggling fence parity.
        - If we end INSIDE a fence, append a closer so the final render
          has a balanced block.

        The synthetic fences affect only the rendered string — they are
        NOT persisted back into `self._accumulated`.
        """
        # Update the instance flag AFTER the render so concurrent Rich
        # re-renders of overlapping slices stay consistent. Use local
        # variables here; commit once at the end.
        entering_fence = self._in_fence
        exiting_fence = entering_fence
        for line in new_content.split("\n"):
            if line.lstrip().startswith("```"):
                exiting_fence = not exiting_fence

        rendered = new_content
        if entering_fence:
            # Open a bare fence so Rich treats this as code from the top.
            rendered = "```\n" + rendered
        if exiting_fence:
            rendered = rendered + "\n```"
        return rendered

    def _advance_fence_state(self, text: str) -> None:
        """Toggle `self._in_fence` for each fence line in *text*. Called
        from chunk() AFTER render so the state reflects accumulated text."""
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                self._in_fence = not self._in_fence

    def chunk(self, text: str) -> None:
        """Append a text chunk and re-render NEW content since last pause."""
        if self._spinner_active:
            self._spinner_active = False
            self._fade_frames_remaining = 3
            if not self._first_chunk_time:
                import time as _t
                self._first_chunk_time = _t.monotonic()
        self._accumulated += text
        if self._live:
            new_content = self._accumulated[self._permanent_len:]
            renderable = self._wrap_for_render(new_content)
            try:
                md = Markdown(renderable, code_theme=_display._get_code_theme())
                if self._fade_frames_remaining > 0:
                    self._fade_frames_remaining -= 1
                    padded = Padding(md, (0, 2), style="dim")
                else:
                    padded = Padding(md, (0, 2))
                self._live.update(padded)
            except Exception:
                # Any Rich render error (MarkupError, LiveError, etc.) must
                # NOT kill the stream callback. Fall back to plain text and
                # swallow failures in the fallback too.
                try:
                    self._live.update(Padding(Text(new_content), (0, 2)))
                except Exception:
                    pass

    def pause(self) -> None:
        """Pause live rendering for tool call display."""
        if not self._live:
            return
        new_content = self._accumulated[self._permanent_len:]
        try:
            if new_content.strip():
                renderable = self._wrap_for_render(new_content)
                try:
                    md = Markdown(renderable, code_theme=_display._get_code_theme())
                    self._live.update(Padding(md, (0, 2)))
                except Exception:
                    try:
                        self._live.update(Padding(Text(new_content), (0, 2)))
                    except Exception:
                        pass
                try:
                    self._live.transient = False
                except Exception:
                    pass
            # permanent_len is about to advance — capture the fence state at
            # the new boundary so the next resume() starts with correct context.
            self._advance_fence_state(new_content)
        finally:
            # Always release Rich Live slot — otherwise a Rich error above
            # would leave it taken and the next turn crashes.
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
            self._permanent_len = len(self._accumulated)
            # Reset parallel counter — the next resume() starts a fresh
            # spinner for the next batch of tool calls.
            self._parallel_active = 0

    def resume(self) -> None:
        """Resume live rendering after tool call with a fresh spinner.

        Parallel tool batches emit one tool_result (and therefore one
        resume()) per tool. Rich only permits one active Live, so when a
        spinner is already running we bump the parallel-tool counter on it
        instead of starting a second Live.
        """
        self._parallel_active += 1
        if self._live is not None:
            spinner = getattr(self._live, "_aura_spinner", None)
            if spinner is not None:
                try:
                    spinner.set_parallel_count(self._parallel_active)
                except Exception:
                    pass
            return
        colors = _display._get_theme_colors()
        _display.console.print(f"  [{colors.get('text_muted', 'dim')}]\u00b7\u00b7\u00b7[/{colors.get('text_muted', 'dim')}]")
        self._spinner_active = True
        from ..spinner import AuraSpinner
        spinner = AuraSpinner()
        if self._parallel_active > 1:
            try:
                spinner.set_parallel_count(self._parallel_active)
            except Exception:
                pass
        self._live = Live(spinner, console=_display.console, refresh_per_second=12, transient=True)
        self._live._aura_spinner = spinner
        self._live.start()

    def finish(self) -> None:
        """Finalize display — print remaining content, show attribution + token stats."""
        import time as _t

        if self._live:
            new_content = self._accumulated[self._permanent_len:]
            try:
                if new_content.strip():
                    # Use the state-aware wrapper in case the stream ends mid-fence
                    # (e.g. LLM truncated mid-block) — otherwise Rich renders the
                    # partial block as prose.
                    renderable = self._wrap_for_render(new_content)
                    try:
                        md = Markdown(renderable, code_theme=_display._get_code_theme())
                        final = Padding(md, (0, 2))
                    except Exception:
                        final = Padding(Text(new_content), (0, 2))
                    try:
                        self._live.update(final)
                        self._live.transient = False
                    except Exception:
                        pass
                    self._displayed = True
                else:
                    self._displayed = bool(self._accumulated.strip())
            finally:
                # Always release Rich Live slot — otherwise next turn crashes.
                try:
                    self._live.stop()
                except Exception:
                    pass
                self._live = None
                self._permanent_len = len(self._accumulated)
        else:
            self._displayed = bool(self._accumulated.strip())

        if self._displayed and self._accumulated and self._first_chunk_time:
            elapsed = _t.monotonic() - self._first_chunk_time
            # Delegate to token_manager which handles CJK / code-heavy
            # content differently — the old len/3.5 heuristic under-counted
            # by ~30% on code-heavy responses.
            try:
                from aura.core.token_manager import estimate_tokens as _est_tok
                token_estimate = _est_tok(self._accumulated)
            except Exception:
                token_estimate = int(len(self._accumulated) / 3.5)
            if elapsed > 0.1 and token_estimate > 5:
                tok_per_sec = token_estimate / elapsed
                extras: list[str] = []
                try:
                    delta = float(getattr(self, "_last_turn_cost", 0.0) or 0.0)
                    if delta > 0.0:
                        extras.append(f"${delta:.4f}")
                except (TypeError, ValueError):
                    pass
                try:
                    used = int(getattr(self, "_ctx_used", 0) or 0)
                    limit = int(getattr(self, "_ctx_limit", 0) or 0)
                    if used > 0 and limit > 0:
                        pct = int(100 * used / max(limit, 1))
                        extras.append(f"{pct}% ctx")
                except (TypeError, ValueError):
                    pass
                suffix = (" \u00b7 " + " \u00b7 ".join(extras)) if extras else ""
                _display.console.print(
                    f"  [dim]({token_estimate} tokens \u00b7 {tok_per_sec:.1f} tok/s{suffix})[/dim]"
                )

        if self._model and self._displayed:
            _display.console.print(f"  [dim]{self._model}[/dim]")

        _display.console.print()

    def set_turn_stats(self, cost_delta: float = 0.0, ctx_used: int = 0, ctx_limit: int = 0) -> None:
        """Attach per-turn stats so finish() can include them in the summary."""
        self._last_turn_cost = cost_delta
        self._ctx_used = ctx_used
        self._ctx_limit = ctx_limit

    @property
    def displayed(self) -> bool:
        return self._displayed

    @property
    def text(self) -> str:
        return self._accumulated


def _split_for_streaming(text: str) -> list[str]:
    """Split text into word-based chunks for streaming display."""
    words = text.split(" ")
    chunks = []
    chunk_size = 1 if len(text) < 200 else (3 if len(text) < 1000 else 5)
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words)
        if i + chunk_size < len(words):
            chunk += " "
        chunks.append(chunk)
    return chunks


def _split_into_blocks(text: str) -> list[str]:
    """Split markdown text into top-level blocks by double-newlines.

    Respects code fences (``` blocks stay together as a single block).
    """
    lines = text.split("\n")
    blocks: list[str] = []
    current_lines: list[str] = []
    in_code_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            continue

        if in_code_fence:
            current_lines.append(line)
            continue

        if stripped == "":
            if current_lines and any(l.strip() for l in current_lines):
                while current_lines and current_lines[-1].strip() == "":
                    current_lines.pop()
                if current_lines:
                    blocks.append("\n".join(current_lines))
                current_lines = []
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        remaining = "\n".join(current_lines).strip()
        if remaining:
            blocks.append(remaining)

    if not blocks and text.strip():
        blocks.append(text.strip())

    return blocks
