/**
 * Debounced streaming preview controller.
 * Buffers incoming chunks and renders at safe HTML boundaries.
 *
 * Improvements over v1:
 * - Max-wait timeout (1s) forces render even if tags are unclosed
 * - Tracks lastGoodHTML for error recovery
 * - Skeleton placeholder support via onFirstChunk callback
 */
export class StreamingPreviewController {
  private buffer = '';
  private timer: ReturnType<typeof setTimeout> | null = null;
  private renderFn: (html: string) => void;
  private debounceMs: number;
  private maxWaitMs: number;
  private maxWaitTimer: ReturnType<typeof setTimeout> | null = null;
  private lastGoodHTML = '';
  private hasRendered = false;
  private onFirstChunk?: () => void;

  constructor(
    renderFn: (html: string) => void,
    opts: {
      debounceMs?: number;
      maxWaitMs?: number;
      onFirstChunk?: () => void;
    } = {}
  ) {
    this.renderFn = renderFn;
    this.debounceMs = opts.debounceMs ?? 250;
    this.maxWaitMs = opts.maxWaitMs ?? 1000;
    this.onFirstChunk = opts.onFirstChunk;
  }

  /** Append a chunk and schedule a debounced render. */
  append(chunk: string): void {
    const wasEmpty = this.buffer.length === 0;
    this.buffer += chunk;

    // Fire first-chunk callback (for skeleton → content transition)
    if (wasEmpty && this.onFirstChunk) {
      this.onFirstChunk();
      this.onFirstChunk = undefined; // fire once
    }

    // Start max-wait timer on first append (forces render after maxWaitMs even if unsafe)
    if (!this.maxWaitTimer) {
      this.maxWaitTimer = setTimeout(() => {
        this.maxWaitTimer = null;
        if (this.buffer && (!this.hasRendered || this.buffer !== this.lastGoodHTML)) {
          this.doRender();
        }
      }, this.maxWaitMs);
    }

    // Debounced safe-boundary render
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      if (this.isSafeToRender()) {
        this.doRender();
      }
      // If unsafe, max-wait timer will force render eventually
    }, this.debounceMs);
  }

  /** Perform a render and update tracking state. */
  private doRender(): void {
    try {
      this.renderFn(this.buffer);
      this.lastGoodHTML = this.buffer;
    } catch {
      // Render failed — lastGoodHTML stays at previous safe state
    }
    this.hasRendered = true;

    // Reset max-wait timer since we just rendered
    if (this.maxWaitTimer) {
      clearTimeout(this.maxWaitTimer);
      this.maxWaitTimer = null;
    }
  }

  /** Check if the buffer is at a safe render boundary. */
  private isSafeToRender(): boolean {
    const html = this.buffer;
    // Don't render inside unclosed <script> or <style> tags
    const lastScriptOpen = html.lastIndexOf('<script');
    const lastScriptClose = html.lastIndexOf('</script>');
    if (lastScriptOpen > lastScriptClose) return false;
    const lastStyleOpen = html.lastIndexOf('<style');
    const lastStyleClose = html.lastIndexOf('</style>');
    if (lastStyleOpen > lastStyleClose) return false;
    // Don't render inside unclosed HTML tag
    const lastLt = html.lastIndexOf('<');
    const lastGt = html.lastIndexOf('>');
    if (lastLt > lastGt) return false;
    return true;
  }

  /** Force final render with complete content. */
  flush(): string {
    this.clearTimers();
    this.renderFn(this.buffer);
    this.lastGoodHTML = this.buffer;
    this.hasRendered = true;
    return this.buffer;
  }

  /** Get the last successfully rendered HTML (for error recovery). */
  getLastGoodHTML(): string {
    return this.lastGoodHTML;
  }

  /** Get the current buffer content without rendering. */
  getBuffer(): string {
    return this.buffer;
  }

  /** Reset the controller for a new stream. */
  reset(): void {
    this.clearTimers();
    this.buffer = '';
    this.hasRendered = false;
    // Intentionally keep lastGoodHTML for recovery
  }

  /** Full cleanup — clears everything including lastGoodHTML. */
  dispose(): void {
    this.clearTimers();
    this.buffer = '';
    this.lastGoodHTML = '';
    this.hasRendered = false;
    this.onFirstChunk = undefined;
  }

  private clearTimers(): void {
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (this.maxWaitTimer) { clearTimeout(this.maxWaitTimer); this.maxWaitTimer = null; }
  }
}
