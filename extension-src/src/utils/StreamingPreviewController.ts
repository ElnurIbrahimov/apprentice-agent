/**
 * Debounced streaming preview controller.
 * Buffers incoming chunks and renders at safe boundaries.
 */
export class StreamingPreviewController {
  private buffer = '';
  private timer: ReturnType<typeof setTimeout> | null = null;
  private renderFn: (html: string) => void;
  private debounceMs: number;

  constructor(renderFn: (html: string) => void, debounceMs = 250) {
    this.renderFn = renderFn;
    this.debounceMs = debounceMs;
  }

  /** Append a chunk and schedule a debounced render. */
  append(chunk: string): void {
    this.buffer += chunk;
    if (!this.timer) {
      this.timer = setTimeout(() => {
        this.timer = null;
        if (this.isSafeToRender()) {
          this.renderFn(this.buffer);
        } else {
          // Reschedule — buffer is in an unsafe state, try again after next debounce
          this.timer = setTimeout(() => {
            this.timer = null;
            this.renderFn(this.buffer);
          }, this.debounceMs);
        }
      }, this.debounceMs);
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
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.renderFn(this.buffer);
    return this.buffer;
  }

  /** Get the current buffer content without rendering. */
  getBuffer(): string {
    return this.buffer;
  }

  /** Reset the controller for a new stream. */
  reset(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.buffer = '';
  }

  /** Clean up. */
  dispose(): void {
    this.reset();
  }
}
