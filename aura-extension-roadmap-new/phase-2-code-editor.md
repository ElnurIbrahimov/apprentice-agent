# Phase 2: Code Editor Upgrade — CodeMirror 6 + Diff View

**Effort:** 2-3 days
**Impact:** Transforms every code interaction from "notepad" to "VS Code lite"
**Depends on:** Nothing (can run in parallel with Phase 1)

---

## 2A. Replace textarea with CodeMirror 6

### Problem
All three creation panels (Artifacts, WebCreator, Code) use a plain `<textarea>` with a syntax-highlighted `<pre>` overlay from `utils/highlighter.ts`. This means:
- No autocomplete or intellisense
- No bracket matching or auto-closing
- No code folding
- No multi-cursor editing
- No find/replace
- No inline error markers
- No minimap
- No proper indentation handling
- No undo/redo at the editor level (only version-level via useVersionHistory)

Every competitor (v0, Bolt, Claude Artifacts, Replit) uses a real code editor.

### Why CodeMirror 6 Over Monaco
- **Size:** CodeMirror 6 is ~50KB gzipped. Monaco is ~2MB+ gzipped.
- **Performance:** CM6 is built for performance-constrained environments (perfect for extension sidebar).
- **Extensibility:** CM6's extension system is more composable.
- **Mobile:** CM6 works on mobile/touch. Monaco doesn't.
- Monaco is overkill for a sidebar panel. CM6 is the right tool.

### Installation

```bash
cd D:/Aura/extension-src
npm install @codemirror/state @codemirror/view @codemirror/commands \
  @codemirror/language @codemirror/autocomplete @codemirror/search \
  @codemirror/lint @codemirror/merge \
  @codemirror/lang-html @codemirror/lang-javascript \
  @codemirror/lang-python @codemirror/lang-css @codemirror/lang-json \
  @codemirror/lang-markdown \
  @codemirror/theme-one-dark
```

### New Component: `CodeEditor.tsx`

**File:** `extension-src/src/components/CodeEditor.tsx`

```typescript
import { useRef, useEffect } from 'react';
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, highlightActiveLine, 
         highlightActiveLineGutter, drawSelection } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { syntaxHighlighting, defaultHighlightStyle, 
         bracketMatching, foldGutter, indentOnInput } from '@codemirror/language';
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';
import { searchKeymap, highlightSelectionMatches } from '@codemirror/search';
import { lintGutter } from '@codemirror/lint';
import { oneDark } from '@codemirror/theme-one-dark';

// Language imports
import { html } from '@codemirror/lang-html';
import { javascript } from '@codemirror/lang-javascript';
import { python } from '@codemirror/lang-python';
import { css } from '@codemirror/lang-css';
import { json } from '@codemirror/lang-json';
import { markdown } from '@codemirror/lang-markdown';

interface CodeEditorProps {
  code: string;
  language: 'html' | 'javascript' | 'typescript' | 'jsx' | 'tsx' | 'python' | 'css' | 'json' | 'markdown' | 'svg';
  onChange?: (code: string) => void;
  readOnly?: boolean;
  height?: string;        // CSS height, default '100%'
  lineWrapping?: boolean; // default true
}

// Language extension resolver
function getLanguageExtension(lang: string) {
  switch (lang) {
    case 'html': case 'svg': return html();
    case 'javascript': case 'jsx': return javascript({ jsx: true });
    case 'typescript': case 'tsx': return javascript({ jsx: true, typescript: true });
    case 'python': return python();
    case 'css': return css();
    case 'json': return json();
    case 'markdown': return markdown();
    default: return html();
  }
}
```

Key features to include:
- Dark theme matching Aura's `--bg` / `--tx` CSS variables (extend `oneDark` or build custom theme)
- `EditorView.updateListener` dispatches `onChange` on doc changes
- `EditorState.readOnly` facet for preview-only mode
- Smooth resizing via `EditorView.theme({ '&': { height } })`
- Tab size: 2 spaces

### Integration Points

**ArtifactsPanel** (`panels/ArtifactsPanel.tsx`):
- Replace the `<textarea>` + `<pre>` code view with `<CodeEditor>`
- Language auto-detected from artifact type
- `onChange` updates the artifact code and triggers iframe re-render (debounced 500ms)
- Read-only during streaming generation

**WebCreatorPanel** (`panels/WebCreatorPanel.tsx`):
- Replace code view textarea with `<CodeEditor language="html">`
- Same onChange → re-render pattern
- Read-only during streaming

**CodePanel** (`panels/CodePanel.tsx`):
- Replace each exchange's code textarea with `<CodeEditor language="python">`
- Smaller height per cell (auto-resize based on line count, min 3 lines, max 20)
- Editable always (user can modify code before re-running)

### Remove Old Highlighter
- Delete `extension-src/src/utils/highlighter.ts` after migration
- Remove all `highlightCode()` and `detectLanguage()` calls
- Remove the `<pre>` overlay pattern from all panels

---

## 2B. Diff View for AI Changes

### Problem
When the AI regenerates code, it replaces everything. The user can't see what changed. No accept/reject per change. This is a major UX gap vs. Cursor, Claude Artifacts, and Copilot.

### What To Build

**Using `@codemirror/merge`:**

**New component:** `extension-src/src/components/DiffEditor.tsx`

```typescript
import { MergeView } from '@codemirror/merge';

interface DiffEditorProps {
  original: string;      // previous version
  modified: string;      // new AI-generated version
  language: string;
  onAccept: (code: string) => void;
  onReject: () => void;
}
```

Features:
- Side-by-side diff view (or unified if panel is narrow)
- Green highlights for additions, red for deletions
- "Accept All" and "Reject All" buttons in toolbar
- Optional: per-hunk accept/reject (stretch goal)

### Integration Flow

1. AI finishes generating new code
2. Instead of immediately replacing, check if there's existing code
3. If existing code differs:
   - Show `<DiffEditor original={oldCode} modified={newCode} />`
   - User reviews and clicks "Accept" or "Reject"
   - If "Accept": update code + push to version history
   - If "Reject": discard new code, keep current
4. If no existing code (first generation): apply directly, no diff
5. Settings toggle: "Show diff before applying" (default ON)
6. For streaming: diff is shown AFTER generation completes (not during)

### Auto-Accept Mode
- For small changes (<5 lines changed): auto-accept with a brief flash showing what changed
- For large changes (>30% of code changed): always show diff
- Configurable thresholds in settings

---

## 2C. Inline Error Markers

### Problem
When iframe reports a JS error with a line number, the user has to manually find that line in the code. The error should be shown inline in the editor.

### What To Build

Using `@codemirror/lint`:

```typescript
import { linter, Diagnostic } from '@codemirror/lint';

// When iframe reports an error:
function setEditorDiagnostics(view: EditorView, errors: IframeError[]) {
  const diagnostics: Diagnostic[] = errors.map(err => ({
    from: view.state.doc.line(err.lineno).from,
    to: view.state.doc.line(err.lineno).to,
    severity: 'error',
    message: err.message,
    source: 'Runtime'
  }));
  // Set diagnostics via the lint extension
}
```

- Red underline on the error line
- Hover shows error message
- Click jumps to the error
- Clear diagnostics on next successful render

---

## Definition of Done — Phase 2
- [ ] CodeMirror 6 replaces all textarea+pre code editors
- [ ] Syntax highlighting for HTML, JS/TS/JSX/TSX, Python, CSS, JSON, Markdown
- [ ] Bracket matching, auto-closing, code folding, find/replace all work
- [ ] Theme matches Aura's dark theme CSS variables
- [ ] Diff view shows before/after when AI regenerates code
- [ ] Accept/Reject buttons on diff view
- [ ] Inline error markers from iframe runtime errors
- [ ] Old `highlighter.ts` utility deleted
- [ ] Editor performance tested: no lag with 1000+ line files
