# Phase 3: Code Syntax Highlighting + Code Interpreter

**Effort:** 2-3 days
**Impact:** Makes code responses beautiful + adds Python execution capability

---

## 3A. Syntax Highlighting with Shiki

### Problem
`CodeBlock.tsx` renders code with `language-{X}` class but no syntax highlighter. All code is monochrome white on dark background.

### Why Shiki Over highlight.js/Prism
- Shiki uses VS Code's TextMate grammars — same highlighting as VS Code
- Tree-shaking: only load grammars for languages you need
- Built-in dark/light themes that match our design
- Modern ESM, works great with Vite
- ~50KB for common languages (JS, TS, Python, HTML, CSS, JSON, Bash)

### Installation
```bash
cd D:/Aura/web
npm install shiki
```

### New Utility: `web/src/utils/codeHighlighter.ts`
```typescript
import { createHighlighter, type Highlighter } from 'shiki';

let highlighter: Highlighter | null = null;

const LANGUAGES = ['javascript', 'typescript', 'python', 'html', 'css', 'json', 'bash', 'sql', 'jsx', 'tsx', 'markdown', 'yaml', 'rust', 'go', 'java', 'c', 'cpp'];

export async function getHighlighter(): Promise<Highlighter> {
  if (!highlighter) {
    highlighter = await createHighlighter({
      themes: ['github-dark', 'github-light'],
      langs: LANGUAGES,
    });
  }
  return highlighter;
}

export async function highlightCode(code: string, language: string, theme: 'dark' | 'light'): Promise<string> {
  const h = await getHighlighter();
  const themeName = theme === 'dark' ? 'github-dark' : 'github-light';
  const lang = LANGUAGES.includes(language) ? language : 'text';
  return h.codeToHtml(code, { lang, theme: themeName });
}
```

### Update `CodeBlock.tsx`
```typescript
// Replace monochrome pre/code with Shiki-highlighted HTML
const [highlighted, setHighlighted] = useState('');

useEffect(() => {
  const theme = document.documentElement.classList.contains('light') ? 'light' : 'dark';
  highlightCode(code, language, theme).then(setHighlighted).catch(() => {});
}, [code, language]);

// Render:
if (highlighted) {
  return <div dangerouslySetInnerHTML={{ __html: highlighted }} className="code-block-shiki" />;
}
// Fallback to plain pre/code while loading
```

### Features
- Copy button with "Copied!" feedback (top-right corner)
- Language label badge (top-left)
- Line numbers (optional toggle)
- Word wrap toggle
- Expand/collapse for long blocks (>20 lines)
- Light theme aware (switch theme when light mode active)

---

## 3B. Code Interpreter Panel

### Problem
The extension has a full Python REPL via Pyodide (browser) and backend fallback. The web UI has nothing — users can't execute code.

### What To Build

**New tab or sub-panel within the existing "tools" tab:**

### Architecture
Same dual approach as the extension:
1. **Pyodide (browser)** — for basic Python: numpy, pandas, matplotlib, scipy, sklearn
2. **Backend fallback** — `POST /api/code/execute` for packages needing server (torch, requests, etc.)

### New Component: `web/src/components/CodeInterpreter.tsx`

Exchange-based notebook UI (same model as extension's CodePanel):

```typescript
interface Exchange {
  id: string;
  prompt: string;
  code: string;
  outputs: OutputBlock[];
  variables: VariableInfo[];
  phase: 'idle' | 'generating' | 'executing';
}
```

**Features:**
- Natural language prompt → AI generates Python → auto-executes
- Direct code input mode (toggle)
- Output types: stdout, images (matplotlib base64), HTML tables (DataFrames), errors
- "Fix Error" button sends error + code back to LLM
- Variable inspector (collapsible)
- CSV file upload → auto-analyze
- Session state persists across exchanges
- Browser/Server toggle (same as extension)

### Pyodide Worker
Port `extension-src/src/workers/pyodide-worker.ts` and `PyodideExecutor.ts` to the web project. The worker loads Pyodide v0.27.4 from CDN, initializes micropip, and handles execute/reset/install messages.

Since web workers work identically in regular web apps and extensions, the code can be copied with minimal changes (remove `chrome.*` references, use standard web worker API).

### Integration Options

**Option A: New "Code" tab (7th tab)**
- Add to the tab bar alongside Chat/Monitoring/Tools/Advanced/Activity/Settings
- Full-height panel with exchange list + input

**Option B: Sub-panel within Tools tab**
- Add as a section in ToolsPanel
- Collapsible, expands to full height on use

**Option C: Inline in chat**
- When user asks "run this code" or pastes Python, automatically render a code execution cell in the chat
- Most seamless but more complex

**Recommendation:** Option A (new tab) — cleanest, most discoverable.

### Quick Actions
Same as extension: Analyze CSV, Create Chart, Solve Math, Run Python

---

## 3C. Inline Code Execution in Chat

### Problem
When Aura responds with a code block, the user can't run it without copy-pasting elsewhere.

### What To Build
Add a "Run" button on Python/JavaScript code blocks in assistant messages:

- Detect `python` language code blocks in assistant responses
- Show a small "Run" button (play icon) next to the code block
- On click: send to Pyodide (if Python) or eval in sandbox (if JS)
- Show output inline below the code block (stdout, images, errors)
- "Copy & Open in Code Interpreter" button for complex execution

This is a lightweight version — full Code Interpreter is in the tab, this is just a convenience.

---

## Definition of Done — Phase 3
- [ ] All code blocks have syntax highlighting (Shiki, 16+ languages)
- [ ] Highlighting respects dark/light theme
- [ ] Code blocks have: copy button, language badge, expand/collapse for long blocks
- [ ] Code Interpreter tab with full exchange-based notebook
- [ ] Pyodide worker runs Python in browser with numpy/pandas/matplotlib
- [ ] Backend fallback for server-only packages
- [ ] AI code generation from natural language prompts
- [ ] Variable inspector, CSV upload, Fix Error button
- [ ] "Run" button on Python code blocks in chat messages
- [ ] Inline output rendering (stdout, images, tables, errors)
