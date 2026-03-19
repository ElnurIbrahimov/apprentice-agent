# AI Chat Web UI Patterns Research (March 2026)

Research into the state-of-the-art AI chat interfaces — ChatGPT, Claude.ai, Gemini, Perplexity, v0.dev/bolt.new — distilled into the 20 most impactful patterns Aura should implement.

---

## Per-Product Breakdown

### 1. ChatGPT (OpenAI)

**Top 10 patterns that make it feel polished:**

1. **Collapsible reasoning display** — o1/o3 models show flashing, changing text labels during thinking, then auto-collapse reasoning when done. Users manually expand for details. Reduces cognitive load while providing transparency.
2. **Canvas split-pane workspace** — Two-pane layout: conversation on left, editable document/code on right. Inline editing with highlight-to-edit. Supports `/canvas` command trigger.
3. **Model picker in composer bar** — Dropdown adjacent to the "new chat" button. Shows model name as a pill/badge. Quick switch without leaving the conversation.
4. **Sidebar with pinned conversations** — Collapsible sidebar that can coexist with Canvas if screen width allows. New composer bar fades content underneath as you scroll.
5. **Streaming with cursor** — Token-by-token text rendering with a 2px blinking cursor (500ms animation cycle). Cursor disappears on stream completion.
6. **Generative UI (ChatGPT 5)** — Agents can emit UI components, not just text. Code execution results render inline as interactive elements.
7. **Suggestion chips on empty state** — Pre-written prompt suggestions categorized by task type (write, analyze, code, create) displayed as clickable cards.
8. **Code blocks with copy + language badge** — Syntax-highlighted code blocks with language label, one-click copy button, and "Run in Canvas" option.
9. **File upload with inline preview** — Drag-and-drop with thumbnail preview for images, icon + filename for documents. Files appear above the composer.
10. **Image generation inline display** — DALL-E outputs render as expandable cards within the message stream. Click to enlarge, download, or iterate.

### 2. Claude.ai (Anthropic)

**Top 10 patterns that make it different:**

1. **Artifacts side panel** — Generated content (code, documents, SVGs, diagrams, interactive HTML/JS) renders in a dedicated side panel that opens alongside chat. The artifact is a runtime, not just a code display — interactive elements work live.
2. **Extended Thinking display** — Animated icon + dynamic text label + time counter as progress indicators. Reasoning is expandable but collapsed by default. Separately scrollable, structured with bullets. Best balance of transparency vs. overload among all products.
3. **Generative UI (on-demand visualization)** — Claude generates interactive charts, calculators, animations inline. Ephemeral within conversation, changing as the conversation evolves.
4. **Project system** — Conversations grouped under projects with shared context (project knowledge, custom instructions). Persistent workspace across sessions.
5. **Artifact type detection** — Claude auto-detects when content should be an artifact vs. inline text. Code, SVGs, Mermaid diagrams, HTML apps all route to the appropriate renderer.
6. **500M+ artifacts ecosystem** — Public artifact sharing via URLs. Artifact catalog for discovery. Community-created interactive tools.
7. **Conversation forking** — Edit a previous message to create a branch. Preserves original conversation while exploring alternatives.
8. **Clean message rendering** — Generous whitespace, clear visual hierarchy between user (right-aligned, colored) and assistant (left-aligned, neutral). Markdown renders beautifully with proper typography.
9. **Keyboard-first design** — Cmd+K for commands, extensive keyboard shortcuts for navigation, conversation switching, and settings.
10. **Minimal chrome** — Very little UI ornamentation. Content-first design. Dark mode with near-black base (#0D0D14) that makes content "glow."

### 3. Gemini (Google)

**Top 10 patterns:**

1. **Canvas with Deep Research integration** — Generate a Deep Research report, then one-click transform into web page, infographic, quiz, or audio overview via "Create" button.
2. **Gradient visual language** — Google's Gemini branding uses animated gradients that "gently guide users into the collaborative world." Dynamic, fluid expression of AI intelligence rather than static icons.
3. **True black dark mode** — AMOLED-friendly true black theme (#000000 base). System-aware switching that respects OS preference.
4. **"My Stuff" hub** — Navigation drawer section that acts as a central hub for all creative output. Shows preview of last 3 images/videos/Canvas works.
5. **Grounding with Knowledge Graph** — Responses tied to Google's Knowledge Graph for source verification. Reduces hallucinations. Inline verification indicators.
6. **Inline suggestions** — Contextual suggestions that appear within the conversation flow, not just at the end. "Broaden scope", "Recent data", "Counter-arguments" style chips.
7. **Multi-modal input bar** — Persistent microphone button, camera input, file upload, and text all in one composer bar. Visual waveform animation while listening.
8. **Canvas code visualization** — Algorithms come to life through animations. Complex concepts turned into interactive visual demonstrations.
9. **Shareable Canvas links** — One-click sharing of Canvas creations. Each gets a permanent URL (g.co/gemini/share/...).
10. **Structured response cards** — Responses formatted as expandable cards with clear section headers, not just flowing text.

### 4. Perplexity

**Top 10 patterns:**

1. **Citation-forward answers** — Numbered inline citations with favicon + title metadata for each source. Citations generated through tightly coupled retrieval pipeline, not post-processed.
2. **Sources panel** — Dedicated sources section at top of answer showing all referenced URLs as expandable cards with favicons, titles, and domains.
3. **Follow-up suggestion chips** — Predicted follow-up questions displayed at the end of every answer. Based on the principle that "not all users are great at asking follow-up questions."
4. **Focus modes** — Selectable search scopes: All, Academic, Writing, Wolfram Alpha, YouTube, Reddit. Changes the retrieval pipeline.
5. **Copilot sidebar** — Smart next-step suggestions after each query: "Broaden scope", "Recent data", "Counter-arguments", "Compare expert opinions."
6. **Answer-then-sources flow** — Answer text first, then expandable source verification. Optimized for the insight that "people want information as fast as possible and want to trust that information."
7. **Conversational prompt language** — "Ask a follow-up..." instead of "Enter your query." Uses the mental model of human conversation.
8. **Related questions grid** — Grid of clickable related questions below each answer. Not just follow-ups but adjacent topics.
9. **Media integration** — Images, videos, and maps inline within answers when relevant. Not decorative — sourced and captioned.
10. **Thread-based research** — Each query thread maintains context. Follow-ups refine within the same research context.

### 5. v0.dev / bolt.new / Lovable

**Top 10 patterns:**

1. **Split pane: chat + live preview** — Left pane is the conversation/prompt, right pane shows live rendered output updating in real-time as code generates.
2. **bolt.new's VS Code-like environment** — Full Node.js in browser via WebContainers. Editable code on left, live preview on right. Mimics VS Code look and feel.
3. **v0.dev's image-to-code** — Upload a screenshot or Figma design, get production React + Tailwind + shadcn/ui components. Image appears alongside generated code.
4. **Real-time code streaming into preview** — As tokens generate, the preview updates. You see the UI being "painted" in real-time.
5. **Component iteration in conversation** — "Make the button bigger", "Add dark mode" — conversational refinement of visual output with instant preview.
6. **Lovable's full-stack from chat** — Single chat interface provisions Supabase, configures auth, sets up RLS, and wires frontend. All progress shown inline.
7. **File tree sidebar** — Generated project structure shown as an interactive file tree. Click to view/edit any file.
8. **Framework/stack selector** — Choose React, Vue, Svelte, Next.js, Astro before generating. Affects both code output and preview runtime.
9. **Deploy button** — One-click deployment from the generation interface. Lovable deploys to its own hosting, v0 to Vercel.
10. **Version history** — Each iteration creates a version. Scroll back through visual diffs of the generated output.

---

## THE 20 MOST IMPACTFUL PATTERNS TO STEAL FOR AURA

### Pattern 1: Glassmorphic Dark AI Panels

**Seen in:** ChatGPT, Claude, Gemini (all 2025-2026 AI interfaces)
**Impact:** Sets the visual foundation. 82% of users prefer dark mode for AI-heavy apps.

```css
/* Base surface */
.aura-surface {
  background: #0D0D14;
  color: #E8E8ED;
}

/* AI response panel */
.ai-panel {
  background: rgba(255, 255, 255, 0.06);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 16px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
}

/* User message */
.user-message {
  background: rgba(139, 92, 246, 0.15);
  border: 1px solid rgba(139, 92, 246, 0.25);
  border-radius: 16px 16px 4px 16px;
}
```

**React:** Apply as Tailwind classes: `bg-white/[0.06] backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl`

---

### Pattern 2: Token-by-Token Streaming with Blinking Cursor

**Seen in:** ChatGPT, Claude, all major products
**Impact:** Reduces perceived wait time by 55-70% even when total generation time is identical.

```tsx
// React implementation using FlowToken-style approach
function StreamingText({ tokens }: { tokens: string[] }) {
  return (
    <div className="relative">
      {tokens.map((token, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, filter: 'blur(4px)' }}
          animate={{ opacity: 1, filter: 'blur(0px)' }}
          transition={{ duration: 0.15, delay: i * 0.02 }}
        >
          {token}
        </motion.span>
      ))}
      <motion.span
        className="inline-block w-[2px] h-[1.1em] bg-purple-400 ml-0.5 align-text-bottom"
        animate={{ opacity: [1, 0] }}
        transition={{ duration: 0.5, repeat: Infinity, repeatType: 'reverse' }}
      />
    </div>
  );
}
```

**Key detail:** Use `blur(4px) -> blur(0px)` fade-in per token for a premium feel. Cursor is 2px wide, matches accent color, 500ms blink cycle. Remove cursor when stream completes.

---

### Pattern 3: Collapsible Thinking/Reasoning Display

**Seen in:** Claude (best execution), ChatGPT o1/o3, Grok, DeepSeek
**Impact:** Builds trust through transparency without overwhelming users. Claude's approach wins: minimal by default, expandable on demand.

```tsx
// Using Radix Collapsible (same approach as Vercel AI Elements)
import * as Collapsible from '@radix-ui/react-collapsible';

function ThinkingDisplay({ steps, isThinking, elapsed }: Props) {
  return (
    <Collapsible.Root defaultOpen={false}>
      <Collapsible.Trigger className="flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors">
        {isThinking ? (
          <>
            <motion.div
              className="w-4 h-4 rounded-full border-2 border-purple-400 border-t-transparent"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            />
            <span className="animate-pulse">Thinking...</span>
            <span className="font-mono text-xs">{elapsed}s</span>
          </>
        ) : (
          <>
            <CheckCircle className="w-4 h-4 text-green-400" />
            <span>Thought for {elapsed}s</span>
            <ChevronDown className="w-3 h-3" />
          </>
        )}
      </Collapsible.Trigger>
      <Collapsible.Content className="overflow-hidden data-[state=open]:animate-slideDown data-[state=closed]:animate-slideUp">
        <div className="mt-2 pl-6 border-l-2 border-zinc-700 text-sm text-zinc-400 max-h-60 overflow-y-auto">
          {steps.map((step, i) => (
            <div key={i} className="flex items-start gap-2 mb-1">
              <span className="text-zinc-600">•</span>
              <span>{step}</span>
            </div>
          ))}
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
```

**Key UX rule:** "More transparency ≠ Better UX." Show right reasoning at the right time.

---

### Pattern 4: Artifacts/Canvas Side Panel

**Seen in:** Claude Artifacts, ChatGPT Canvas, Gemini Canvas
**Impact:** Transforms chat from Q&A into a collaborative workspace. Claude has 500M+ artifacts created.

```tsx
function SplitLayout() {
  const [artifactOpen, setArtifactOpen] = useState(false);

  return (
    <div className="flex h-screen">
      {/* Chat pane - responsive width */}
      <div className={cn(
        "flex flex-col transition-all duration-300 ease-in-out",
        artifactOpen ? "w-1/2" : "w-full max-w-3xl mx-auto"
      )}>
        <ChatMessages />
        <Composer />
      </div>

      {/* Artifact pane - slides in from right */}
      <AnimatePresence>
        {artifactOpen && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: '50%', opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            className="border-l border-zinc-800 bg-zinc-950 overflow-hidden"
          >
            <ArtifactHeader onClose={() => setArtifactOpen(false)} />
            <ArtifactRenderer type={artifactType} content={artifactContent} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

**Key distinction:** Claude = execution environment (code runs live). ChatGPT Canvas = editing environment (collaborative document editing). Aura should support both modes.

---

### Pattern 5: Inline Citation Cards with Favicons

**Seen in:** Perplexity (best execution), Gemini, ChatGPT with browsing
**Impact:** Instantly builds trust. Perplexity's entire product is built on this pattern.

```tsx
function InlineCitation({ index, source }: { index: number; source: Source }) {
  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <button className="inline-flex items-center justify-center w-5 h-5 rounded-full
          bg-zinc-700 text-[10px] font-medium text-zinc-300 hover:bg-purple-600
          hover:text-white transition-colors align-super ml-0.5 cursor-pointer">
          {index}
        </button>
      </HoverCardTrigger>
      <HoverCardContent className="w-72 bg-zinc-900 border-zinc-700">
        <div className="flex items-start gap-3">
          <img src={source.favicon} alt="" className="w-4 h-4 mt-1 rounded-sm" />
          <div>
            <p className="text-sm font-medium text-zinc-100 line-clamp-2">{source.title}</p>
            <p className="text-xs text-zinc-500 mt-0.5">{source.domain}</p>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

function SourcesBar({ sources }: { sources: Source[] }) {
  return (
    <div className="flex gap-2 mb-3 overflow-x-auto pb-1">
      {sources.map((s, i) => (
        <a key={i} href={s.url} target="_blank" rel="noopener"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-zinc-800
            border border-zinc-700 text-xs text-zinc-300 hover:border-purple-500
            transition-colors shrink-0">
          <img src={s.favicon} alt="" className="w-3.5 h-3.5 rounded-sm" />
          <span className="truncate max-w-[120px]">{s.domain}</span>
        </a>
      ))}
    </div>
  );
}
```

---

### Pattern 6: Skeleton Loading with Shimmer

**Seen in:** ChatGPT, Claude, most 2026 AI apps
**Impact:** Reduces perceived load time by 40% vs. blank panels with spinners.

```tsx
function MessageSkeleton() {
  return (
    <div className="space-y-3 animate-pulse">
      {[100, 92, 78, 85, 60].map((width, i) => (
        <div
          key={i}
          className="h-4 rounded-md bg-gradient-to-r from-zinc-800 via-zinc-700 to-zinc-800
            bg-[length:200%_100%] animate-shimmer"
          style={{ width: `${width}%` }}
        />
      ))}
    </div>
  );
}

/* In tailwind.config.js */
// animation: { shimmer: 'shimmer 1.5s ease-in-out infinite' }
// keyframes: { shimmer: { '0%': { backgroundPosition: '200% 0' }, '100%': { backgroundPosition: '-200% 0' } } }
```

**Key detail:** 3-5 lines, decreasing widths mimicking natural text variation. Grey shimmer gradient, not a spinner.

---

### Pattern 7: Smart Empty State with Suggestion Grid

**Seen in:** ChatGPT (best), Claude, Gemini
**Impact:** Reduces blank-page anxiety. Guides new users. Shows capabilities.

```tsx
function EmptyState() {
  const suggestions = [
    { icon: <PenLine />, label: 'Write', examples: ['Draft an email...', 'Write a story...'] },
    { icon: <Code />,    label: 'Code',  examples: ['Build a React component...', 'Debug this...'] },
    { icon: <Search />,  label: 'Research', examples: ['Compare X vs Y...', 'Explain...'] },
    { icon: <Sparkles />, label: 'Create', examples: ['Generate an image...', 'Design a...'] },
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full max-w-2xl mx-auto px-4">
      {/* Aura avatar with breathing animation */}
      <AuraBreathingAvatar size="lg" className="mb-6" />
      <h1 className="text-2xl font-semibold text-zinc-100 mb-1">What can I help with?</h1>
      <p className="text-sm text-zinc-500 mb-8">Ask anything or pick a suggestion</p>

      <div className="grid grid-cols-2 gap-3 w-full">
        {suggestions.map((cat) => (
          <button key={cat.label}
            className="flex items-start gap-3 p-4 rounded-xl bg-zinc-800/50
              border border-zinc-700/50 hover:border-purple-500/50 hover:bg-zinc-800
              transition-all duration-200 text-left group">
            <div className="p-2 rounded-lg bg-zinc-700/50 text-zinc-400
              group-hover:text-purple-400 group-hover:bg-purple-500/10 transition-colors">
              {cat.icon}
            </div>
            <div>
              <p className="text-sm font-medium text-zinc-200">{cat.label}</p>
              <p className="text-xs text-zinc-500 mt-0.5">{cat.examples[0]}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
```

---

### Pattern 8: Model Picker Pill

**Seen in:** ChatGPT, Claude, all multi-model interfaces
**Impact:** Users need to know which model they're talking to. Pill format is compact and always visible.

```tsx
function ModelPicker({ selected, models, onChange }: Props) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-full
          bg-zinc-800 border border-zinc-700 text-sm text-zinc-300
          hover:border-zinc-500 transition-colors">
          <span className="w-2 h-2 rounded-full bg-green-400" />
          <span className="font-medium">{selected.shortName}</span>
          <ChevronDown className="w-3 h-3 text-zinc-500" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-72 p-1 bg-zinc-900 border-zinc-700">
        {models.map((model) => (
          <button key={model.id} onClick={() => onChange(model)}
            className={cn(
              "w-full flex items-start gap-3 p-3 rounded-lg text-left transition-colors",
              selected.id === model.id ? "bg-purple-500/10" : "hover:bg-zinc-800"
            )}>
            <div>
              <p className="text-sm font-medium text-zinc-200">{model.name}</p>
              <p className="text-xs text-zinc-500">{model.description}</p>
            </div>
            {model.badge && (
              <span className="ml-auto px-1.5 py-0.5 rounded text-[10px]
                font-medium bg-purple-500/20 text-purple-300">{model.badge}</span>
            )}
          </button>
        ))}
      </PopoverContent>
    </Popover>
  );
}
```

---

### Pattern 9: Tool Call Visualization

**Seen in:** ChatGPT (code interpreter), Claude (tool use), Perplexity (search)
**Impact:** Makes AI actions visible and trustworthy. Critical for agent interfaces like Aura.

```tsx
function ToolCallCard({ tool }: { tool: ToolCall }) {
  const [expanded, setExpanded] = useState(false);
  const statusColors = {
    running: 'border-yellow-500/50 bg-yellow-500/5',
    complete: 'border-green-500/50 bg-green-500/5',
    error: 'border-red-500/50 bg-red-500/5',
  };

  return (
    <div className={cn("rounded-xl border p-3 my-2 transition-colors", statusColors[tool.status])}>
      <button onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left">
        {tool.status === 'running' ? (
          <Loader2 className="w-4 h-4 animate-spin text-yellow-400" />
        ) : tool.status === 'complete' ? (
          <CheckCircle2 className="w-4 h-4 text-green-400" />
        ) : (
          <XCircle className="w-4 h-4 text-red-400" />
        )}
        <span className="text-sm font-mono text-zinc-300">{tool.name}</span>
        <span className="text-xs text-zinc-500 ml-auto">{tool.duration}ms</span>
        <ChevronDown className={cn("w-3 h-3 text-zinc-500 transition-transform",
          expanded && "rotate-180")} />
      </button>
      {expanded && (
        <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }}
          className="mt-2 pt-2 border-t border-zinc-700/50 overflow-hidden">
          <pre className="text-xs font-mono text-zinc-400 overflow-x-auto">
            {JSON.stringify(tool.args, null, 2)}
          </pre>
          {tool.result && (
            <pre className="mt-2 text-xs font-mono text-zinc-300 overflow-x-auto">
              {typeof tool.result === 'string' ? tool.result : JSON.stringify(tool.result, null, 2)}
            </pre>
          )}
        </motion.div>
      )}
    </div>
  );
}
```

---

### Pattern 10: Follow-Up Suggestion Chips

**Seen in:** Perplexity (best), Gemini, ChatGPT
**Impact:** Keeps conversation flowing. Compensates for users who don't know what to ask next.

```tsx
function FollowUpChips({ suggestions, onSelect }: Props) {
  return (
    <div className="flex flex-wrap gap-2 mt-4">
      {suggestions.map((s, i) => (
        <motion.button
          key={i}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 * i, duration: 0.2 }}
          onClick={() => onSelect(s)}
          className="px-3 py-1.5 rounded-full border border-zinc-700 bg-zinc-800/50
            text-sm text-zinc-300 hover:border-purple-500/50 hover:text-purple-300
            hover:bg-purple-500/5 transition-all duration-200"
        >
          {s}
        </motion.button>
      ))}
    </div>
  );
}
```

**Key detail:** Staggered fade-in animation (100ms delay per chip) from Perplexity. Makes them feel dynamically generated, not static.

---

### Pattern 11: Message Transition Animations

**Seen in:** All top products, FlowToken library
**Impact:** Makes the interface feel alive. 100-300ms micro-animations for all state changes.

```tsx
// Motion (formerly Framer Motion) for message entry
function ChatMessage({ message }: { message: Message }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
      className={cn("flex gap-3 px-4 py-3",
        message.role === 'user' ? "justify-end" : "justify-start"
      )}
    >
      {message.role === 'assistant' && <AuraAvatar size="sm" />}
      <div className={cn("max-w-[80%] rounded-2xl px-4 py-3",
        message.role === 'user'
          ? "bg-purple-600/20 border border-purple-500/30"
          : "bg-zinc-800/80 border border-zinc-700/50"
      )}>
        <MarkdownRenderer content={message.content} />
      </div>
    </motion.div>
  );
}
```

**Key details:**
- Entry: `opacity 0->1, y 12->0, scale 0.98->1` over 250ms with Material ease curve
- Height expansion: `auto-animate` as streaming content arrives
- Colour transitions: 200ms when confidence or status updates
- Dismissal: gentle `opacity 1->0` fade-out

---

### Pattern 12: Code Block with Syntax Highlighting and Actions

**Seen in:** ChatGPT, Claude, all code-capable products
**Impact:** Code is the #1 non-text output. Must be excellent.

```tsx
function CodeBlock({ code, language, filename }: Props) {
  const [copied, setCopied] = useState(false);

  return (
    <div className="rounded-xl overflow-hidden border border-zinc-700/50 my-3">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-800/80 border-b border-zinc-700/50">
        <div className="flex items-center gap-2">
          {filename && <span className="text-xs text-zinc-400 font-mono">{filename}</span>}
          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium
            bg-zinc-700 text-zinc-400 uppercase">{language}</span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => { navigator.clipboard.writeText(code); setCopied(true); }}
            className="p-1.5 rounded-md hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors">
            {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
          {/* Optional: Run in sandbox, open in artifact */}
        </div>
      </div>
      {/* Code content */}
      <div className="p-4 bg-zinc-900 overflow-x-auto">
        <SyntaxHighlighter language={language} theme={customDarkTheme}>
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
```

---

### Pattern 13: Typing/Processing State Indicators

**Seen in:** All products, but with different approaches
**Impact:** The "elevator mirror effect" — well-designed progress indicators reduce perceived wait time.

```tsx
function ProcessingIndicator({ stage }: { stage: string }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3">
      <AuraAvatar size="sm" breathing />
      <div className="flex items-center gap-2">
        {/* Animated dots */}
        <div className="flex gap-1">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-purple-400"
              animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1, 0.8] }}
              transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
            />
          ))}
        </div>
        {/* Dynamic stage label */}
        <motion.span
          key={stage}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-sm text-zinc-400"
        >
          {stage}
        </motion.span>
      </div>
    </div>
  );
}
```

**Stage labels should change:** "Thinking..." -> "Searching the web..." -> "Analyzing results..." -> "Writing response..."

---

### Pattern 14: Responsive Composer Bar

**Seen in:** ChatGPT (best), Claude, all products
**Impact:** The composer is the most-interacted element. Must be flawless.

```tsx
function Composer() {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    }
  }, [value]);

  return (
    <div className="sticky bottom-0 px-4 pb-4 pt-2 bg-gradient-to-t from-zinc-950 via-zinc-950/95 to-transparent">
      <div className="max-w-3xl mx-auto relative rounded-2xl border border-zinc-700
        bg-zinc-800/80 backdrop-blur-xl focus-within:border-purple-500/50
        focus-within:ring-1 focus-within:ring-purple-500/25 transition-all">

        {/* Attachment preview row */}
        <AttachmentPreview />

        {/* Input area */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Message Aura..."
          rows={1}
          className="w-full resize-none bg-transparent px-4 pt-3 pb-1 text-zinc-100
            placeholder-zinc-500 focus:outline-none text-sm leading-relaxed"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
          }}
        />

        {/* Bottom toolbar */}
        <div className="flex items-center justify-between px-3 pb-2">
          <div className="flex items-center gap-1">
            <ModelPicker />
            <button className="p-1.5 rounded-lg hover:bg-zinc-700 text-zinc-400"><Paperclip className="w-4 h-4" /></button>
            <button className="p-1.5 rounded-lg hover:bg-zinc-700 text-zinc-400"><Mic className="w-4 h-4" /></button>
          </div>
          <button
            disabled={!value.trim()}
            className="p-2 rounded-xl bg-purple-600 text-white disabled:opacity-30
              disabled:cursor-not-allowed hover:bg-purple-500 transition-colors"
          >
            <ArrowUp className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Key details:** Auto-growing textarea (max 200px), gradient fade at top, focus ring animation, Shift+Enter for newlines, model picker integrated into toolbar.

---

### Pattern 15: Collapsible Sidebar with Conversation History

**Seen in:** ChatGPT, Claude, Gemini
**Impact:** Navigation without leaving the current conversation.

```tsx
function Sidebar({ isOpen, onToggle }: Props) {
  return (
    <>
      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: isOpen ? 280 : 0 }}
        transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
        className="h-screen overflow-hidden border-r border-zinc-800 bg-zinc-950 flex-shrink-0"
      >
        <div className="w-[280px] h-full flex flex-col">
          {/* New chat button */}
          <div className="p-3">
            <button className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl
              border border-zinc-700/50 hover:bg-zinc-800 text-sm text-zinc-300 transition-colors">
              <Plus className="w-4 h-4" /> New chat
            </button>
          </div>

          {/* Search */}
          <div className="px-3 pb-2">
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800/50 border border-zinc-700/30">
              <Search className="w-3.5 h-3.5 text-zinc-500" />
              <input placeholder="Search conversations..." className="bg-transparent text-xs text-zinc-300 placeholder-zinc-600 outline-none w-full" />
            </div>
          </div>

          {/* Grouped conversation list */}
          <div className="flex-1 overflow-y-auto px-2">
            {groups.map(group => (
              <div key={group.label}>
                <p className="px-3 py-2 text-[11px] font-medium text-zinc-600 uppercase tracking-wider">{group.label}</p>
                {group.conversations.map(conv => (
                  <ConversationItem key={conv.id} conversation={conv} />
                ))}
              </div>
            ))}
          </div>
        </div>
      </motion.aside>

      {/* Toggle button - always visible */}
      <button onClick={onToggle}
        className="fixed top-3 left-3 z-50 p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 transition-colors">
        <PanelLeft className="w-4 h-4" />
      </button>
    </>
  );
}
```

**Key detail:** Group conversations by time (Today, Yesterday, Previous 7 days, Older). ChatGPT now groups by topic as well.

---

### Pattern 16: Confidence & Source Indicators

**Seen in:** Perplexity (citations), emerging in all products
**Impact:** Critical for trust. Color-coded confidence borders + source links.

```css
/* Confidence borders on response cards */
.confidence-high   { border-left: 3px solid #22c55e; } /* green */
.confidence-medium { border-left: 3px solid #f59e0b; } /* amber */
.confidence-low    { border-left: 3px solid #ef4444; } /* red */
```

```tsx
function ConfidenceBadge({ score }: { score: number }) {
  const color = score > 0.85 ? 'text-green-400 bg-green-400/10' :
                score > 0.6  ? 'text-amber-400 bg-amber-400/10' :
                               'text-red-400 bg-red-400/10';
  return (
    <span className={cn("px-1.5 py-0.5 rounded text-[10px] font-mono", color)}>
      {Math.round(score * 100)}%
    </span>
  );
}
```

---

### Pattern 17: Micro-Animation State System

**Seen in:** All top products. 2026 standard.
**Impact:** Replace text labels with visual signals. Reduces cognitive load.

```css
/* State animation timings */
:root {
  --aura-duration-fast: 100ms;    /* Instant feedback: button press, hover */
  --aura-duration-normal: 200ms;  /* State changes: expand, color shift */
  --aura-duration-slow: 300ms;    /* Layout shifts: panel open, sidebar */
  --aura-ease: cubic-bezier(0.4, 0, 0.2, 1); /* Material Design ease */
}

/* Processing pulse */
@keyframes aura-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.4); }
  50%      { box-shadow: 0 0 0 8px rgba(139, 92, 246, 0); }
}
.is-processing { animation: aura-pulse 2s ease-in-out infinite; }

/* Content height expansion */
.message-content {
  transition: height var(--aura-duration-normal) var(--aura-ease);
}
```

**Specific animations:**
- Processing: pulse on avatar (2s cycle)
- Height expansion: as streaming content arrives (200ms)
- Colour transitions: when confidence scores update (200ms)
- Dismissal: gentle `opacity 1->0` fade-out (150ms)

---

### Pattern 18: Voice Input with Waveform Visualization

**Seen in:** Gemini (best), ChatGPT Advanced Voice, Claude
**Impact:** Voice usage in AI apps grew 65% year-on-year. Essential for 2026.

```tsx
function VoiceInput({ isListening, audioLevel }: Props) {
  return (
    <motion.button
      className={cn(
        "relative p-3 rounded-full transition-colors",
        isListening ? "bg-red-500/20 text-red-400" : "hover:bg-zinc-700 text-zinc-400"
      )}
      whileTap={{ scale: 0.95 }}
    >
      <Mic className="w-5 h-5 relative z-10" />
      {isListening && (
        <motion.div
          className="absolute inset-0 rounded-full bg-red-500/20"
          animate={{ scale: [1, 1 + audioLevel * 0.5, 1] }}
          transition={{ duration: 0.15, repeat: Infinity }}
        />
      )}
    </motion.button>
  );
}
```

---

### Pattern 19: Live Preview Pane (for code/generative output)

**Seen in:** v0.dev, bolt.new, Claude Artifacts, Gemini Canvas
**Impact:** Seeing output in real-time as it generates is the defining feature of code-gen UIs.

```tsx
function LivePreview({ html, isStreaming }: Props) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    if (iframeRef.current) {
      const doc = iframeRef.current.contentDocument;
      if (doc) {
        doc.open();
        doc.write(html);
        doc.close();
      }
    }
  }, [html]);

  return (
    <div className="relative h-full rounded-xl overflow-hidden border border-zinc-700/50">
      {/* Preview toolbar */}
      <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800/80 border-b border-zinc-700/50">
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500/60" />
          <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
          <div className="w-3 h-3 rounded-full bg-green-500/60" />
        </div>
        <span className="text-xs text-zinc-500 font-mono ml-2">Preview</span>
        {isStreaming && (
          <motion.div className="ml-auto flex items-center gap-1.5 text-xs text-yellow-400"
            animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 1.5, repeat: Infinity }}>
            <Loader2 className="w-3 h-3 animate-spin" /> Generating...
          </motion.div>
        )}
      </div>

      {/* Sandboxed iframe */}
      <iframe
        ref={iframeRef}
        sandbox="allow-scripts"
        className="w-full h-[calc(100%-40px)] bg-white"
        title="Preview"
      />
    </div>
  );
}
```

---

### Pattern 20: ARIA-First Accessibility for Dynamic AI Content

**Seen in:** Required by WCAG 2.2. Best execution in ChatGPT and Claude.
**Impact:** Not just compliance — screen reader users are a real audience. Dynamic AI content needs special handling.

```tsx
// Live region for streaming responses
function AccessibleStreamingResponse({ content, isStreaming }: Props) {
  return (
    <div
      role="log"
      aria-live="polite"
      aria-atomic={false}
      aria-relevant="additions"
      aria-label="AI response"
    >
      {content}
      {isStreaming && (
        <span role="status" className="sr-only">
          Aura is generating a response...
        </span>
      )}
    </div>
  );
}

// Focus management when response completes
function useResponseFocus(isStreaming: boolean) {
  const responseRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!isStreaming && responseRef.current) {
      // Move focus to the response without scrolling
      responseRef.current.focus({ preventScroll: true });
    }
  }, [isStreaming]);
  return responseRef;
}
```

**ARIA requirements:**
- `aria-live="polite"` on response containers
- `role="status"` on loading indicators
- Keyboard focus to completed responses
- Descriptive alt text on generated images
- Skip-to-content links for sidebar navigation

---

## Vercel AI Elements — Ready-Made Component Library

**Critical discovery:** Vercel released [AI Elements](https://elements.ai-sdk.dev) — a shadcn/ui-style component library specifically for AI interfaces. Install with `npx ai-elements@latest add <component>`.

Available components that map directly to the patterns above:

| Component | Maps to Pattern # |
|-----------|------------------|
| `chain-of-thought` | #3 Collapsible Thinking |
| `inline-citation` | #5 Inline Citations |
| `sources` | #5 Sources Bar |
| `shimmer` | #6 Skeleton Loading |
| `suggestion` | #10 Follow-Up Chips |
| `message` | #11 Message Transitions |
| `code-block` | #12 Code Blocks |
| `reasoning` | #3 Reasoning Display |
| `model-selector` | #8 Model Picker |
| `tool` | #9 Tool Call Visualization |
| `artifact` | #4 Artifacts Panel |
| `sandbox` | #19 Live Preview |
| `prompt-input` | #14 Composer Bar |
| `web-preview` | #19 Live Preview |
| `conversation` | #15 Conversation History |

This is the fastest path to a production-quality AI chat UI. Composable, accessible, typed, and designed for Vercel AI SDK integration.

---

## Implementation Priority for Aura

### Phase 1 — Foundation (Week 1)
1. Pattern 1: Glassmorphic dark panels (CSS foundation)
2. Pattern 14: Responsive composer bar
3. Pattern 2: Streaming with cursor
4. Pattern 6: Skeleton loading shimmer
5. Pattern 11: Message transition animations

### Phase 2 — Intelligence Display (Week 2)
6. Pattern 3: Collapsible thinking display
7. Pattern 9: Tool call visualization
8. Pattern 13: Processing state indicators
9. Pattern 8: Model picker pill
10. Pattern 15: Collapsible sidebar

### Phase 3 — Advanced Features (Week 3)
11. Pattern 4: Artifacts/canvas side panel
12. Pattern 5: Inline citation cards
13. Pattern 7: Smart empty state
14. Pattern 10: Follow-up suggestion chips
15. Pattern 12: Code blocks with actions

### Phase 4 — Polish (Week 4)
16. Pattern 16: Confidence indicators
17. Pattern 17: Micro-animation state system
18. Pattern 18: Voice input waveform
19. Pattern 19: Live preview pane
20. Pattern 20: ARIA accessibility

---

## Sources

- [UI/UX Design Trends for AI-First Apps in 2026](https://www.groovyweb.co/blog/ui-ux-design-trends-ai-apps-2026)
- [AI Chat UI Libraries Evaluation 2026](https://dev.to/alexander_lukashov/i-evaluated-every-ai-chat-ui-library-in-2026-heres-what-i-found-and-what-i-built-4p10)
- [Vercel AI Elements - Chain of Thought Component](https://ai-sdk.dev/elements/components/chain-of-thought)
- [How AI Models Show Reasoning - Digestible UX](https://www.digestibleux.com/p/how-ai-models-show-their-reasoning)
- [Claude Generative UI vs Canvas vs Artifacts](https://www.mindstudio.ai/blog/what-is-claude-generative-ui-vs-canvas-artifacts)
- [Perplexity Platform Guide - Citation-Forward Design](https://www.unusual.ai/blog/perplexity-platform-guide-design-for-citation-forward-answers)
- [Gemini Canvas Overview](https://gemini.google/overview/canvas/)
- [Top 10 Chatbot Designs 2025-26](https://www.letsgroto.com/blog/top-10-chatbot-design-examples)
- [Innovative Chat UI Design Trends 2025](https://multitaskai.com/blog/chat-ui-design/)
- [ChatGPT Canvas Introduction](https://openai.com/index/introducing-canvas/)
- [Conversational AI UI Comparison 2025](https://intuitionlabs.ai/articles/conversational-ai-ui-comparison-2025)
- [AI Citation UX Patterns](https://www.shapeof.ai/patterns/citations)
- [Perplexity UX Analysis - NN/g](https://www.nngroup.com/articles/perplexity-henry-modisett/)
- [FlowToken - Streaming LLM Animation Library](https://github.com/Ephibbs/flowtoken)
- [Motion (Framer Motion) React Animation](https://motion.dev/docs/react-animation)
