> SUPERSEDED 2026-04-13. Current source of truth: D:/Aura/CURRENT_STATE.md

# Aura UI Polish Roadmap — State of the Art

**Created:** 2026-03-20
**Research basis:** 5 parallel research agents analyzed ChatGPT, Claude, Gemini, Perplexity, v0/bolt.new, Sider, Monica, Merlin, MaxAI, Arc, Claude Code, Cursor, Aider, Warp, Linear, Raycast, and 30+ design system references.

---

## Current State

| Surface | Status | Main Issues |
|---------|--------|-------------|
| **Web UI** | Functional, ChatGPT-clone feel | No message animations, raw streaming, no suggestion chips, no thinking display |
| **Extension** | 24 panels, feature-rich | 26-icon rail is overwhelming, no command palette, no inline responses, flat transitions |
| **CLI** | Strongest surface (Rich library) | No block-level streaming, no context summary line, status bar could be richer |
| **Mobile Web** | Basic fixes applied (dvh, touch targets) | No gestures, no bottom sheets, no PWA, no haptics, no suggestion chips |

---

## Design Tokens (Apply Across All Surfaces)

### Surface Elevation (5 levels, not shadows)
```css
--surface-0: #09090b;  /* deepest — page background */
--surface-1: #0a0a0a;  /* main content area */
--surface-2: #111113;  /* cards, panels */
--surface-3: #171719;  /* elevated cards, popovers */
--surface-4: #1c1c1f;  /* tooltips, command palette */
--surface-5: #232326;  /* highest elevation — modals */
```

### Borders (opacity-based, not solid colors)
```css
--border-subtle: rgba(255, 255, 255, 0.06);
--border-default: rgba(255, 255, 255, 0.08);
--border-hover: rgba(255, 255, 255, 0.12);
--border-focus: rgba(255, 255, 255, 0.20);
```

### Text Hierarchy
```css
--text-primary: #ededed;    /* 93% brightness, not pure white */
--text-secondary: #a1a1aa;  /* zinc-400 */
--text-tertiary: #71717a;   /* zinc-500 */
```

### AI Semantic Colors
```css
--color-thinking: #a78bfa;     /* violet — AI processing */
--color-tool-call: #f59e0b;    /* amber — tool execution */
--color-tool-result: #10b981;  /* emerald — tool success */
--color-error: #ef4444;        /* red — errors */
--color-user: #e2e8f0;         /* slate-200 — user messages */
--color-assistant: transparent; /* no bg — assistant is the default voice */
```

### Animation Durations
```css
--duration-instant: 100ms;   /* hover color change */
--duration-fast: 150ms;      /* button press, toggle */
--duration-normal: 200ms;    /* popover, dropdown */
--duration-moderate: 300ms;  /* sidebar, panel slide */
--duration-slow: 500ms;      /* page transition, modal */
--ease-out: cubic-bezier(0.16, 1, 0.3, 1);
--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
```

### Spring Physics (for Motion/Framer Motion)
```
Snappy (popovers, tooltips):  stiffness: 400, damping: 30
Smooth (page transitions):    stiffness: 200, damping: 25
Gentle (content reveal):      stiffness: 120, damping: 20
Bouncy (success states):      stiffness: 300, damping: 15
```

---

## Phase 1: Foundation

*Highest impact, needed for everything else. Mostly CSS/design token changes.*

| # | Change | Surface | Impact | Effort | Details |
|---|--------|---------|--------|--------|---------|
| 1.1 | **Design tokens overhaul** | All | Critical | Medium | Apply surface elevation, opacity borders, text hierarchy, duration scale across web + extension CSS |
| 1.2 | **Message enter animations** | Web + Mobile | High | Low | Spring physics: `opacity:0, y:8, scale:0.98` → `1, 0, 1` (stiffness:300, damping:30). Messages slide up subtly, not just appear |
| 1.3 | **Token streaming polish** | Web + Mobile | High | Low | Per-token fade+blur animation (150ms). Blinking 2px cursor at stream end (500ms cycle). Reduces perceived wait 55-70% |
| 1.4 | **Skeleton shimmer loading** | Web + Mobile | High | Low | Replace blank "thinking" state with 3-5 shimmer bars of decreasing width. `animation: shimmer 1.5s ease-in-out infinite` |
| 1.5 | **Input bar overhaul** | Mobile | Critical | Low | 16px font (prevents iOS zoom), safe-area padding, auto-grow textarea max 120px, gradient focus ring |

### Shimmer CSS
```css
.shimmer {
  background: linear-gradient(90deg, var(--surface-2) 0%, var(--surface-3) 40%, var(--surface-2) 60%);
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
}
@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### Message Enter Animation
```css
@keyframes messageEnter {
  from { opacity: 0; transform: translateY(8px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
.message-enter { animation: messageEnter 250ms cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }
```

### Token Streaming
```css
.token-reveal {
  animation: tokenFade 150ms ease-out forwards;
}
@keyframes tokenFade {
  from { opacity: 0; filter: blur(2px); }
  to { opacity: 1; filter: blur(0); }
}
```

---

## Phase 2: Navigation & Structure

| # | Change | Surface | Impact | Effort | Details |
|---|--------|---------|--------|--------|---------|
| 2.1 | **Extension: Command palette** | Extension | Massive | Medium | Ctrl+K opens fuzzy search across all 26 panels. Glass overlay, auto-focus, arrow key nav. Replaces hunting through icon rail |
| 2.2 | **Extension: Category tabs** | Extension | Massive | Medium | Replace 26-icon rail with 5 tabs: Chat, Create, Research, Tools, Settings. Sub-panels as horizontal pills within each tab |
| 2.3 | **Web: Suggestion chips** | Web + Mobile | High | Low | Empty state: 4-6 contextual quick-start cards. After responses: 2-3 follow-up suggestion chips (horizontal scroll) |
| 2.4 | **Web: Collapsible thinking display** | Web | High | Medium | During reasoning: spinner + "Thinking..." + elapsed timer. On complete: collapse to "Thought for Xs", click to expand bullet points |
| 2.5 | **CLI: Status line** | CLI | High | Low | Token gauge (colored bar), cost ($X.XX), git branch, model name in prompt_toolkit bottom_toolbar |
| 2.6 | **CLI: Contextual spinner** | CLI | Medium | Low | Phase-aware verbs: "Thinking...", "Searching...", "Writing...", "Analyzing..." instead of generic spinner |
| 2.7 | **CLI: Context summary line** | CLI | High (unique!) | Low | Before each response: `context: 3 memories | KG: "FastAPI" | mood: curious | screen: VS Code` — no other CLI does this |

### Extension Category Structure
```
Chat:     Chat, Ask, Voice Notes, Record
Create:   Write, Translate, Grammar, Artifacts, Web Creator, Image, Slides, Code
Research: Search, Research, YouTube, PDF Chat, Summary, Math
Tools:    Tools, Compare, Capture, Browser Agent, OCR, Wisebase
Settings: Settings, Models
```

---

## Phase 3: Interaction Polish

| # | Change | Surface | Impact | Effort | Details |
|---|--------|---------|--------|--------|---------|
| 3.1 | **Mobile: Bottom sheet** | Mobile | High | Medium | Swipe-to-dismiss for tools/settings. Snap to half/full height. Drag handle at top. Replace modal dialogs on mobile |
| 3.2 | **Mobile: Swipe drawer** | Mobile | High | Medium | Swipe right from left edge → conversation history drawer. Spring easing, overscroll-behavior:contain |
| 3.3 | **Extension: Inline responses** | Extension | High | Medium | Highlight text → floating toolbar → answer appears ON the page in a glass card. No sidebar needed for simple actions |
| 3.4 | **Extension: Spring transitions** | Extension | High | Low | Panel switches slide directionally (deeper=right, up=left). 250ms spring easing instead of flat fade |
| 3.5 | **Web: Tool call cards** | Web | High | Medium | Styled cards: icon + tool name + duration + collapsible result. Amber border for running, green for success, red for error |
| 3.6 | **Web: Code blocks polish** | Web | Medium | Low | Language badge top-left, copy button top-right (always visible on mobile), "Run" button for executable code |
| 3.7 | **CLI: Progressive tool call disclosure** | CLI | High | Medium | One-line summary by default (`> edit_file main.py (+12/-3) 0.4s`), Ctrl+O to expand full detail |
| 3.8 | **CLI: Inline syntax-highlighted diffs** | CLI | Critical | Medium | Green/red + language-aware syntax highlighting within diff hunks. Rich Syntax widget with diff lexer |

### Tool Call Card Design
```
+-------------------------------------------+
| [amber dot] web_search "FastAPI docs"  2.1s |
| [collapsed: click to expand]              |
+-------------------------------------------+
     ↓ (expanded)
+-------------------------------------------+
| [green dot] web_search "FastAPI docs"  2.1s |
| Found 5 results:                          |
| 1. FastAPI - Official Docs (fastapi..)    |
| 2. FastAPI Tutorial (realpython.com)      |
+-------------------------------------------+
```

---

## Phase 4: Premium Details

| # | Change | Surface | Impact | Effort | Details |
|---|--------|---------|--------|--------|---------|
| 4.1 | **PWA manifest** | Mobile | High | Low | `manifest.json` with standalone display, proper icons (192+512), theme_color #030303. Service worker for shell caching. Install banner after 2-3 visits |
| 4.2 | **CLI: Block-level streaming markdown** | CLI | Critical | Medium | Parse streaming into top-level blocks. Only re-render active block. Freeze finalized blocks. Eliminates flicker |
| 4.3 | **Extension: Breathing logo states** | Extension | Medium | Low | Idle=slow breathe, thinking=fast purple pulse, error=red, agent mode=orbiting ring. Living brand element |
| 4.4 | **Haptic feedback** | Mobile | Medium | Low | `navigator.vibrate()` on message send (50ms), long-press (30ms), errors (100ms). Android only — iOS uses audio fallback |
| 4.5 | **Mobile: Long-press context menus** | Mobile | Medium | Medium | Long-press message → Copy, Regenerate, Edit, Share. Scale-up animation, backdrop blur |
| 4.6 | **Extension: Page-aware context bar** | Extension | High | Low | Auto-detect page type (article/YouTube/code/email). Show contextual actions: "Summarize this article", "Get transcript" |
| 4.7 | **Web: View Transitions API** | Web | Medium | Medium | Shared-element transitions between conversation list and chat. Chrome/Safari/Firefox supported |
| 4.8 | **CLI: Checkpoint & rewind** | CLI | Critical | Medium | Snapshot files before each edit. Esc+Esc to rewind. Show picker of recent checkpoints |
| 4.9 | **Extension: Glassmorphic panels** | Extension | Medium | Low | Apply glass treatment to all cards/inputs inside panels. Let mesh gradient background show through |
| 4.10 | **Sound design** | All | Medium | Low | Optional audio cues: message sent (80ms whoosh), response received (150ms chime), error (200ms tone). Behind toggle |

---

## Phase 5: Differentiators (Aura-Unique)

| # | Change | Surface | Impact | Effort | Details |
|---|--------|---------|--------|--------|---------|
| 5.1 | **Emotion-aware UI** | Web + Extension | High | Medium | UI subtly shifts based on ALMA emotional state — warm tones when happy, cool when analytical, muted when calm. Aura's logo glow color reflects mood |
| 5.2 | **Memory transparency** | Web + CLI | High | Low | Show what memories/KG/emotion state influenced each response. "Aura remembered: you prefer Python, you're working on BroadMind" |
| 5.3 | **Dream journal UI** | Web | Medium | Medium | Visual display of dream consolidation results — clusters, insights, contradictions found. Morning briefing card |
| 5.4 | **Multi-agent live dashboard** | Web + CLI | High | High | When /fleet runs, show live dashboard of all sub-agents with status, current action, elapsed time. Rich Table + Live in CLI |
| 5.5 | **Proactive notification cards** | Web + Extension | High | Medium | When daemon detects something (meeting soon, stale knowledge, deep work state), show as dismissible glass cards with actions |

---

## Implementation Priority (Quick Wins First)

### Weekend Sprint (Low effort, high impact)
1. Design tokens in CSS (1.1)
2. Message enter animations (1.2)
3. Shimmer loading states (1.4)
4. Suggestion chips empty state (2.3)
5. CLI status line (2.5)
6. CLI contextual spinner (2.6)
7. CLI context summary line (2.7)
8. PWA manifest (4.1)
9. Extension breathing logo states (4.3)

### Week 1
10. Token streaming polish (1.3)
11. Collapsible thinking display (2.4)
12. Extension spring transitions (3.4)
13. Tool call cards (3.5)
14. Code blocks polish (3.6)
15. Extension page-aware context bar (4.6)

### Week 2
16. Extension command palette (2.1)
17. Extension category tabs (2.2)
18. Mobile bottom sheet (3.1)
19. Mobile swipe drawer (3.2)
20. Haptic feedback (4.4)

### Week 3
21. Extension inline responses (3.3)
22. CLI block-level streaming (4.2)
23. CLI progressive tool disclosure (3.7)
24. CLI inline diffs (3.8)
25. CLI checkpoint & rewind (4.8)

### Week 4
26. Emotion-aware UI (5.1)
27. Memory transparency (5.2)
28. Dream journal UI (5.3)
29. Multi-agent dashboard (5.4)
30. Proactive notification cards (5.5)

---

## Research Sources

Full research saved to: `AI_CHAT_UI_PATTERNS_RESEARCH.md`

**Products analyzed:** ChatGPT, Claude.ai, Gemini, Perplexity, v0.dev, bolt.new, Sider AI, Monica AI, Merlin AI, MaxAI, Arc Browser, Claude Code CLI, Cursor, Aider, Warp, Goose, OpenCode

**Design systems referenced:** Vercel/Geist, Linear, Raycast, shadcn/ui, Material Design Dark Theme, Apple Liquid Glass

**Key insight:** Aura's "context summary line" (showing what memories/KG/emotion influenced each response) and "emotion-aware UI" are genuine differentiators that no competitor has. These should be prioritized as brand-defining features.