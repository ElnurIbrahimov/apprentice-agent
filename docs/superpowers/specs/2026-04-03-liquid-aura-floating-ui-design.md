# Liquid Aura — World-Class Floating UI Redesign

**Date:** 2026-04-03
**Scope:** Complete redesign of Aura browser extension floating UI surfaces
**Status:** Approved

---

## Philosophy

Aura never floats *above* content — it manifests *on* and *through* content. It flows like liquid between states. It's ambient until needed, intelligent about when to appear, and morphs seamlessly between forms.

---

## 1. Design System — Tokens & Identity

### 1.1 Context-Aware Color Palette

The entire UI shifts color based on detected page context:

| Page Context | Accent Hex | Glow RGBA | Detection |
|---|---|---|---|
| General | `#7c3aed` | `rgba(124,58,237,0.3)` | Default fallback |
| Article/docs | `#3b82f6` | `rgba(59,130,246,0.3)` | `<article>`, text density >70% |
| Video/media | `#f59e0b` | `rgba(245,158,11,0.3)` | YouTube, Netflix, `<video>` |
| Code/dev | `#10b981` | `rgba(16,185,129,0.3)` | GitHub, `<code>` blocks |
| Email/compose | `#6366f1` | `rgba(99,102,241,0.3)` | Gmail, Outlook web |
| Shopping | `#ec4899` | `rgba(236,72,153,0.3)` | Product schema, cart elements |

### 1.2 FAB Icon Morphing — Contextual Glyphs

| Context | Icon |
|---|---|
| General | Aura triangle logo |
| Article | Open book silhouette |
| Video/media | Waveform bars |
| Code/dev | Angle brackets `< >` |
| Email | Quill/pen nib |
| Shopping | Sparkle/tag |

Icons transition via cross-dissolve (old fades out, new fades in) over 400ms.

### 1.3 Animation Tokens

| Token | Value | Usage |
|---|---|---|
| `morph-duration` | 350ms | Element appear/retract |
| `morph-easing` | `cubic-bezier(0.4, 0, 0.0, 1)` | Decelerate curve |
| `flow-duration` | 500ms | Ghost bar → modal transitions |
| `glow-pulse` | 3s infinite, opacity 0.15–0.35 | FAB ambient glow |
| `sequential-stagger` | 40ms | Popout button cascade |
| `dismiss-delay` | 400ms | Hover-out grace period |

All transitions use `will-change` and `transform`/`opacity` only — zero layout thrashing.

### 1.4 Glassmorphism Tokens (shared across all surfaces)

```
background:      rgba(10, 8, 24, 0.88)
backdrop-filter:  blur(20px) saturate(1.5)
border:          1px solid rgba({context-color}, 0.25)
shadow:          0 0 1px {context-glow}, 0 8px 32px rgba(0,0,0,0.4)
```

---

## 2. Living FAB

### 2.1 Resting State

- Sits on screen edge (right or left), draggable with persisted position (same as current)
- Soft ambient glow pulses behind it in the current context color (3s cycle)
- Diffused radiance, not a ring — like the pill is warm
- Glass refraction effect via backdrop-filter — page content behind it slightly distorted
- Icon inside cross-dissolves to match page context

### 2.2 Hover → Popout

- Action buttons flow out sequentially — each oozes out 40ms after the previous (liquid beading effect)
- Each action button is icon-only; icon cross-dissolves to a text label on individual button hover
- Popout border glow inherits current context color
- Actions: Chat, Search, This Page, Translate, Save to Memory (same as current, reordered by context)

### 2.3 Drag Behavior

- On grab: pill detaches from edge, becomes full circle
- Liquid stretch effect: brief elastic tether to origin for ~100ms before snapping free
- Ghost outline shows dock target while dragging
- On release: flows to nearest edge with elastic settle (slight overshoot + single bounce-back)

### 2.4 Click

- Single click → opens sidebar panel (unchanged)
- FAB is the persistent entry point; ghost bars are contextual entry points
- No competition between them

---

## 3. Ghost Bar — Text Selection

### 3.1 Appearance Sequence

1. User selects text → browser native highlight appears
2. After 300ms stable selection, Aura enhances the highlight: a positioned overlay div is placed over the selection range rects with `rgba(context-color, 0.12)` tint and a subtle bottom border (cannot override native `::selection` reliably across sites, so this is an additive overlay, not a CSS replacement)
3. Ghost bar oozes out from the bottom edge of the last line of selection — starts as thin line at selection width, expands to 28px height over 350ms (liquid pooling)
4. Bar is visually attached to selection — same width, no gap, shared border radius at bottom

### 3.2 Contents

Compact icon-only row (15px icons) with subtle separators:

**Primary row:** Ask (Aura logo) · Copy · Explain · Summarize · Translate · Highlight · More ›

- Each icon shows 1-word tooltip on hover (no expanding labels)
- "Ask" triggers morph-to-modal transition
- "More ›" reveals second row flowing out below

**Extended row:** Rewrite · Grammar · Define · Read aloud

Action order is influenced by context engine:
- Article context: Summarize and Explain promoted first
- Code context: Explain and Rewrite promoted first
- Email context: Rewrite and Grammar promoted first

### 3.3 Position Tracking

- Bar moves with text on scroll (anchored to selection range bounding rect)
- If selection scrolls off-screen, bar dissolves
- If user deselects (click elsewhere), bar retracts upward — reverse of appear animation, liquid reabsorbed into text

### 3.4 Ghost Bar → Modal Transition

1. Ghost bar stretches — width expands smoothly toward center-screen
2. Simultaneously grows in height, glass background intensifying
3. Selected text flows into the modal's preview area (text animates from original position into modal body)
4. Input field, action buttons, model selector fade in sequentially
5. Page dims behind (`rgba(0,0,0,0.3)` overlay, 500ms fade)
6. Total transition: 500ms — the ghost bar *becomes* the modal

---

## 4. Ghost Bar — Image Interaction

### 4.1 Appearance Sequence

1. User hovers image (≥80px, same filtering as current)
2. After 800ms stable hover, image bottom edge starts glowing — thin line of context-color light
3. Ghost bar flows upward from that glowing edge into the image's bottom area
4. Bar sits *inside* image bounds, overlaying bottom ~32px with heavier glass (`rgba(10,8,24,0.75)`) for legibility
5. Bar spans image width, bottom corners match image's border-radius

### 4.2 Contents

Icon + label buttons (16px icons, slightly larger for image contrast):

**Describe** (eye) · **Edit** (pen) · **Save** (download) · **Ask** (Aura logo)

- "Ask" opens morph-to-modal with image as context (image preview in modal instead of text)

### 4.3 Behavior

- Mouse leaves image AND bar → bar retracts downward (reverse flow), glow fades. 400ms dismiss delay.
- Scroll moves bar with image. Off-screen → dissolve.
- Only one image ghost bar at a time. New hover dissolves old one first.

### 4.4 Directional Inversion

- Text ghost bar: appears *below* selection (flows down)
- Image ghost bar: appears *inside bottom edge* of image (flows up)
- Same material, same animation language, directionally inverted — each feels native to its content type

---

## 5. Focus Modal

### 5.1 Appearance

- Centered glassmorphism modal
- Morphs FROM the ghost bar (see section 3.4 for transition)
- Page dims with `rgba(0,0,0,0.3)` overlay

### 5.2 Layout & Contents

- **Max dimensions:** 520px wide, 480px max height. Vertically centered in viewport.
- **Preview area:** Selected text (max 6 lines visible, scrollable overflow) or image thumbnail (max 200px height, aspect-ratio preserved). Selections >2000 chars are truncated with "... (N more chars)" indicator.
- **Input field:** Freeform prompt, placeholder adapts to context ("Ask about this article...", "Ask about this code...")
- **Action buttons:** Explain, Summarize, Chat with AURA, Save to Memory, Translate (same as current Quick Launch)
- **Model selector:** Auto / Fast / Balanced / Powerful

### 5.3 Dismiss

Reverse flow animation:
1. Modal shrinks toward ghost bar origin position
2. Text/image flows back to original page position
3. Page un-dims
4. Ghost bar reappears (or dissolves if selection is gone)

Triggers: Esc key, clicking overlay, close button.

---

## 6. Context Engine — 5-Layer Intelligence

### 6.1 Layer 1 — Page Type Detection

**Runs:** On page load, URL change (SPA `popstate`/`pushstate`), debounced on major DOM mutations. Max once per 2 seconds.

**Cascading priority:**

1. URL pattern match (fastest):
   - `github.com`, `gitlab.com` → `code`
   - `youtube.com`, `netflix.com` → `media`
   - `mail.google.com`, `outlook.live.com` → `email`
   - Known shopping domains, `/product/` or `/cart/` → `shopping`

2. DOM signal analysis (fallback):
   - `<article>`, `role="article"`, text density >70% → `article`
   - `<pre>`, `<code>` covering >20% of content → `code`
   - `<video>`, `<audio>` elements → `media`
   - Product `ld+json` schema → `shopping`

3. Default: `general`

### 6.2 Layer 2 — Content Topology

- **Reading position:** `IntersectionObserver` on major content blocks. Knows which section you're reading. FAB glow intensifies as you reach article end (signaling "I can summarize what you just read").
- **Image density map:** Image-heavy zones → image ghost bars appear faster (500ms). Text-heavy zones → image ghost bars slower (1200ms). Reduces noise.
- **Interactive element awareness:** Detects forms, inputs, editors. During active typing in `contenteditable` or `textarea`, all ghost bars and FAB popout suppressed. FAB glow dims to near-invisible. Aura goes quiet.

### 6.3 Layer 3 — Session Memory

- If you already summarized a selection and select nearby text, ghost bar promotes **"Continue"** instead of Summarize — feeds new selection as follow-up to same conversation.
- After 2+ saved highlights on a page, FAB popout gains temporary **"Review highlights"** action.
- Dismissals increase ghost bar appear delay by 200ms each (up to 2s max). Resets on navigation. Aura learns you're not in the mood.

### 6.4 Layer 4 — Behavioral Cadence

Rolling window analysis: scroll velocity (last 10s), selection frequency (last 30s), input activity.

| State | Detection | Aura Behavior |
|---|---|---|
| `passive` | Fast scrolling, no selections | FAB dims, ghost bars won't trigger |
| `engaged` | Slow scroll, pausing on paragraphs | FAB glow warms, ghost bars at normal speed |
| `active` | Frequent selections, typing, tab switching | Ghost bars appear faster (200ms), modal remembers last action |

State transitions require 3+ seconds of consistent signal to avoid flicker.

### 6.5 Layer 5 — Cross-Tab Awareness

Via `chrome.storage.session` (ephemeral, not persisted to disk):

- If a modal conversation is open in another tab and you select text on this tab, ghost bar offers **"Send to conversation"** — adds selection as context without opening new modal.
- Tab-level research theme is shared: if you're researching a topic across tabs, context engine pre-populates modal input placeholder with relevant prompt suggestions.

### 6.6 Context Signal Shape

```typescript
interface ContextSignal {
  type: 'article' | 'code' | 'media' | 'email' | 'shopping' | 'general';
  accent: string;        // hex color
  glow: string;          // rgba color
  icon: string;          // SVG markup for FAB glyph
  actions: string[];     // ghost bar actions in relevance order
  cadence: 'passive' | 'engaged' | 'active';
  suppressGhostBars: boolean;   // true when user is typing
  sessionActions: string[];     // extra actions from session memory
  readingProgress: number;      // 0-1, how far through main content
}
```

---

## 7. Module Architecture

### 7.1 File Structure

```
extension-src/src/content/
├── index.ts              # Coordinator (~150 lines) — mounts shared Shadow DOM,
│                         #   initializes modules, wires message passing
├── context-engine.ts     # 5-layer intelligence. Emits ContextSignal via pub/sub.
├── fab.ts                # Living Pill: render, drag, popout, glow, icon morphing.
│                         #   Subscribes to ContextSignal.
├── ghost-bar.ts          # Text + image ghost bars. Appear/retract animations,
│                         #   action dispatch, position tracking. Subscribes to
│                         #   ContextSignal for action ordering and timing.
├── modal.ts              # Glassmorphism focus modal. Morph-from-ghost-bar
│                         #   transition, input, actions, model selector, dismiss.
├── highlights.ts         # Highlight save/restore/tooltip. Behavior unchanged,
│                         #   adopts shared tokens and animation language.
├── capture.ts            # Element capture overlay mode.
├── gmail.ts              # Gmail compose AI button injection.
├── link-preview.ts       # Link hover preview popup.
├── animator.ts           # Shared liquid-morph animation primitives:
│                         #   flow(), morph(), dissolve(), crossFade().
│                         #   All modules import from here. Returns Promises
│                         #   for chaining.
├── tokens.ts             # Design tokens: colors, timing, easing, glassmorphism.
│                         #   Single source of truth. Context engine updates
│                         #   CSS custom properties on context change.
├── styles.ts             # Generates all CSS from tokens. One stylesheet
│                         #   injected into the shared Shadow DOM.
└── types.ts              # Shared interfaces: ContextSignal, GhostBarState,
                          #   ModalState, FabState, AnimationConfig.
```

### 7.2 Module Communication

```
context-engine ──emits──▸ ContextSignal (reactive pub/sub, ~30 lines)
     │
     ├──▸ fab.ts         reads → color, icon, glow, popout actions
     ├──▸ ghost-bar.ts   reads → action order, appear timing, suppression
     ├──▸ modal.ts       reads → placeholder text, default action
     └──▸ styles.ts      reads → CSS custom properties updated on root
```

- Ghost bar emits events (`ghost-bar:ask-clicked`, `ghost-bar:action`) routed by coordinator to modal or background script
- All DOM rendering into single shared Shadow DOM created by `index.ts`
- Modules receive a container element, not the shadow root directly
- `animator.ts` exports: `flow(el, from, to, opts)`, `morph(elA, elB, opts)`, `dissolve(el, opts)` — all return Promises for chaining

### 7.3 Migration from content.ts

- Current `content.ts` (4400 lines) becomes thin shim: imports `content/index.ts`, calls `init()`
- YouTube inject, Netflix inject, background message handling stay in existing files untouched
- All current functionality preserved — this is a restructure + upgrade, not a rewrite of behavior

---

## 8. Unchanged Systems

These retain current behavior, adopting only the shared design tokens and animation language:

- **Highlights** — mark styling, tooltip, save/delete. Colors shift to context palette.
- **Capture overlay** — element selection mode. Border color matches context.
- **Gmail integration** — AI compose button injection. Unchanged in behavior.
- **Link preview** — Hover popup on links. Glass material updated to match system.
- **Toast notifications** — Adopt glass material and context color.

---

## 9. Key Constraints

- All floating UI runs inside Shadow DOM — zero CSS leakage to/from host page
- No external font loading — system font stack only
- No runtime dependencies beyond Chrome Extension APIs
- Must work on Chrome and Firefox (via existing `browser`/`chrome` shim)
- Performance budget: context engine must not block main thread (use `requestIdleCallback` for DOM analysis)
- Image ghost bar legibility: heavier glass overlay (`0.75` opacity vs `0.88`) to ensure readability against any image content
