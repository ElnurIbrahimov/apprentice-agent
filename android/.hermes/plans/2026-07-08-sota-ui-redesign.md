# Aura Android SOTA UI — Design System Overhaul

**Branch:** `feat/tier-1-friction`
**Source of truth:** `D:\Aura\web\src\{components/MessageBubble.tsx, components/MessageInput.tsx, index.css, tailwind.config.js}`
**Verdict (from user):** "ui/ux is so bad that i wanted to vomit" — must end at Kira-tier.

The current Android theme is teal/charcoal (color is OK) but the **shapes, motion, density, and tone** are generic Material 3. The 8 SOTA feature commits shipped on top of that generic skin. This plan replaces the skin with Aura Web's design language so the features render in a system that matches their ambition.

## What I'm taking from Aura Web

### Tokens (1:1 from web `--vars` in `index.css`)
- Surfaces 0–4 (5 elevation steps, not 3)
- Border subtle 4% / default 8% / strong 16%
- Text primary `ededed` / secondary `a1a1aa` / tertiary `6b6b6b`
- Accent `8b5cf6` (vivid violet)
- Glows: purple/blue/green/red/orange
- Mood accent variable for proactive context
- Radii: sm 8, md 12, lg 24, full

### Typography
- Inter Tight body (already in Type.kt? need to add)
- Fraunces for the home-screen hero ("What should we explore?")
- JetBrains Mono for code
- Tabular-nums for tok/s speed badge

### Shapes (where the 1990s feel lives)
- **User bubble**: asymmetric radius `24, 24, 4, 24` — "pointed" at the sender. Currently Android uses uniform 16dp.
- **AI bubble**: NO bubble. Just avatar + content. Currently Android uses a Surface.
- **Send button**: morphs — pill (20dp radius) when ready, square (12dp radius) when empty. Spring-eased.
- **Glass input**: backdrop-blur 24dp, border-subtle (4% white), 24dp radius.

### Motion
- Spring easing `cubic-bezier(0.34, 1.56, 0.64, 1)` for spring-up / spring-scale
- `breathe` keyframe for the AI avatar
- `pulse-glow` for thinking states
- `slide-up-fade` for empty-state cards
- Streaming cursor: 2px wide, blinking, fades out 200ms after stream ends
- Live `tok/s` badge during streaming
- `msg-animate-in` stagger (0–4 indices)

### Home / empty state
- Centered hero with breathing glow backdrop (radial gradient blur)
- H1: "What should we explore?" (Fraunces, large)
- Subtitle: "Research, create, code, and compare"
- 4–6 quick-action cards with icons, staggered spring-up
- No list view, no empty `Column` of "Ask me anything…"

### Citation chip
- 18px circle, purple-tinted background, scales 1.15× on hover/highlight
- Connected hover: when you hover a `[1]` in text AND a `[1]` in the source list, both highlight

### Memory recall
- Pill: "Used 5 memories · 2 hands" with a brain icon
- Tap → bottom sheet with the actual items

## Anti-features (deliberate non-goals)
- No bottom nav drawer (web is single-canvas, not bottom tabs)
- No new icons (use the same Heroicons semantic set as web)
- No light theme port (user said dark first; web is dark-first)
- No haptic/motion preference port (Android handles that in system settings)
- No modelCompare / ArtifactsPanel port (web-only features)
- No split-pane Artifacts side-panel (Android is single-pane)
- No citations right-side panel (use a bottom sheet on Android)

## Commits (atomic, each green per existing gate)

1. **`feat(ui): web-grade design tokens`** — write `AuraDesignTokens.kt` matching web CSS vars (surfaces 0-4, borders, text, glows, mood, radii). Switch `Theme.kt` to read from tokens. Drop the dynamic-color default-off behavior (we want the brand, not the wallpaper).
2. **`feat(ui): Inter Tight + Fraunces typography`** — `Type.kt` adopts Inter Tight body, Fraunces display, JetBrains Mono code. Use the weights web uses (300–700).
3. **`feat(ui): message bubble redesign`** — `MessageBubble.kt`. User bubble asymmetric radius 24/24/4/24 with shadow. AI bubble: NO bubble, just avatar + content. Avatar is 9×9 with gradient (white/12 → white/2), 1px white/07 border. Spring-up entry animation per index.
4. **`feat(ui): streaming cursor + tok/s badge`** — `StreamingText.kt` gains a blinking 2dp cursor at end of text, fades 200ms after stream ends. During stream, shows a `42 tok/s` badge in JetBrains Mono tabular-nums. Reuse the existing `StreamingMarkdownState` (don't break it).
5. **`feat(ui): glass input with morphing send button`** — `ChatInputBar.kt`. Input wrapper becomes glass: surface-1, border-subtle, blur 24dp, 24dp radius, 16dp text, 44dp min-height, 140dp max. Send button morphs: pill 20dp radius when ready (with purple shadow), square 12dp when empty. Spring-eased transition. Plus button (40dp, surface-2) keeps current shape.
6. **`feat(ui): empty-state home with breathing glow + quick actions`** — `HomeScreen.kt` (or new `EmptyChatState.kt`). Centered: breathing purple radial glow behind, "What should we explore?" in Fraunces 36–60sp, "Research, create, code, and compare" subtitle, 4–6 quick-action cards (icons from web's `QUICK_ACTIONS` set: research / create / code / compare). Staggered spring-up entry. Tap → fills draft + focus input.
7. **`feat(ui): inline citation chip with connected hover + memory recall polish`** — Refine what I just shipped. Citation chip: 18dp circle, purple tint, scale 1.15× on highlight. Memory recall pill: brain icon + "Used 5 memories · 2 hands" + tap → bottom sheet listing the actual items.
8. **`feat(ui): chat header glass + breathing AI avatar`** — `ChatHeader.kt` (or wherever). Glass surface (blur 16dp, border-subtle, surface-1). AI avatar: 36–44dp breathing radial glow ring (matching `AuraBreathingAvatar.tsx` exactly), connected status dot, model pill.

## Verification
- `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug :app:lintDebug` green after each commit
- Visual: take a screenshot of Home + Chat (empty + with messages + streaming) and verify against web's design language
- Lint: no new deprecations

## Out of scope (deliberate, will not be done)
- Light theme (user runs dark)
- Bottom tab nav
- Artifacts / Citations side panel
- Light/dark switcher (user prefers dark)
