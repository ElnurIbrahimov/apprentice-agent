# Aura Android v0.15.0 — UI/UX polish + design system honest

Release date: 2026-07-15
Branch: feat/tier-1-friction
Head: c8ef82c (and 7 prior commits in this slice)
APK: 35 MB (debug)
Test count: 593 aura-core + 236 app = 829 total (0 failures)

## What this release is

This is the first release of Aura where every Compose surface
reads its colors from the design system — not from Material 3's
default color scheme. Combined with a new adaptive icon, a
splash drawable that matches the launcher tile, and free-text
search on the two list screens that didn't have it.

It is the result of the 2026-07-15 UI/UX audit
(`.hermes/plans/2026-07-15-task6-ui-ux-audit.md`) — 25 gaps
identified, 17 shipped in this release, 5 audited as
"already correct" (the audit grep was wrong), 3 deferred
(needing their own session each).

## Shipped

### Wired-but-unusable (Task 5 cleanup)

- **Custom provider has a UI now.** Settings shows a
  `CustomEndpointCard` with separate fields for base URL and
  API key, a "Test connection" button, and persists both to
  `CustomEndpointState` + DataStore. The custom provider
  previously registered an API key prefix but no place to
  set the endpoint.
- **Capability providers are honest.** Exa / Jina / Brave /
  Tavily / Firecrawl are surfaced as enabled (real, used by
  tools). ElevenLabs / Stability / Kling / WorldLabs render
  disabled with a "Coming soon — this key isn't consumed by
  any tool yet" hint. No more fake-promises in Settings.
- **Skills reach the chat composer.** The chat composer
  attachment sheet now lists installed skills; tapping one
  inserts a `/use_skill <name>` directive. The user no longer
  has to remember skill names from memory.

### Brand polish

- **Adaptive launcher icon.** Background was the stock system
  `@android:color/holo_purple` (#AA00FF) — completely off
  brand. New gradient drawable matches the brand
  (`#0E1A2B → #1E2440 → #3F2A77`) and proper adaptive-icon
  definitions in `mipmap-anydpi-v26/`. `<monochrome>` layer
  added for Android 13+ themed icons.
- **Splash drawable.** `aura_splash.xml` is a layer-list with
  the brand gradient behind the centered Aura mark. The launch
  theme (light + night) now points at this drawable. The
  previous 200-400ms light-flash on cold start is gone.
- **Provider label hardcoding collapsed to a single source
  of truth.** `providerLabel(prefix)` lives in
  `aura-core/providers/ProviderLabels.kt` and is used by
  ModelPickerSheet, ModelLabels, and the onboarding
  ModelSelectionStep. New providers (mistral, xai, together,
  cerebras, nvidia, llama, agnes, chatgpt, exa, jina,
  elevenlabs, stability, kling, worldlabs) now display with
  their friendly names.

### Daily-use friction

- **Tasks search + filter.** Free-text search across title,
  description, and tags. Search-aware empty state. Trailing
  clear button.
- **Reminders search.** Free-text search across message and
  recurrence. Search-aware empty state.
- **Hands search.** Free-text search across hand name and
  trigger phrase.
- **Profile "Add" affordances** changed from the Edit
  (pencil) icon to the Add (+) icon. The buttons now look
  like they add a trait / fact, not edit one.

### Token-migration ratchet (G12, G16, G17)

The design system was defined in `AuraSemanticColors.kt` and
exposed via `AuraThemeTokens.colors`. Production code was
supposed to consume it. Production code was consuming
`MaterialTheme.colorScheme.primary` 415 times instead.

The fix is mechanical but it changes everything: every Compose
surface now reads its colors from the Aura token system, which
flips on system dark mode and respects the brand. The M3
default `colorScheme` is still wired (the `AuraTheme`
composable sets it for the M3 components that don't take
explicit colors), but no app screen reads from it directly.

`MaterialTheme.colorScheme.*` uses in `:app`: 415 → 0.
`AuraPaletteBoundaryTest` still passes.

## Audited as "already shipped" (no code change)

- **G6 (Calendar monitor Settings toggle):** the toggle flips
  the flag, `ProactiveBootstrap.reconcile()` reads the flag
  and calls `CalendarMonitorService.start()` / `stopService()`.
  The "stop the persistent foreground service" label is honest.
- **G4 (Light theme):** `AuraTheme` correctly switches
  `LocalAuraSemanticColors` between Dark and Light variants
  based on system mode. `AuraPaletteBoundaryTest` enforces
  that no production code reads the raw palettes.
- **G14 (Onboarding error visibility):** `ProviderSetupStep`
  passes `state.providerMessages[prefix]` as `verifyResult` to
  `ProviderKeyField`, which renders it under the field. Failed
  test, 401, network timeout, rate limit, malformed catalog
  — all surface inline.
- **G20 (KnowledgeGraph search):** the screen already had both
  a free-text search and the `TypeChip` filter row.
- **G21 (Hands step builder UI):** `HandEditorDialog` already
  has a `StepEditor` composable with an `ExposedDropdownMenu`
  of available tools and per-tool argument fields. Not a raw
  JSON field.

## Deferred to its own session

- **G15 (ElevenLabs TTS chat hookup).** Requires extending
  the `TextToSpeech` wrapper to accept a `TextToSpeechProvider`,
  adding MediaPlayer streaming, and wiring a UI toggle. The
  capability provider (`ElevenLabsTtsProvider`) is
  implemented and tested; the chat-side integration is the
  missing piece.
- **Mood theming** (initial commit in the prior session).
  Needs a Python backend API integration + state plumbing.
- **Graph visualization** (initial commit in the prior
  session). No existing graph renderer.

## Commits in this slice

```
7ee07de feat(brand): dark splash drawable, kill the cold-start flash
f20e732 feat(settings): add Custom Endpoint URL field
4323b6c feat(providers): single source of truth for provider labels
e56b550 feat(chat): add Skills to composer attachment sheet
23cee40 feat(settings): mark capability providers as Coming soon
73340c2 feat(tasks): search + filter, fix ProfileScreen Add icons
374f891 feat(screens): search bars for Reminders and Hands
925b8dc feat(theme): ratchet top 4 MaterialTheme offenders
111f3ba feat(theme): ratchet 2nd batch — 10 more files
c8ef82c feat(theme): ratchet 3rd batch — zero MaterialTheme uses
```

## Download

`releases/aura-debug-v0.15.0-20260715-180000.apk` (35 MB)
