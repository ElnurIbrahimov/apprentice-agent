# Aura Android UX Contract

This document is the implementation contract for Aura Android’s model reliability and mobile UI recovery. If a screen-level preference conflicts with this file, this file wins until deliberately revised.

## 1. Product constraints

- Aura is a personal-use, sideloaded Android assistant.
- Preserve every existing capability: chat, memory, graph, tasks, reminders, hands, tools, proactive events, voice, widgets, diagnostics, profile, and identity.
- Do not add Play Store, analytics, marketing, team, or distribution work.
- Use Aura Web’s visual language, not its three-pane desktop layout.
- Support system, light, and dark themes. Forcing dark mode is not an acceptable fix.

## 2. Model reliability contract

### Credentials

- Editing an API key changes local draft state only.
- Persistence occurs only through **Save & test** or **Clear**.
- A provider has at most one active save/test operation; a new operation supersedes the old one.
- Keys are trimmed once, encrypted once, and never logged, placed in semantics, or rendered in screenshots.
- Every load/save reaches a terminal state: not configured, saved, valid, invalid, or storage error.

### Catalog discovery

- One shared repository owns model discovery for Onboarding, Settings, Chat, widgets, background roles, and MoA.
- Configured providers are queried concurrently under supervisor isolation.
- Each provider has a 10-second discovery timeout.
- One provider failure must not hide successful provider results.
- Provider adapters return live models or typed errors. They never fabricate fallback model catalogs.
- The last successful catalog may be cached for display continuity, but stale cache never validates credentials.
- Chat sending never performs model discovery and never resolves a `default` sentinel over the network.

### Model roles

The following choices are nullable and user-selected from the catalog:

- default chat model;
- embedding model;
- vision model;
- background/proactive model;
- MoA role/preset mapping.

No concrete provider model ID may be embedded in production Kotlin or JSON behavior. An unavailable persisted choice remains visible as unavailable until the user replaces it; Aura never silently switches models.

### Honest no-model state

When no usable model exists:

- Chat shows **Connect provider** or **Choose model**;
- Send is disabled;
- model picker exposes direct recovery;
- local-only features remain usable;
- onboarding skip is allowed but cannot create a fake active model.

## 3. Semantic visual system

Screen code consumes semantic theme values, never raw `AuraTokens.Dark` or `AuraTokens.Light` palettes.

Required semantics:

- base/background and elevated surfaces 0–3;
- primary, secondary, tertiary text;
- subtle/strong borders;
- primary action and disabled action;
- success, warning, error, information;
- tool risk states;
- active mode accents;
- scrim, focus, selection, and streaming states.

Use one spacing/dimension system and one family of primitives for cards, buttons, chips, icon buttons, editors, loading, empty, error, and inline status.

## 4. Mobile proportion contract

Reference compact viewport: **393 × 851dp**.

| Element | Contract |
|---|---|
| Horizontal gutter | 16dp compact; 20–24dp medium |
| Top app bar | 56dp |
| Bottom navigation | 60–64dp visual height, excluding system inset |
| Bottom-nav icon / label | 20–22dp / 11–12sp |
| Primary title | 26–30sp |
| Section title | 18–20sp |
| Body | 15–16sp |
| Metadata | 12–13sp with readable contrast |
| Standard card padding | 14–16dp |
| Dense list row | 64–80dp |
| Tabs | 48dp |
| Visual icon button | 36–40dp inside a 48dp touch target |
| Model pill | 36–40dp high, at most 55% of compact width |
| Composer | 52dp minimum, 144dp maximum before internal scroll |
| Modal sheet | Content-adaptive, maximum 90% height |
| Empty-state content | Maximum 280dp width; intentional alignment |
| Motion | 150–220ms standard; no invisible idle animation |
| Normal text contrast | At least 4.5:1 in both themes |

Medium/landscape content is centered and capped at 600dp except graph/canvas surfaces that explicitly opt out.

## 5. System inset contract

- `NavGraph` root shell owns safe drawing and bottom-navigation policy.
- Child scaffolds use zero content insets and consume root padding once.
- No child applies `statusBarsPadding()` or `navigationBarsPadding()` after receiving root safe-area padding.
- Chat/editor surfaces own IME behavior only.
- Modal sheets/dialogs own their own IME behavior.

## 6. Route and state contract

Complex surfaces use Route/Content separation:

- Route collects ViewModel state and performs navigation/side effects.
- Content is a pure composable over immutable state and callbacks.
- Every applicable route has loading, empty, populated, error, and recovery behavior.
- Initial loading never flashes the empty state.
- Errors explain what failed and provide the nearest safe recovery action.
- Empty states offer one useful next action.

## 7. Interaction hierarchy

- One dominant action per surface; at most two visible secondary actions.
- Lower-frequency and destructive actions live in overflow/contextual areas.
- Emoji are not primary UI icons.
- All visual controls expose at least 48dp touch targets.
- Chat streaming follows only while the user is at the live edge. Once the user scrolls up, only **Jump to latest** returns them.
- Voice entry points—dictation, hold-to-talk, and continuous voice—are explicitly labeled and discoverable.

## 8. Visual proof requirement

No screen is considered polished until:

1. the exact APK is installed;
2. Aura is confirmed as the resumed foreground package;
3. light and dark states are captured;
4. relevant loading/empty/populated/error states are inspected;
5. compact, reference, medium, font-scale, and keyboard cases pass;
6. P0/P1 defects are fixed and recaptured.

A screenshot filename is not proof of its pixels or foreground package.
