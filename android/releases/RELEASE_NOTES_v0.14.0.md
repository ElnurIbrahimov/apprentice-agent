# Aura Android v0.14.0 — Providers + Skills

This release adds **12 new LLM and capability providers** with live model
discovery, a new **user-authored Skills** subsystem, and updates the Home
secondary-actions row to surface both.

## New chat LLM providers (live model catalog, OpenAI-compatible)

- Mistral AI
- xAI Grok
- Together AI
- Cerebras
- NVIDIA NIM (build.nvidia.com)
- Meta Llama (api.llama.com)
- Agnes AI
- ChatGPT subscription via OAuth bearer (paste token from `codex login`)

## Custom OpenAI-compatible endpoint

A "Custom Endpoint" provider that lets the user add a base URL + API key
at runtime and pick models from the live `/models` response. The door
to any OpenAI-compatible host — self-hosted vLLM, LM Studio, OpenRouter,
a custom gateway — without shipping a new Provider class.

## New non-chat capability providers

- **Exa** — neural web search.
- **Jina Reader** — URL-to-text / search.
- **ElevenLabs** — text-to-speech.
- **Stability AI** — image generation.
- **Kling AI** — text-to-video (async, polled).
- **World Labs Marble** — text-to-3D world (async, polled).

Each one registers in Hilt via the new `capabilities/di/CapabilityModule`
multibindings; adding a new capability backend is a 3-step recipe
(impl class + @Binds + prefix in `ProviderKeys.PREFIXES`).

## New: Skills subsystem

A Skills hub where the user authors reusable instruction modules that
the agent can invoke on demand.

- Each skill: name + description + free-form markdown body.
- Persisted as a JSON envelope in DataStore.
- Agent invokes by name via the new `use_skill` tool — the body
  becomes part of the next LLM context.
- Skills UI: list / detail / create dialog with Material 3.
- Home → secondary actions row → "Skills" card with live count.

## Foundations

- `capabilities/CapabilityProvider`, `CapabilityKind`, `CapabilityRegistry`
  as the parallel registry for non-chat APIs.
- `capabilities/http/CapabilityHttp` (shared JSON / SSE / error helpers).
- `capabilities/http/JsonTreeExtensions` — null-safe tree navigation.
- `ToolCategories.SKILLS = "skills"` with 📚 icon.
- `SETTINGS_CREDENTIAL_SPECS` extended to surface all new providers.

## Verification

- Aura-core: 593 unit tests, 0 failures.
- App: 236 unit tests, 0 failures.
- `./gradlew :app:assembleDebug` green.
- Pushed to `origin/feat/tier-1-friction`.
