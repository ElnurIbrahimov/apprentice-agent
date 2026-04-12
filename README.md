# AURA

AURA is a personal AI assistant platform that spans a CLI, FastAPI backend, React web app, browser extension, and Telegram surfaces. This repository contains a lot of real functionality, but it also contains research-heavy systems that are still evolving.

This README is intentionally conservative. It separates what is working in the repo today from what is experimental and what is still roadmap material.

## What AURA Is

- A local-first assistant runtime built around `ApprenticeAgent`, `OllamaBrain`, and an agentic tool loop.
- A multi-surface product with shared memory and orchestration primitives.
- A large experimental codebase that mixes product features, R&D ideas, and infrastructure.

## Status At A Glance

### Implemented

These areas have concrete code paths and are part of the repo's current working surface:

- CLI assistant with chat and one-shot agentic execution.
- FastAPI API server with route modules for chat, status, auth, tools, memory, research, uploads, and related features.
- React web UI under [`web/`](./web) with chat, tools, settings, and insight-oriented views.
- Browser extension under [`extension-src/`](./extension-src) with chat and productivity tooling.
- Core memory stack, conversation persistence, and retrieval helpers.
- Model routing and provider integration across local and cloud-backed model paths.
- Tool execution for files, code search/editing, shell, git, testing, and web fetch/search flows.
- Telegram bot and mini-app related code paths.
- A sizeable automated test suite covering security, routing, memory, CLI behavior, and supporting utilities.

### Experimental

These systems exist in the codebase, but should be treated as unstable, partial, config-dependent, or still under active shaping:

- Proactive awareness, "inner thoughts", and always-on agent behavior.
- Emotion, mood, neuromodulator, and personality subsystems.
- Dreaming, sleep-cycle, and world-model style background cognition.
- Multi-agent orchestration and swarm-style execution.
- Self-improvement and evolution flows.
- Some advanced visual-generation and feedback-loop workflows.
- Several optional providers and integrations that depend on local setup, credentials, or external services.

### Planned

These are near-term directions rather than guarantees:

- Stronger production hardening and clearer deployment profiles.
- More explicit contracts around which routes and subsystems are required in each environment.
- Further decomposition of large orchestration modules.
- More reliable functional tests that assert success instead of tolerating missing or failing behavior.
- Better separation between productized features and research prototypes.

## Repository Layout

- [`aura/`](./aura): core runtime, CLI, cognition systems, tools, memory, routing, integrations.
- [`api/`](./api): FastAPI app, bootstrap code, route modules, API-facing services.
- [`web/`](./web): React frontend.
- [`extension-src/`](./extension-src): browser extension source.
- [`tests/`](./tests): Python test suite.
- [`docs/`](./docs): supporting documentation and notes.

## Running The Project

Setup details vary depending on which surface you want to run, but the common starting points are:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

API:

```bash
python run_web.py
```

Web UI:

```bash
cd web
npm install
npm run dev
```

CLI:

```bash
python main.py
```

The project depends on environment configuration for model providers, auth, and several optional integrations. Expect some features to stay disabled until those variables and services are configured.

## Testing

There are many tests in this repository, but coverage quality is not uniform yet.

- Unit and security-focused tests are generally more trustworthy than broad functional smoke tests.
- Some functional tests still tolerate `404` or `500` responses and should be tightened over time.
- If you are changing startup, routing, permissions, or tool execution behavior, prefer adding narrow regression tests in `tests/` alongside the change.

Typical commands:

```bash
python -m pytest tests -q
ruff check aura api tests
```

## Current Engineering Priorities

- Split oversized orchestration modules by responsibility.
- Fail closed in production when critical routes or middleware are missing.
- Tighten default mutation permissions for shell and git actions.
- Make the README and test suite reflect the real maturity of each subsystem.

## Notes For Contributors

- Treat this as a mixed maturity codebase: some parts are stable application code, others are research infrastructure.
- Prefer small, explicit modules over adding more logic to the existing "god files".
- Keep public behavior stable when refactoring, and add regression tests when hardening behavior.
- Be cautious about turning exploratory subsystems into implied product guarantees.
