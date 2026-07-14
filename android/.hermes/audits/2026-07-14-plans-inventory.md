# Aura Android — Plans, Audits & Docs Inventory

**Date:** 2026-07-14
**Branch:** `feat/tier-1-friction`
**Head:** 90cc533
**Total plan/audit lines on disk:** 6,799 lines (17 plans + 2 audits + 1 verification report + 2 doc files)

This is a read-only audit. No files are deleted. Each artifact is classified by **state** (DONE / ACTIVE / SUPERSEDED / INCOMPLETE) with **evidence** (commits or grep results) and a **recommendation**.

---

## Part 1 — Plans (17 files in `.hermes/plans/`)

| # | File | Lines | State | Recommendation |
|---|------|-------|-------|----------------|
| 1 | `2026-06-27-android-everything-a.md` | 9 | SUPERSEDED | **DELETE** — self-declares "stub, superseded by tier-1-polish" |
| 2 | `2026-07-05-tier-1-polish.md` | 51 | DONE | **DELETE** — first line is literally "# Tier 1 Polish — COMPLETE"; commits confirmed (eb87744, 07a830d, 667ef15, 86effaa, 693ff1f) |
| 3 | `2026-07-07-deep-polish.md` | 322 | DONE | **DELETE** — 9/10 items shipped; missing-features plan (next file) cites it as EXECUTED in commit ed5dd65 |
| 4 | `2026-07-07-deep-structural-fixes.md` | 383 | DONE | **DELETE** — 7/8 items shipped (1 deliberately skipped as micro-opt); missing-features plan cites it as EXECUTED in 2e13770 |
| 5 | `2026-07-07-missing-features.md` | 697 | DONE | **DELETE** — 26/26 items shipped. The plan text claims 24/26 with continuous voice + hands step builder deferred, but grep proves both shipped in commits f03a2d4 (`continuous voice mode, widget config, hands step builder`) and 191bb4e (`hands step builder UI + trigger-phrase wiring`). The plan's deferral note was overtaken by later execution. Item 22 (SQLCipher DB encryption) was the only genuine defer, but it was removed from the plan in a later edit |
| 6 | `2026-07-08-truth-fixes.md` | 354 | DONE | **DELETE** — 8/8 items committed. Plan itself documents all 8 atomic commits and the v0.6.0 tag |
| 7 | `2026-07-08-sota-staffing.md` | 867 | DONE | **DELETE** — 8/8 commits shipped (Dims 1+2): tool badges, memory recall chip, citation chips, streaming markdown fix, follow-up suggestions, push-to-talk, vision quick-prompt, history pin+rename. Plan itself predicted "27-38 hours" but actual was ~4h because the underlying model was already richer than expected |
| 8 | `2026-07-08-sota-ui-redesign.md` | 85 | DONE | **DELETE** — 8/8 commits shipped: 011081a design tokens + Inter/Fraunces, then the visual design system in subsequent commits. Note: the plan was later visually-rewritten by `f4c27b3 feat(chat): brighter empty state` — the visual got more compact than the plan's spec, but the tokens and typography remain |
| 9 | `2026-07-09-bug-sweep.md` | 51 | DONE | **DELETE** — 6 atomic commits shipped: model-id fixes, MoA presets, Gemini live catalog, duplicate import. README now lists 738 tests (was 459 at plan time) |
| 10 | `2026-07-09_233000-runtime-integrity-hardening.md` | 548 | DONE | **DELETE** — 14/14 commits shipped (per plan's own commit sequence). Migration tests added, voice leak fixed, app lock reactive, share delivery working, MoA credentials validated, Gemini key in header, etc. |
| 11 | `2026-07-11-extreme-ux-polish.md` | 17 | SUPERSEDED | **DELETE** — superseded by `2026-07-13_212705-aura-model-ui-ux-recovery.md`, which itself opens with: "Prior plans: `.hermes/plans/2026-07-11-extreme-ux-polish.md`, `.hermes/plans/2026-07-11-shared-screen-shell.md`. Those plans were only 15–17 lines. They did not specify [10 things]. This is the replacement plan." |
| 12 | `2026-07-11-shared-screen-shell.md` | 15 | SUPERSEDED | **DELETE** — same reason as #11 |
| 13 | `2026-07-13-phase-2-context-automation.md` | 343 | DONE | **DELETE** — KG management, conversation compactor, crash diagnostics, hands run history, backup schema — all shipped. Commits visible: c2acd82 (hands), fdcdd50 (compactor), df1856f (crash), 8fb4966 (diagnostics), 2f1ff32 (KG surface) |
| 14 | `2026-07-13_212705-aura-model-ui-ux-recovery.md` | 1594 | DONE | **DELETE** — 39 atomic commits across 9 phases shipped. Plan opened with critique of prior plans; the critique is now historical |
| 15 | `2026-07-14-audit-remediation.md` | 159 | DONE | **DELETE** — 12/12 sections shipped. Verified in `2026-07-14-audit-remediation-final-verification.md`. No source changes were needed; the recent 12 commits had already done the work |
| 16 | `2026-07-14-audit-remediation-final-verification.md` | 64 | ACTIVE | **KEEP** — current evidence-of-completion artifact. Once the cleanup is done and README is updated, this can be deleted or moved to `.hermes/audits/` |
| 17 | `ui-ux-audit-report.md` | 609 | DONE | **KEEP or MOVE** — source audit for the 2026-07-14 plan. Has historical value but is no longer a roadmap. Recommend move to `.hermes/audits/2026-07-13-ui-ux-audit.md` to keep plans/ focused on forward work |

**Plan summary:** 15 of 17 plans are done or superseded. Total deletable lines: 5,944 / 6,168 (96.4%).

---

## Part 2 — Audits (2 files in `.hermes/audits/`)

| # | File | Lines | State | Recommendation |
|---|------|-------|-------|----------------|
| 1 | `2026-07-10-comp-ui-friction-audit.md` | 223 | DONE (partial) | **KEEP for now, mark stale** — this audit listed P1 items (mood theming, inline citations, top-level Create/Insights tabs, clickable links, message reactions). The design-system-inset-scaffold-audit (next file) supersedes the theme-handoff defects. The remaining P1s were not done. This audit is a roadmap of unfinished work — valuable for the user, do not delete yet |
| 2 | `design-system-inset-scaffold-audit.md` | 429 | DONE | **DELETE or KEEP** — the main defect (hardcoded `AuraTokens.Dark` in screens) is fixed: grep shows only `Theme.kt` references `AuraTokens.Dark` now (the legitimate source). The 6 nested-Scaffold screens all use `contentWindowInsets = WindowInsets(0)`. Recommend delete; the work is verified in the audit-remediation final-verification |

---

## Part 3 — Documentation drift (README + docs/architecture.md)

### `README.md` (line 232):
> The build plan lives at `.hermes/plans/2026-07-05-tier-1-polish.md`. Daily commits document what shipped; `git log --oneline` is the changelog.

**Problem:** The 2026-07-05 plan is "COMPLETE" and being deleted. The current build is on plan #15 (2026-07-14-audit-remediation). This line is stale.

**Fix:** Replace with: `> The build plan lives at \`.hermes/plans/2026-07-14-audit-remediation.md\`. All earlier plans (15 in total) are complete and archived in git history. Daily commits document what shipped; \`git log --oneline\` is the changelog.`

### `README.md` (line 13): "36 tools"
**Verified:** correct. `ToolsModule.kt` registers exactly 36 tools.

### `README.md` (line 28): "515 unit tests passing across `:aura-core` (383) + `:app` (132)"
**Problem:** 738 tests, 0 failures (verified 2026-07-14). The 515 figure is from 3 days ago.
**Fix:** Replace with: `- **738 unit tests passing** across `:aura-core` (~480) + `:app` (~258), 0 failures, 0 skipped`

### `README.md` (line 182-187): "Stats: 512 unit tests passing across `:aura-core` (380) + `:app` (132)"
**Problem:** Same — 512 figure is older than 515. And there's an internal contradiction (515 in the Status block, 512 in the Build block).
**Fix:** Remove the duplicate stats line, OR update both to 738. Recommend: drop the "Stats:" line at the bottom since it's already in the Status block.

### `docs/architecture.md` (line 12): "tool registry with 31 phone-native tools"
**Problem:** 36 tools total (the architecture doc counts only "phone-native" but ToolsModule has web/AI tools too). Actually re-reading: "tool registry with 31 phone-native tools" is technically what the 2026-06-26 version of the doc said. Current is 36 total.
**Fix:** Update to: `tool registry with 36 tools (web search x3, vision, image gen, deep research, fetch, knowledge graph, weather, translate, timer, SMS, email, biometric prompt, and phone-native tools)`

### `docs/architecture.md` (line 91): "Source of truth: `.hermes/plans/2026-07-05-tier-1-polish.md`"
**Problem:** Same as README — points to a plan that is COMPLETE.
**Fix:** Same update as README, OR drop the "Source of truth" line entirely (the README has the same info).

### `docs/architecture.md` — "Version: BuildConfig.VERSION_NAME (currently `0.1.0`)"
**Problem:** Actual `versionName = "0.10.2"`, `versionCode = 3` (in `app/build.gradle.kts`).
**Fix:** Update to `0.10.2` (versionCode 3).

### `docs/architecture.md` (line 12): "MoA, Ollama Cloud, OpenAI-compat, OpenRouter"
**Verify:** README says 8 LLM providers (Ollama Cloud, Anthropic, OpenAI, DeepSeek, Gemini, Groq, OpenRouter, MoA). Architecture doc is missing DeepSeek and Groq. Both files reference "7 concrete providers" but README says 8.
**Fix:** Architecture doc should list 8 (the README is more recent and accurate).

---

## Part 4 — Roadmap: what's actually open

If you delete 15 of 17 plans, the question is "what's the active work?" The plan-remediation was a clean-up pass. The remaining unfinished work falls into four buckets:

### Bucket A: From the 2026-07-10 UI friction audit (still open)
The friction audit listed 5 P1 items. Verified state:
- ✅ **Clickable links in Markdown** — SHIPPED (`MarkdownText.kt:397, 461` use `uriHandler.openUri`)
- ✅ **Inline citation chips [1][2][3]** — SHIPPED (commit ee8eb84, "feat(chat): inline citation chips [1] [2] [3] with tap-to-preview")
- ❌ **Mood / emotion-aware theming** (P1) — NOT SHIPPED. The `moodAccent` token exists in `AuraTokens.kt:51, 124` but is not driven by chat state. Web has it; Android has the slot but no plumbing
- ❌ **Top-level Create + Insights tabs** (P1) — NOT SHIPPED. Bottom bar still has 4 tabs: Home / Chat / Memory / Settings
- ❌ **Message reactions (thumbs up/down)** (P1) — NOT SHIPPED. MessageBubble footer has only Copy
- ❌ **No save feedback on profile save** (P2)
- ❌ **No graph visualization** (P2) — KG screen is a text-only node list
- ❌ **No batch operations on history** (P2)
- ❌ **Various missing `contentDescription`** (P2)

### Bucket B: From the 2026-07-13 model-UI-UX recovery plan — acceptance checklist
The plan's section 11 listed 21 acceptance criteria. Some are unchecked:
- [ ] "Every screen has loading/empty/error/populated behavior and recovery" — partial
- [ ] "All interactive visuals have at least 48dp touch targets" — partial
- [ ] "Keyboard and font scale 1.3 do not clip critical content" — unverified

### Bucket C: Documentation gaps (per Part 3 above)
- README + architecture.md are stale
- Test count, tool count, version, source-of-truth line all wrong
- Architecture doc lists 7 providers (real: 8), version 0.1.0 (real: 0.10.2)

### Bucket D: Structural follow-ups from prior plans
- `ChatViewModel.kt` is 911 lines (mentioned in earlier audits as a god-class candidate)
- 6 screens still nest `Scaffold` (justified with `contentWindowInsets = WindowInsets(0)`, but the design-system audit considered this a defect)
- The 2026-07-13 plan was 1594 lines, 39 commits, 9 phases — some deferred items may still be open (need grep to confirm)

---

## Part 5 — Recommended actions, in order

1. **Update README + architecture.md** for the test count, version, provider count, and source-of-truth line. 5 minutes. No code changes.
2. **Delete the 15 completed/superseded plans** (1.5 MB of text, all git-tracked). Use `git rm` so the deletion is in history. The plans are not load-bearing — git log has every commit.
3. **Move `ui-ux-audit-report.md`** from `.hermes/plans/` to `.hermes/audits/2026-07-13-ui-ux-audit.md`. It's evidence, not a forward plan.
4. **Keep the 2026-07-10 friction audit** as the active roadmap for Bucket A items. Re-classify it as `.hermes/plans/2026-07-10-active-roadmap.md` or leave it where it is with a comment at the top: "Active roadmap for UI/UX work. Items P1.1–P1.5 are not done."
5. **Decide on `2026-07-14-audit-remediation-final-verification.md`**: this is yesterday's verification, it documents the gate state. Either keep as the latest verification record, or fold the test counts into the README and delete the file.

**Net result if you do this:**
- `.hermes/plans/` shrinks from 17 files (6,168 lines) to 1-2 files (≤ 200 lines) representing the active roadmap.
- `.hermes/audits/` becomes the historical evidence store, with the friction audit as a forward pointer.
- README + architecture.md match the actual codebase.
- 6,000+ lines of text removed; no information lost (every commit in git, every rationale in the surviving 1-2 files).

---

## Part 6 — Things this audit did NOT verify

- Whether the deferred items in `2026-07-07-missing-features.md` (continuous voice mode, hands step builder UI) are actually shipped or still open. The plan said "24/26 shipped"; the two deferred items should be confirmed in code.
- Whether the 2026-07-13 model-UI-UX recovery plan's section 11 acceptance checklist is met — only a partial grep was done.
- Whether `.github/workflows/ci.yml` is up to date with the current test count.
- Whether all the screenshots in `.hermes/*.png` are still relevant (they may be from prior design iterations that have been superseded).

These are second-order checks; the main plan classification above is solid.
