Aura Android debug build v0.13.0

Highlights
- Creative Studio: world bible (characters, locations, factions, rules, timeline, outline, simulations) with five forms (novel, short story, screenplay, RPG world, character study).
- New Writer specialist routed by writing/worldbuilding keywords.
- Creative Engine uses the user-configured default model. No hardcoded model IDs.
- Two new agent tools: creative_read_project, creative_add_world_item.
- Document ingestion landed earlier in this branch (PDF, DOCX, TXT, MD, CSV, JSON, YAML, HTML, source files) with provenance.
- Reactive morning-brief / calendar-monitor settings, deep-link payload routing, exact memory undo, edit-history UI, hybrid History search, conversation stats, proactive event tap-to-act, and the rest of the friction-audit cleanup are all integrated.

Verification
- 570 aura-core unit tests, 0 failures.
- 236 app unit tests, 0 failures.
- assembleDebug green.
- Two atomic commits: 5570bdf (engine + tools + specialists + persistence) and 8bdd952 (Home wiring + screens + NavGraph).
