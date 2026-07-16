# Artifacts and Events

The beyond-SOTA plan separates **metadata** (Room) from **bytes** (files).

## Artifact

An artifact is any durable unit of creative or operational output. It has:

- A stable UUID
- A project/run association
- A kind (text, image, audio, video, 3D, document, reference, export)
- A current revision ID
- A content hash
- A storage URI (app-private file, never a blob in Room)
- Provenance: provider prefix, model ID, prompt, settings
- Status: pending, ready, failed, archived

Artifacts are immutable once committed. Revisions are append-only. Branches
create divergent revision chains from a shared parent.

## Event Ledger

Every agent action (tool call, subagent spawn, workflow step, approval, verification)
appends an event to an ordered, append-only ledger. Events carry:

- Correlation ID (the run)
- Step ID
- Parent event ID (for parallel branches)
- Timestamp
- Type (tool_call, tool_result, approval_requested, approval_decided, checkpoint, etc.)
- Redacted payload (never secrets)

The event ledger is the source of truth for replay, debugging, and audit.

## Rules

1. Never store binary media in Room. Store the URI + hash.
2. Never delete an artifact revision. Archive instead.
3. Every event must be attributable to a run and step.
4. Events are append-only. Corrections create new events, not mutations.
5. Replay of the event ledger must reproduce the same final state.