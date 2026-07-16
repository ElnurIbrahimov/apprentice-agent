# Privacy Boundaries

## Data classification

| Class | Examples | Storage | Backup |
|-------|----------|---------|--------|
| Secret | API keys, OAuth tokens, passwords | SecureDataStore (Keystore) | Never |
| Personal | Memories, KG, profile, beliefs, taste signals | Room (local) | Encrypted export, user-initiated |
| Artifact | Creative projects, media, canon | Room metadata + app-private files | `.aura` archive (no secrets) |
| Ephemeral | Ground frames, raw UI snapshots, streaming tokens | In-memory only | Never |
| Public | Tool definitions, provider lists, model catalogs | Room/cache | N/A |

## Ground rules

1. Raw Ground frames (screen/camera/mic) are never persisted unless the user explicitly saves an artifact from them.
2. Accessibility UI snapshots are redacted before reaching the model: password fields, sensitive nodes, and disallowed packages are stripped.
3. Conversation history and memory are local-only. No cloud sync.
4. Backup exports never include secrets, embeddings, or running job state.
5. Remote MCP/Bridge tools cannot read local secrets.
6. The user can inspect, edit, and delete any stored personal data at any time.
7. Taste/preference signals are local-only with explicit retention controls.