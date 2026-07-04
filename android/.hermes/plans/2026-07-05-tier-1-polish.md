# Tier 1 Polish — COMPLETE

All five Tier 1 structural gaps shipped on `feat/tier-1-friction`.

## Commits

| Hash | Subject | Scope |
|---|---|---|
| `396f09d` (base) | fix(audit): security, reliability and doc corrections from engineering review | pre-Tier 1 review fixes |
| `95e9895` (base) | docs(architecture): refresh anti-features + security/privacy sections | docs |
| `...` | feat(tasks): reminder list/cancel + full task fields UI | Task DB, SetReminderTool, TasksScreen |
| `...` | feat(profile): view/edit/clear UserProfile screen | UserProfileStore, ProfileScreen |
| `eb87744` | feat(proactive): tap-to-act navigation + morning brief actions | Proactive events, notifications, MainActivity deep-link |

Note: exact middle two hashes omitted from this summary; inspect `git log --oneline -5`.

## What changed

1. **Reminder list/cancel**
   - New `ReminderEntity` + `ReminderDao` in TaskDatabase v2
   - `SetReminderTool` persists reminder metadata; `ReminderWorker` deletes row on fire
   - `TasksViewModel` loads upcoming reminders + cancels by WorkRequest id
   - `TasksScreen` shows upcoming reminders section with cancel buttons

2. **Full task fields UI**
   - `AddTaskDialog` captures description, due date/time, priority, tags
   - Task rows render due time, priority chip, tags

3. **UserProfile screen**
   - New `ProfileScreen` with editable name, traits list, facts list
   - New `ProfileViewModel` wired from `SettingsScreen`
   - `UserProfileStore.update` fixed to replace full traits/facts lists

4. **Proactive event tap-to-act**
   - `ProactiveEventEntity` now stores event payload
   - `ProactiveHistoryScreen` navigates to chat/memory/calendar per event type
   - New `ProactiveHistoryViewModel` decodes events deterministically

5. **Morning brief notification actions**
   - Notification adds "Tell me more" and "Snooze 1h" actions
   - `MorningBriefReceiver` + `MorningBriefSnoozeWorker` reschedule in 1 hour
   - MainActivity/NavGraph support `openChat` + `morningBriefSummary` deep-link
   - ChatScreen auto-forwards summary as a user message

## Verification

```bash
cd /d/Aura/android && ./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug
```

Result: BUILD SUCCESSFUL. All existing tests pass. APK rebuilt.
