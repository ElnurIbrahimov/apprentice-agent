# Tier 1 Polish — Reminder list, Task fields, Profile UI, Proactive tap-to-act, Morning-brief actions

Goal: close the five minimum-surgery structural gaps where the backend already exists but the UI/control surface is missing.

## 1. Reminder list / cancel (TasksScreen)
- Add `ReminderEntity` + `ReminderDao`, include in `TaskDatabase` (version 2).
- `SetReminderTool`: insert a reminder row with work-request id, message, trigger time.
- `ReminderWorker`: convert to `@HiltWorker` + `AssistedInject`, delete the reminder row on fire.
- `TasksViewModel`: load upcoming reminders from `ReminderDao`, expose `cancelReminder(id)`.
- `TasksScreen`: render a collapsible "Upcoming reminders" section with message, time, and cancel button.
- Tests: ReminderDao + ViewModel.

## 2. Surface task fields (TasksScreen)
- Expand `AddTaskDialog` to capture: description (multiline), dueAt (date+time picker), priority (0-3 chips), tags (comma-separated).
- Update `TasksViewModel.add()` to write all fields.
- Render due time, priority badge, tags on each task row.
- Tests: ViewModel add with full fields.

## 3. UserProfile view / edit / clear (Settings or new tab)
- Add a new `ProfileScreen` route in `NavGraph` reachable from Settings.
- `ProfileViewModel`: read/write `UserProfileStore` (name, traits, facts).
- UI: editable name, list of traits with delete, list of facts with delete, "Clear profile" confirmation.
- Tests: ViewModel + store integration.

## 4. Proactive event tap-to-act
- Add `onEventTap` to `ProactiveHistoryScreen` + `EventRow`.
- Route by event type: CalendarEventSoon → open system calendar; MemoryDecayWarning → ChatScreen with "review fading memory"; MorningBrief → ChatScreen with brief context.
- Tests: navigation mapping.

## 5. Morning-brief notification actions
- `MorningBriefWorker`: add notification actions "Tell me more" and "Snooze 1h".
- "Tell me more" launches ChatScreen with the brief context pre-filled as a user message.
- "Snooze 1h" reschedules a one-shot `MorningBriefWorker` for +1h (or a lightweight snooze worker).
- Tests: action intent extras + reschedule.

## Verification
- `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug`
- Target: 0 failures, 0 new lint errors.
