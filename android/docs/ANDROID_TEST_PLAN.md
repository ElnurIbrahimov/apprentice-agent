# Android Test Plan

How to manually exercise every feature of Aura on a real device or emulator.

## 0. Install

```bash
# Build + install on connected device
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 1. First run

1. Open Aura.
2. Accept the permissions the app requests (microphone, location, calendar, contacts, notifications).
3. Tap **Settings** → paste an API key (Ollama Cloud recommended) → tap the chip to set the default model.
4. Tap the **Chat** tab.

## 2. Test the basics (chat)

| # | Action | Expected |
|---|---|---|
| 1 | Tap the chat header (model name) | Bottom sheet appears with model picker |
| 2 | Pick a model | Sheet dismisses, header shows new name |
| 3 | Type "hello" | Status: ●●● indicator, then streaming text |
| 4 | Tap send | Response streams in, TTS reads it back |

## 3. Test voice

1. Tap the mic icon in the input bar.
2. If prompted, allow microphone.
3. VoiceOverlay opens with pulsing red mic.
4. Speak: "set a reminder for 3pm to call mom".
5. Tap stop (or wait for auto-end).
6. Transcript is sent to the agent.
7. Agent dispatches `set_reminder` → tool card shows.
8. Reminder fires at 3pm with notification "⏰ call mom".

## 4. Test memory

1. In chat, say "I prefer dark mode everywhere".
2. Agent calls `remember` tool → tool card appears.
3. Tap **Memory** tab.
4. You see "user prefers dark mode everywhere" with category dot, age, decay.
5. Tap the trash icon to forget.
6. Confirm the entry disappears.

## 5. Test auto-TTS

1. Ensure the volume icon in the chat header is filled (not muted).
2. Send any message.
3. After the agent responds, the answer is spoken aloud via the device's TTS engine.
4. Tap the volume icon to mute.
5. Send another message — no TTS.

## 6. Test calendar

1. Open the system calendar app and create an event 20 minutes from now.
2. In Aura chat, say "what's on my calendar today".
3. Agent calls `calendar_read` → returns the event.
4. Wait — the calendar monitor runs every 5 minutes, so within 5 minutes
   you should see a notification "📅 [event name] starts in N minutes".

## 7. Test the morning brief

The morning brief is scheduled for 7am. To test immediately:

1. Open Settings → enable "Morning brief" (TODO: add this toggle).
2. Use adb to trigger the job:
   ```bash
   adb shell cmd jobscheduler run -f com.aura 0
   ```
3. Within seconds, a notification appears with the brief.

## 8. Test the home screen

1. Tap **Home** tab.
2. You see "Good morning" (or whatever time it is).
3. After chatting for a while, you see the recent memories card populated.
4. After creating tasks, you see the open tasks card.

## 9. Verify the build

```bash
./gradlew :aura-core:testDebugUnitTest   # 32 unit tests pass
./gradlew :app:testDebugUnitTest         # 3 unit tests pass
./gradlew :app:assembleDebug             # APK builds
```

## 10. Common issues

- "No providers configured" → add an API key in Settings.
- Reminder doesn't fire → check notification permissions.
- Mic doesn't work → check the permission.
- TTS is silent → check the device volume and that the volume icon in the chat header is on.
