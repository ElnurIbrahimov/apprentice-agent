# Aura Android Visual QA

Visual verification is a release gate, not a post-release suggestion.

## 1. Required environment record

Every capture run records:

- commit SHA;
- package and activity;
- APK path and SHA-256;
- versionName/versionCode;
- viewport pixels and density;
- font scale;
- light/dark theme;
- Android API/device profile.

## 2. Foreground proof

After any connected-test run, reinstall the exact APK before screenshots because instrumentation may remove it.

```bash
ADB="$LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe"
"$ADB" install -r app/build/outputs/apk/debug/app-debug.apk
"$ADB" shell am start -W -n com.aura.debug/com.aura.MainActivity
"$ADB" shell dumpsys activity activities
"$ADB" shell uiautomator dump /sdcard/aura.xml
"$ADB" pull /sdcard/aura.xml .hermes/screenshots/aura.xml
```

Pass only when:

- resumed activity contains `com.aura.debug/com.aura.MainActivity` or the debug UI catalog activity;
- UIAutomator root package is `com.aura.debug`;
- launcher/System UI is not foreground.

The screenshot harness must abort if either assertion fails.

## 3. Required viewports

- 360 × 800dp compact;
- 393 × 851dp reference;
- 600 × 960dp medium;
- compact landscape.

Modes:

- light;
- dark;
- font scale 1.0;
- font scale 1.3;
- keyboard open where editable.

## 4. Required state matrix

| Surface | Required states |
|---|---|
| Startup | light, dark, app lock, storage error |
| Onboarding | intro, credential draft, testing, invalid, valid catalog, model chosen, local-only skip |
| Home | loading, empty, populated, data error |
| Chat | no model, empty, loading conversation, populated, streaming, scrolled up, tool, citation, image, error, keyboard |
| Model picker | no provider, loading, stale cache, partial failure, zero models, success, search, long labels |
| Settings | home and each detail route, save/test states, model roles, light/dark |
| Memory | loading, empty, filtered empty, populated, editor, operation error |
| History | loading, empty, search empty, populated, action error |
| Tasks/reminders | loading, empty, overdue, done, editors, operation error |
| Hands | loading, empty, populated, editor stages, running, result, history filters |
| Tools | empty, search empty, grouped, risk/detail states |
| Proactive | empty, events, permission disabled, error |
| Knowledge graph | loading, empty, populated, filtered, node sheet, error |
| Profile/Identity | default, dirty, saving, error, reset confirmation |
| Diagnostics | loading, empty, populated, expanded, error, clear confirmation |
| Voice | listening, thinking, speaking, error, cancel; all three entry modes |
| Quick Ask/widget | no model, input, loading, response, error, config |

## 5. Geometry checks

- Content reserves each system inset exactly once.
- Bottom navigation occupies 60–64dp before the device navigation inset.
- Top bars are 56dp and share a common baseline.
- Model pill never pushes every header action offscreen.
- Composer stays above IME with no unexplained gap.
- No core surface has accidental blank space greater than roughly 30% of usable height.
- Memory controls use no more than 40% of compact height before the list.
- Medium layouts center content at at most 600dp, except canvas opt-outs.

## 6. Interaction checks

- Every visible control has a 48dp touch target.
- Focus and selected states remain visible in both themes.
- Loading does not flash empty content.
- Error recovery reaches the correct destination/action.
- Streaming does not yank a user who scrolled up.
- Voice modes have explicit labels.
- Destructive actions are secondary and confirmed.

## 7. Contrast and motion

- Normal text reaches 4.5:1 contrast.
- Large text and meaningful non-text controls reach at least 3:1.
- Infinite animations stop when the content is not active/visible.
- Motion communicates state and does not delay primary actions.
- Verify with system animator scale at normal and disabled.

## 8. Evidence storage

Store runtime evidence under `.hermes/screenshots/<commit>/`; it remains untracked. Commit only the screenshot matrix documentation and automation harness—not personal provider data, typed secrets, or runtime databases.
