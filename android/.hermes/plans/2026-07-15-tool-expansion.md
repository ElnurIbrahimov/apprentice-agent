# 2026-07-15-tool-expansion.md

Add 6 requested tool categories to Aura Android.

1. **TTS tool** (`tts_speak`) — speaks text via platform TTS or ElevenLabs if configured. Returns audio base64 (ElevenLabs) or queues platform TTS.
2. **Clipboard tools** (`clipboard_read`, `clipboard_write`) — read/write plain text via `ClipboardManager`.
3. **Screenshot tool** (`capture_screen`) — uses `MediaProjection` API; requires foreground activity + permission grant. Returns base64 JPEG.
4. **In-app browser tool** (`open_browser_tab`) — launches Chrome Custom Tab for a URL. Adds `androidx.browser` dependency.
5. **Background email tool** (`send_email_background`) — SMTP send. Adds SMTP config card in Settings (host/port/user/password/from). Tool reads config from DataStore.
6. **Remote file tools** (`http_file_read`, `http_file_write`) — generic HTTP GET/PUT file sync. Reads/writes text content to a URL. Serves as cloud/background-file bridge when URL is WebDAV/S3-presigned.

Files to touch:
- `gradle/libs.versions.toml`: add `androidx-browser`.
- `app/build.gradle.kts`: add browser dependency.
- `aura-core/src/main/kotlin/com/aura/tools/`: 8 new tool files.
- `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt`: register new tools.
- `aura-core/src/main/kotlin/com/aura/security/`: add `ScreenshotActivityHolder` (mirror `BiometricActivityHolder`).
- `app/src/main/kotlin/com/aura/MainActivity.kt`: bind holder lifecycle.
- `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt`: add SMTP config fields + specs.
- `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt`: add SMTP config card.
- `app/src/main/AndroidManifest.xml`: add `android.permission.RECORD_AUDIO`? No, screenshot needs `MediaProjection` intent. Add `android.permission.INTERNET` already present. Custom Tabs doesn't need manifest.
- Add unit tests for tools that are testable (clipboard read/write, TTS param parsing, remote file). Screenshot/browser/SMTP will be harder to unit-test.

Verification: `:aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug :app:lintDebug` green.
