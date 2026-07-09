package com.aura.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Single owner of the user-facing DataStore ("aura_settings").
 *
 * Earlier revisions of the codebase had three separate
 * `Context.auraPrefs` extension declarations across this file,
 * [com.aura.FirstRunGate], and
 * [com.aura.ui.settings.SettingsViewModel], each using the same
 * `preferencesDataStore(name = "aura_settings")` delegate. That
 * happened to work because the AndroidX delegate is process-cached
 * by name, but the duplication hid a worse bug: [FirstRunGate]
 * used `booleanPreferencesKey` while [SettingsViewModel] used
 * `stringPreferencesKey` for the same logical `first_run_complete`
 * field, so writes from the VM never reached the gate's reader.
 *
 * Now consolidated here. Lives in `:aura-core/data` so the
 * [com.aura.backup.BackupManager] (also in `:aura-core`) can
 * inject it without a circular dependency.
 */
private val Context.auraPrefs by preferencesDataStore(name = "aura_settings")
internal val KEY_DEFAULT_MODEL = stringPreferencesKey("default_model")

/**
 * The default chat model used when the user hasn't picked one yet.
 * Pinned in DefaultModelTest to prevent a regression of the 2026-07-09
 * bug where this string included a `:cloud` suffix that does not exist
 * on Ollama Cloud (real model ids are bare — e.g. `deepseek-v4-pro`).
 * The picker refreshes from the live `/v1/models` endpoint on open, so
 * this is only the model used for the very first chat before the
 * user has touched the picker.
 *
 * Public (not `internal`) because the UI module's [ChatUiState] default
 * also reads it, and `internal` is module-scoped in Kotlin.
 */
const val DEFAULT_MODEL = "ollama:deepseek-v4-pro"
internal val KEY_APP_LOCK_ENABLED = booleanPreferencesKey("app_lock_enabled")
internal val KEY_FIRST_RUN_COMPLETE = booleanPreferencesKey("first_run_complete")
internal val KEY_LAST_SEEN_PROACTIVE_AT = longPreferencesKey("last_seen_proactive_at")
/**
 * Proactive worker gates. Both default to true so a fresh install
 * gets the morning brief + calendar monitor out of the box. The
 * Settings screen exposes toggles for each.
 */
internal val KEY_MORNING_BRIEF_ENABLED = booleanPreferencesKey("morning_brief_enabled")
internal val KEY_CALENDAR_MONITOR_ENABLED = booleanPreferencesKey("calendar_monitor_enabled")
internal val KEY_TTS_ENABLED = booleanPreferencesKey("tts_enabled")
internal val KEY_INCOGNITO_DEFAULT = booleanPreferencesKey("incognito_default")
internal val KEY_THEME_MODE = stringPreferencesKey("theme_mode")
internal val KEY_CUSTOM_IDENTITY = stringPreferencesKey("custom_identity")
internal val KEY_SPECIALIST_OVERRIDES = stringPreferencesKey("specialist_overrides")
internal val KEY_MORNING_BRIEF_HOUR = intPreferencesKey("morning_brief_hour")
internal val KEY_SPECIALIST_TOOL_OVERRIDES = stringPreferencesKey("specialist_tool_overrides")

@Singleton
class UserPreferences @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    val defaultModel: Flow<String> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_DEFAULT_MODEL] ?: DEFAULT_MODEL
    }

    /**
     * Whether the user has enabled biometric app lock. When true,
     * the main activity gates on a [androidx.biometric.BiometricPrompt]
     * challenge before showing the rest of the UI. Stored locally;
     * defaults to false (no lock) for a fresh install.
     */
    val appLockEnabled: Flow<Boolean> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_APP_LOCK_ENABLED] ?: false
    }

    /**
     * Whether the user has completed first-run onboarding. Read by
     * [com.aura.FirstRunGate] to decide whether to show the wizard
     * vs. the main app. Written by
     * [com.aura.ui.settings.SettingsViewModel] after the user saves
     * their first API key.
     */
    val firstRunComplete: Flow<Boolean> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_FIRST_RUN_COMPLETE] ?: false
    }

    /**
     * Last time the user opened the proactive history screen (or
     * dismissed the proactive event card on Home). Used by
     * [com.aura.proactive.ProactiveEvents] to compute the
     * "unread proactive events" badge on Home. Default 0L means
     * everything since the Unix epoch counts as unread on a fresh
     * install — which is the right behavior for the first session.
     */
    val lastSeenProactiveAt: Flow<Long> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_LAST_SEEN_PROACTIVE_AT] ?: 0L
    }

    /**
     * Whether the user wants the daily morning brief. When false,
     * [com.aura.proactive.ProactiveBootstrap] skips scheduling the
     * WorkManager job and cancels any in-flight schedule. Default
     * true — opt-out, not opt-in.
     */
    val morningBriefEnabled: Flow<Boolean> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_MORNING_BRIEF_ENABLED] ?: true
    }

    /**
     * Whether the user wants the calendar monitor foreground
     * service running. When false, [com.aura.proactive.ProactiveBootstrap]
     * does not start the service on app launch. The user must
     * toggle this off to dismiss the persistent notification.
     * Default true — opt-out.
     */
    val calendarMonitorEnabled: Flow<Boolean> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_CALENDAR_MONITOR_ENABLED] ?: true
    }

    /**
     * Whether TTS (text-to-speech) is enabled. The chat screen
     * auto-reads assistant responses aloud when this is true.
     * Defaults to true for a fresh install; persisted so the user's
     * mute preference survives app restarts.
     */
    val ttsEnabled: Flow<Boolean> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_TTS_ENABLED] ?: true
    }

    /**
     * Whether incognito mode is on by default for new sessions.
     * The chat screen's incognito toggle still works on top of this
     * default — it's the starting value for each new app launch.
     * Defaults to false (off) for a fresh install. A privacy-conscious
     * user can set this to true in Settings so every session starts
     * in incognito without re-toggling.
     */
    val incognitoDefault: Flow<Boolean> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_INCOGNITO_DEFAULT] ?: false
    }

    /**
     * Theme mode: "system" (follow system dark/light), "light", or
     * "dark". Defaults to "system" so a fresh install respects the
     * device's current setting. The user can override in Settings.
     */
    val themeMode: Flow<String> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_THEME_MODE] ?: "system"
    }

    /**
     * User-provided custom system prompt / identity override. When
     * non-blank, the [com.aura.agent.Brain] prepends this to the
     * default [Brain.IDENTITY], giving the user a way to personalize
     * Aura's persona without editing code. Defaults to blank (use
     * built-in identity only).
     */
    val customIdentity: Flow<String> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_CUSTOM_IDENTITY] ?: ""
    }

    /**
     * JSON map of specialist name → custom system prompt. When a
     * specialist is selected, its entry here overrides the built-in
     * [com.aura.agent.Specialist.systemPrompt]. Empty string or
     * missing key = use the built-in prompt.
     */
    val specialistOverrides: Flow<String> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_SPECIALIST_OVERRIDES] ?: "{}"
    }

    /**
     * Hour of day (0-23) for the morning brief. Default 7 (7am).
     * The user can change this in Settings → Morning brief time.
     */
    val morningBriefHour: Flow<Int> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_MORNING_BRIEF_HOUR] ?: 7
    }

    /**
     * JSON map of specialist name → set of allowed tool names.
     * When a specialist is selected, its toolsAllowed is replaced
     * with this set if present. Empty map = use built-in defaults.
     */
    val specialistToolOverrides: Flow<String> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_SPECIALIST_TOOL_OVERRIDES] ?: "{}"
    }

    suspend fun setDefaultModel(model: String) {
        context.auraPrefs.edit { it[KEY_DEFAULT_MODEL] = model }
    }

    suspend fun setAppLockEnabled(enabled: Boolean) {
        context.auraPrefs.edit { it[KEY_APP_LOCK_ENABLED] = enabled }
    }

    suspend fun setFirstRunComplete(complete: Boolean) {
        context.auraPrefs.edit { it[KEY_FIRST_RUN_COMPLETE] = complete }
    }

    suspend fun setLastSeenProactiveAt(timestamp: Long) {
        context.auraPrefs.edit { it[KEY_LAST_SEEN_PROACTIVE_AT] = timestamp }
    }

    suspend fun setMorningBriefEnabled(enabled: Boolean) {
        context.auraPrefs.edit { it[KEY_MORNING_BRIEF_ENABLED] = enabled }
    }

    suspend fun setCalendarMonitorEnabled(enabled: Boolean) {
        context.auraPrefs.edit { it[KEY_CALENDAR_MONITOR_ENABLED] = enabled }
    }

    suspend fun setTtsEnabled(enabled: Boolean) {
        context.auraPrefs.edit { it[KEY_TTS_ENABLED] = enabled }
    }

    suspend fun setIncognitoDefault(enabled: Boolean) {
        context.auraPrefs.edit { it[KEY_INCOGNITO_DEFAULT] = enabled }
    }

    suspend fun setThemeMode(mode: String) {
        context.auraPrefs.edit { it[KEY_THEME_MODE] = mode }
    }

    suspend fun setCustomIdentity(identity: String) {
        context.auraPrefs.edit { it[KEY_CUSTOM_IDENTITY] = identity }
    }

    suspend fun setSpecialistOverrides(json: String) {
        context.auraPrefs.edit { it[KEY_SPECIALIST_OVERRIDES] = json }
    }

    suspend fun setSpecialistToolOverrides(json: String) {
        context.auraPrefs.edit { it[KEY_SPECIALIST_TOOL_OVERRIDES] = json }
    }

    suspend fun setMorningBriefHour(hour: Int) {
        context.auraPrefs.edit { it[KEY_MORNING_BRIEF_HOUR] = hour.coerceIn(0, 23) }
    }

    /**
     * Suspend helper to read the current first-run flag as a
     * one-shot. Used by callers that need a synchronous-style read.
     */
    suspend fun isFirstRunComplete(): Boolean = firstRunComplete.first()
}