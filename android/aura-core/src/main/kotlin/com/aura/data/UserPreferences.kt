package com.aura.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
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
internal val KEY_APP_LOCK_ENABLED = booleanPreferencesKey("app_lock_enabled")
internal val KEY_FIRST_RUN_COMPLETE = booleanPreferencesKey("first_run_complete")

@Singleton
class UserPreferences @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    val defaultModel: Flow<String> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_DEFAULT_MODEL] ?: "ollama:deepseek-v4-pro:cloud"
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

    suspend fun setDefaultModel(model: String) {
        context.auraPrefs.edit { it[KEY_DEFAULT_MODEL] = model }
    }

    suspend fun setAppLockEnabled(enabled: Boolean) {
        context.auraPrefs.edit { it[KEY_APP_LOCK_ENABLED] = enabled }
    }

    suspend fun setFirstRunComplete(complete: Boolean) {
        context.auraPrefs.edit { it[KEY_FIRST_RUN_COMPLETE] = complete }
    }

    /**
     * Suspend helper to read the current first-run flag as a
     * one-shot. Used by callers that need a synchronous-style read.
     */
    suspend fun isFirstRunComplete(): Boolean = firstRunComplete.first()
}
