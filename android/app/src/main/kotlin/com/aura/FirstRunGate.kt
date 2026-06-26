package com.aura

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

private val Context.auraPrefs by preferencesDataStore(name = "aura_settings")
private val KEY_FIRST_RUN = booleanPreferencesKey("first_run_complete")

/**
 * Gate that blocks the main UI until the user has completed onboarding.
 * Reads a simple boolean flag from DataStore — the same flag that
 * SettingsViewModel writes when the user enters their first API key.
 */
@Singleton
class FirstRunGate @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    suspend fun isFirstRunComplete(): Boolean =
        context.auraPrefs.data.map { it[KEY_FIRST_RUN] ?: false }.first()

    suspend fun markComplete() {
        context.auraPrefs.edit { it[KEY_FIRST_RUN] = true }
    }
}
