package com.aura

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

private val Context.auraPrefs by preferencesDataStore(name = "aura_settings")
internal val KEY_DEFAULT_MODEL = stringPreferencesKey("default_model")

@Singleton
class UserPreferences @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    val defaultModel: Flow<String> = context.auraPrefs.data.map { prefs ->
        prefs[KEY_DEFAULT_MODEL] ?: "ollama:deepseek-v4-pro:cloud"
    }

    suspend fun setDefaultModel(model: String) {
        context.auraPrefs.edit { it[KEY_DEFAULT_MODEL] = model }
    }
}
