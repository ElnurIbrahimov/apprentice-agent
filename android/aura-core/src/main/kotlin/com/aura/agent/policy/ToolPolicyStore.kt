package com.aura.agent.policy

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

private val Context.toolPolicyPrefs by preferencesDataStore(name = "aura_tool_policies")

/**
 * Persists per-tool [ToolPolicy] in DataStore. Non-secret data only —
 * secrets remain in [com.aura.security.SecureDataStore].
 *
 * Stored as a JSON map of tool name → ToolPolicy. A missing key means
 * the tool uses its built-in default (enabled, no extra restrictions).
 */
@Singleton
class ToolPolicyStore @Inject constructor(
    private val context: Context,
) {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    val allPolicies: Flow<Map<kotlin.String, ToolPolicy>> = context.toolPolicyPrefs.data.map { prefs ->
        prefs[KEY_POLICIES]?.let { raw ->
            runCatching {
                json.decodeFromString<Map<kotlin.String, ToolPolicy>>(raw)
            }.getOrDefault(emptyMap())
        } ?: emptyMap()
    }

    suspend fun getPolicy(toolName: kotlin.String): ToolPolicy? {
        val map = allPolicies.first()
        return map[toolName]
    }

    suspend fun setPolicy(toolName: kotlin.String, policy: ToolPolicy) {
        context.toolPolicyPrefs.edit { prefs ->
            val current = prefs[KEY_POLICIES]?.let { raw ->
                runCatching {
                    json.decodeFromString<Map<kotlin.String, ToolPolicy>>(raw)
                }.getOrDefault(emptyMap())
            } ?: emptyMap()
            val updated = current + (toolName to policy)
            prefs[KEY_POLICIES] = json.encodeToString(updated)
        }
    }

    suspend fun removePolicy(toolName: kotlin.String) {
        context.toolPolicyPrefs.edit { prefs ->
            val current = prefs[KEY_POLICIES]?.let { raw ->
                runCatching {
                    json.decodeFromString<Map<kotlin.String, ToolPolicy>>(raw)
                }.getOrDefault(emptyMap())
            } ?: emptyMap()
            val updated = current - toolName
            prefs[KEY_POLICIES] = if (updated.isEmpty()) "" else json.encodeToString(updated)
        }
    }

    companion object {
        private val KEY_POLICIES = stringPreferencesKey("tool_policies_json")
    }
}