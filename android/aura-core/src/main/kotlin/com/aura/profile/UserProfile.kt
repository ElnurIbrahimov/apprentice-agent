package com.aura.profile

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString

@Serializable
data class UserProfile(
    val name: String? = null,
    val traits: List<String> = emptyList(),
    val preferences: Map<String, String> = emptyMap(),
    val facts: List<String> = emptyList(),
) {
    companion object {
        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = false }

        fun fromEntity(e: UserProfileEntity): UserProfile = UserProfile(
            name = e.name,
            traits = try { json.decodeFromString<List<String>>(e.traitsJson) } catch (_: Exception) { emptyList() },
            preferences = try { json.decodeFromString<Map<String, String>>(e.preferencesJson) } catch (_: Exception) { emptyMap() },
            facts = try { json.decodeFromString<List<String>>(e.factsJson) } catch (_: Exception) { emptyList() },
        )

        fun toEntity(p: UserProfile): UserProfileEntity = UserProfileEntity(
            name = p.name,
            traitsJson = json.encodeToString(p.traits),
            preferencesJson = json.encodeToString(p.preferences),
            factsJson = json.encodeToString(p.facts),
            lastUpdated = System.currentTimeMillis(),
        )
    }

    fun toSystemPrompt(): String = buildString {
        append("## User Profile\n")
        name?.let { append("Name: $it\n") }
        if (traits.isNotEmpty()) append("Traits: ${traits.joinToString(", ")}\n")
        if (preferences.isNotEmpty()) preferences.forEach { (k, v) -> append("Prefers $k: $v\n") }
        if (facts.isNotEmpty()) {
            append("Facts about the user:\n")
            facts.forEach { append("- $it\n") }
        }
    }
}
