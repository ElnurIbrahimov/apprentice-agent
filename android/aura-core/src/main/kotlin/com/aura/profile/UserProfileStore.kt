package com.aura.profile

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class UserProfileStore @Inject constructor(
    private val dao: UserProfileDao,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val _profile = MutableStateFlow(UserProfile())
    val profile: StateFlow<UserProfile> = _profile.asStateFlow()

    init {
        scope.launch {
            dao.get()?.let { _profile.value = UserProfile.fromEntity(it) }
        }
    }

    suspend fun update(name: String? = null, traits: List<String>? = null, preferences: Map<String, String>? = null, facts: List<String>? = null) {
        val current = _profile.value
        val merged = UserProfile(
            name = name ?: current.name,
            traits = traits ?: current.traits,
            preferences = current.preferences + (preferences ?: emptyMap()),
            facts = facts ?: current.facts,
        )
        _profile.value = merged
        dao.upsert(UserProfile.toEntity(merged))
    }

    fun getSystemPrompt(): String = _profile.value.toSystemPrompt()
}
