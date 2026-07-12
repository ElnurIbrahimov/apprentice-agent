package com.aura.profile

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class UserProfileStore @Inject constructor(
    private val dao: UserProfileDao,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val _profile = MutableStateFlow(UserProfile())
    val profile: StateFlow<UserProfile> = _profile.asStateFlow()
    private val updateMutex = Mutex()

    /**
     * Keep the initial read as a job so every write can wait for it. Without
     * this barrier an early update can overwrite persisted fields or be
     * overwritten when the database read completes.
     */
    private val loadJob = scope.launch {
        dao.get()?.let { _profile.value = UserProfile.fromEntity(it) }
    }

    suspend fun update(
        name: String? = null,
        traits: List<String>? = null,
        preferences: Map<String, String>? = null,
        facts: List<String>? = null,
    ) {
        loadJob.join()
        updateMutex.withLock {
            val current = _profile.value
            persist(
                UserProfile(
                    name = name ?: current.name,
                    traits = traits ?: current.traits,
                    preferences = current.preferences + (preferences ?: emptyMap()),
                    facts = facts ?: current.facts,
                ),
            )
        }
    }

    /** Merge newly extracted facts without deleting facts learned earlier. */
    suspend fun mergeFacts(newFacts: List<String>) {
        loadJob.join()
        updateMutex.withLock {
            val current = _profile.value
            val mergedFacts = (current.facts + newFacts)
                .map(String::trim)
                .filter(String::isNotBlank)
                .distinctBy(String::lowercase)
            persist(current.copy(facts = mergedFacts))
        }
    }

    suspend fun awaitLoaded() = loadJob.join()

    fun getSystemPrompt(): String = _profile.value.toSystemPrompt()

    private suspend fun persist(value: UserProfile) {
        _profile.value = value
        dao.upsert(UserProfile.toEntity(value))
    }
}
