package com.aura.skills

import com.aura.security.SecureDataStore
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

private const val KEY_SKILLS = "aura_skills_v1"

/**
 * Persists user-authored skills in DataStore as a single JSON envelope.
 *
 * The store is intentionally minimal: there is no per-skill on-disk file
 * (DataStore isn't a good fit for many small files) and no Room table
 * (skills are configuration, not state). The envelope is loaded lazily
 * on first access; if the JSON is malformed, the store treats the install
 * as having zero skills and overwrites the bad blob on the next write.
 */
@Singleton
class SkillsStore @Inject constructor(
    private val secureDataStore: SecureDataStore,
) {
    private val _skills = MutableStateFlow<List<Skill>>(emptyList())
    val skills: StateFlow<List<Skill>> = _skills.asStateFlow()

    @Volatile private var loaded: Boolean = false
    private val lock = Any()

    suspend fun awaitLoaded() {
        if (loaded) return
        // Read the persisted blob outside the critical section (getString is suspend);
        // only mutate state under the lock so concurrent callers don't double-init.
        val raw: String? = secureDataStore.getString(KEY_SKILLS)
        synchronized(lock) {
            if (loaded) return
            _skills.value = (raw ?: "").decodeAsSkillList()
            loaded = true
        }
    }

    suspend fun add(skill: Skill) {
        awaitLoaded()
        val updated = _skills.value + skill
        _skills.value = updated
        secureDataStore.putString(KEY_SKILLS, updated.encodeToJsonString())
    }

    suspend fun update(skill: Skill) {
        awaitLoaded()
        val updated = _skills.value.map { if (it.id == skill.id) skill else it }
        _skills.value = updated
        secureDataStore.putString(KEY_SKILLS, updated.encodeToJsonString())
    }

    suspend fun remove(id: String) {
        awaitLoaded()
        val updated = _skills.value.filterNot { it.id == id }
        _skills.value = updated
        secureDataStore.putString(KEY_SKILLS, updated.encodeToJsonString())
    }

    fun findById(id: String): Skill? = _skills.value.firstOrNull { it.id == id }

    fun findByName(name: String): Skill? = _skills.value.firstOrNull { it.name.equals(name, ignoreCase = true) }
}
