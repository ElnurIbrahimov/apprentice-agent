package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.skills.Skill
import com.aura.skills.SkillsStore
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * State holder for the Skills UI. Owns three intents: add, update, remove.
 * Reads the live [SkillsStore.skills] StateFlow; mutations go through the
 * store (which persists to DataStore).
 */
@HiltViewModel
class SkillsViewModel @Inject constructor(
    private val skillsStore: SkillsStore,
) : ViewModel() {

    private val _selectedId = MutableStateFlow<String?>(null)
    val selectedId: StateFlow<String?> = _selectedId.asStateFlow()

    val skills: StateFlow<List<Skill>> = skillsStore.skills

    init {
        viewModelScope.launch { skillsStore.awaitLoaded() }
    }

    fun select(id: String?) { _selectedId.value = id }

    fun add(name: String, description: String, body: String) {
        val safe = name.trim()
        if (safe.isEmpty()) return
        viewModelScope.launch {
            skillsStore.add(Skill(name = safe, description = description.trim(), body = body))
        }
    }

    fun update(skill: Skill) {
        viewModelScope.launch { skillsStore.update(skill) }
    }

    fun remove(id: String) {
        viewModelScope.launch {
            skillsStore.remove(id)
            if (_selectedId.value == id) _selectedId.value = null
        }
    }
}
