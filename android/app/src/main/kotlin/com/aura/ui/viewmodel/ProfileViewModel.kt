package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.profile.UserProfile
import com.aura.profile.UserProfileStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ProfileUiState(
    val name: String = "",
    val traits: List<String> = emptyList(),
    val facts: List<String> = emptyList(),
)

@HiltViewModel
class ProfileViewModel @Inject constructor(
    app: Application,
    private val store: UserProfileStore,
) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(ProfileUiState())
    val state: StateFlow<ProfileUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            store.profile.collect { profile ->
                _state.value = profile.toUiState()
            }
        }
    }

    fun setName(name: String) {
        viewModelScope.launch { store.update(name = name.trim().takeIf { it.isNotBlank() }) }
    }

    fun addTrait(trait: String) {
        val trimmed = trait.trim()
        if (trimmed.isBlank()) return
        viewModelScope.launch {
            val current = store.profile.value
            if (trimmed !in current.traits) {
                store.update(traits = current.traits + trimmed)
            }
        }
    }

    fun removeTrait(trait: String) {
        viewModelScope.launch {
            val current = store.profile.value
            store.update(traits = current.traits.filter { it != trait })
        }
    }

    fun addFact(fact: String) {
        val trimmed = fact.trim()
        if (trimmed.isBlank()) return
        viewModelScope.launch {
            val current = store.profile.value
            if (trimmed !in current.facts) {
                store.update(facts = current.facts + trimmed)
            }
        }
    }

    fun removeFact(fact: String) {
        viewModelScope.launch {
            val current = store.profile.value
            store.update(facts = current.facts.filter { it != fact })
        }
    }

    fun clear() {
        viewModelScope.launch {
            store.update(
                name = "",
                traits = emptyList(),
                facts = emptyList(),
            )
        }
    }
}

private fun UserProfile.toUiState(): ProfileUiState = ProfileUiState(
    name = name ?: "",
    traits = traits,
    facts = facts,
)
