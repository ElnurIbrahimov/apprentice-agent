package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.profile.UserProfile
import com.aura.profile.UserProfileStore
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ProfileUiState(
    val name: String = "",
    val traits: List<String> = emptyList(),
    val facts: List<String> = emptyList(),
)

@EntryPoint
@InstallIn(SingletonComponent::class)
interface ProfileEntry {
    fun userProfileStore(): UserProfileStore
}

@HiltViewModel
class ProfileViewModel @Inject constructor(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(ProfileUiState())
    val state: StateFlow<ProfileUiState> = _state.asStateFlow()

    private val store = EntryPointAccessors.fromApplication(getApplication(), ProfileEntry::class.java).userProfileStore()

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
        viewModelScope.launch { store.update(traits = listOf(trimmed)) }
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
        viewModelScope.launch { store.update(facts = listOf(trimmed)) }
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
