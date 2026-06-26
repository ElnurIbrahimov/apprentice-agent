package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.agent.Conversation
import com.aura.agent.ConversationStore
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class HistoryUiState(
    val conversations: List<Conversation> = emptyList(),
    val loading: Boolean = true,
)

@EntryPoint @InstallIn(SingletonComponent::class)
interface HistoryConvEntry { fun conversationStore(): ConversationStore }

@HiltViewModel
class HistoryViewModel @Inject constructor(
    application: Application,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(HistoryUiState())
    val state: StateFlow<HistoryUiState> = _state.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            val store = EntryPointAccessors.fromApplication(
                getApplication(), HistoryConvEntry::class.java,
            ).conversationStore()
            _state.value = HistoryUiState(
                conversations = store.recent(50),
                loading = false,
            )
        }
    }

    fun delete(id: String) {
        viewModelScope.launch {
            val store = EntryPointAccessors.fromApplication(
                getApplication(), HistoryConvEntry::class.java,
            ).conversationStore()
            store.delete(id)
            load()
        }
    }
}
