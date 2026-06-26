package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.hands.Hand
import com.aura.hands.HandDao
import dagger.hilt.EntryPoint; import dagger.hilt.InstallIn; import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.android.lifecycle.HiltViewModel; import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.MutableStateFlow; import kotlinx.coroutines.flow.StateFlow; import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class HandsUiState(val hands: List<Hand> = emptyList(), val loading: Boolean = true)
@EntryPoint @InstallIn(SingletonComponent::class) interface HandsEntry { fun handDao(): HandDao }
@HiltViewModel
class HandsViewModel @Inject constructor(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(HandsUiState())
    val state: StateFlow<HandsUiState> = _state.asStateFlow()
    init { load() }
    fun load() { viewModelScope.launch { val d = EntryPointAccessors.fromApplication(getApplication(), HandsEntry::class.java).handDao(); _state.value = HandsUiState(hands = d.getAll(), loading = false) } }
}
