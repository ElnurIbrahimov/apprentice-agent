package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.tasks.ReminderEntity
import com.aura.tasks.ReminderStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Owns the user-facing "my reminders" list. The list is a
 * [StateFlow] derived from [ReminderStore.observeUpcoming], so
 * insert / delete / cancel from any caller redraws the screen.
 */
@HiltViewModel
class RemindersViewModel @Inject constructor(
    private val reminderStore: ReminderStore,
) : ViewModel() {
    val reminders: StateFlow<List<ReminderEntity>> =
        reminderStore.observeUpcoming()
            .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    fun cancel(id: String) {
        viewModelScope.launch { reminderStore.cancel(id) }
    }
}
