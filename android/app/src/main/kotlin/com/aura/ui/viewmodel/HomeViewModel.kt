package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.memory.MemoryEntity
import com.aura.memory.MemoryStore
import com.aura.tasks.TaskDao
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class HomeUiState(
    val today: List<String> = emptyList(),  // calendar events (formatted)
    val recentMemories: List<MemoryEntity> = emptyList(),
    val pendingTasks: List<String> = emptyList(),
    val userName: String? = null,
    val hour: Int = 0,
    val loading: Boolean = true,
)

@HiltViewModel
class HomeViewModel @Inject constructor(
    application: Application,
    private val memoryStore: MemoryStore,
    private val taskDao: TaskDao,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(HomeUiState())
    val state: StateFlow<HomeUiState> = _state.asStateFlow()

    init { refresh() }

    fun refresh() {
        viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            val cal = java.util.Calendar.getInstance()
            val hour = cal.get(java.util.Calendar.HOUR_OF_DAY)
            // Recent memories
            val recent = memoryStore.recent(5)
            // Try to find the user's name in a memory (preference: "i am X" or "my name is X")
            val name = memoryStore.recent(50).mapNotNull { mem ->
                val lower = mem.content.lowercase()
                when {
                    lower.contains("my name is ") -> mem.content.substringAfter("my name is ", "").trim().take(40)
                    lower.startsWith("i am ") && mem.category == "person" -> mem.content.removePrefix("i am ").removePrefix("I am ").take(40)
                    lower.startsWith("call me ") -> mem.content.removePrefix("call me ").removePrefix("Call me ").take(40)
                    else -> null
                }
            }.firstOrNull()?.takeIf { it.isNotBlank() && it.length < 40 }
            // Tasks
            val tasks = taskDao.allPending().take(5).map { it.title }
            // Calendar — best effort, ignore exceptions. Tool is invoked via
            // dependency injection in a future day; for now leave the list empty.
            val events: List<String> = emptyList()
            _state.update {
                it.copy(
                    today = events,
                    recentMemories = recent,
                    pendingTasks = tasks,
                    userName = name,
                    hour = hour,
                    loading = false,
                )
            }
        }
    }
}
