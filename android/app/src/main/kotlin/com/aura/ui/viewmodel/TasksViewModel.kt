package com.aura.ui.viewmodel

import android.app.Application; import androidx.lifecycle.AndroidViewModel; import androidx.lifecycle.viewModelScope
import com.aura.tasks.TaskDao; import com.aura.tasks.TaskEntity
import dagger.hilt.EntryPoint; import dagger.hilt.InstallIn; import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.android.lifecycle.HiltViewModel; import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.MutableStateFlow; import kotlinx.coroutines.flow.StateFlow; import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch; import javax.inject.Inject; import java.util.UUID

data class TasksUiState(val tasks: List<TaskEntity> = emptyList(), val loading: Boolean = true)

@EntryPoint @InstallIn(SingletonComponent::class) interface TasksEntry { fun taskDao(): TaskDao }

@HiltViewModel class TasksViewModel @Inject constructor(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(TasksUiState()); val state: StateFlow<TasksUiState> = _state.asStateFlow()
    private fun dao() = EntryPointAccessors.fromApplication(getApplication(), TasksEntry::class.java).taskDao()
    init { load() }
    fun load() { viewModelScope.launch { _state.value = TasksUiState(tasks = dao().all(), loading = false) } }
    fun delete(id: String) { viewModelScope.launch { dao().delete(id); load() } }
    fun markDone(id: String) { viewModelScope.launch { dao().markComplete(id, System.currentTimeMillis()); load() } }
    fun add(title: String) {
        if (title.isBlank()) return
        viewModelScope.launch {
            dao().insert(TaskEntity(
                id = UUID.randomUUID().toString(),
                title = title.trim(),
                status = "pending",
                createdAt = System.currentTimeMillis(),
            ))
            load()
        }
    }
}
