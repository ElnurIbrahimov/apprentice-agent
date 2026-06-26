package com.aura.ui.viewmodel

import android.app.Application; import androidx.lifecycle.AndroidViewModel; import androidx.lifecycle.viewModelScope
import com.aura.tasks.TaskDao; import com.aura.tasks.TaskEntity
import dagger.hilt.EntryPoint; import dagger.hilt.InstallIn; import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.android.lifecycle.HiltViewModel; import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.MutableStateFlow; import kotlinx.coroutines.flow.StateFlow; import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch; import javax.inject.Inject

data class TasksUiState(val tasks: List<TaskEntity> = emptyList(), val loading: Boolean = true)
@EntryPoint @InstallIn(SingletonComponent::class) interface TasksEntry { fun taskDao(): TaskDao }
@HiltViewModel class TasksViewModel @Inject constructor(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(TasksUiState()); val state: StateFlow<TasksUiState> = _state.asStateFlow()
    init { load() }
    fun load() { viewModelScope.launch { val d = EntryPointAccessors.fromApplication(getApplication(), TasksEntry::class.java).taskDao(); _state.value = TasksUiState(tasks = d.all(), loading = false) } }
    fun delete(id: String) { viewModelScope.launch { EntryPointAccessors.fromApplication(getApplication(), TasksEntry::class.java).taskDao().delete(id); load() } }
}
