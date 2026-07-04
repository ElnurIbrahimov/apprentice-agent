package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.WorkManager
import com.aura.tasks.ReminderDao
import com.aura.tasks.ReminderEntity
import com.aura.tasks.TaskDao
import com.aura.tasks.TaskEntity
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

data class TasksUiState(
    val tasks: List<TaskEntity> = emptyList(),
    val reminders: List<ReminderEntity> = emptyList(),
    val loading: Boolean = true,
)

@HiltViewModel
class TasksViewModel @Inject constructor(
    app: Application,
    private val taskDao: TaskDao,
    private val reminderDao: ReminderDao,
) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(TasksUiState())
    val state: StateFlow<TasksUiState> = _state.asStateFlow()

    init { load() }

    fun load() {
        _state.value = TasksUiState(loading = true)
        viewModelScope.launch {
            val tasks = taskDao.all()
            _state.value = TasksUiState(tasks = tasks, loading = false)
        }
        viewModelScope.launch {
            reminderDao.observeUpcoming(System.currentTimeMillis()).collectLatest { reminders ->
                _state.update { it.copy(reminders = reminders) }
            }
        }
    }

    fun deleteTask(id: String) {
        viewModelScope.launch {
            taskDao.delete(id)
            // Any linked reminder is now obsolete.
            reminderDao.deleteByTaskId(id)
            WorkManager.getInstance(getApplication()).cancelUniqueWork("task-$id")
            refreshTasks()
        }
    }

    fun markDone(id: String) {
        viewModelScope.launch {
            taskDao.markComplete(id, System.currentTimeMillis())
            reminderDao.deleteByTaskId(id)
            WorkManager.getInstance(getApplication()).cancelUniqueWork("task-$id")
            refreshTasks()
        }
    }

    fun addTask(
        title: String,
        description: String = "",
        dueAt: Long? = null,
        priority: Int = 0,
        tags: String = "",
    ) {
        if (title.isBlank()) return
        viewModelScope.launch {
            val id = UUID.randomUUID().toString()
            taskDao.insert(
                TaskEntity(
                    id = id,
                    title = title.trim(),
                    description = description.trim(),
                    createdAt = System.currentTimeMillis(),
                    dueAt = dueAt,
                    status = "pending",
                    priority = priority.coerceIn(0, 3),
                    tags = tags,
                )
            )
            refreshTasks()
        }
    }

    fun cancelReminder(id: String) {
        viewModelScope.launch {
            reminderDao.delete(id)
            // Cancel by the work request id that the row stores.
            WorkManager.getInstance(getApplication()).cancelWorkById(UUID.fromString(id))
        }
    }

    private fun refreshTasks() {
        viewModelScope.launch {
            _state.update { it.copy(tasks = taskDao.all()) }
        }
    }
}
