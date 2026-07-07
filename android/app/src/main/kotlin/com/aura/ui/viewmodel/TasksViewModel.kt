package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.WorkManager
import com.aura.tasks.ReminderDao
import com.aura.tasks.ReminderEntity
import com.aura.tasks.TaskDao
import com.aura.tasks.TaskEntity
import com.aura.tools.ReminderWorker
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
    /** Filter: "all", "pending", "done" */
    val statusFilter: String = "all",
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
        _state.value = _state.value.copy(loading = true)
        viewModelScope.launch {
            val tasks = taskDao.all()
            _state.update { it.copy(tasks = tasks, loading = false) }
        }
        viewModelScope.launch {
            reminderDao.observeUpcoming(System.currentTimeMillis()).collectLatest { reminders ->
                _state.update { it.copy(reminders = reminders) }
            }
        }
    }

    fun setStatusFilter(filter: String) {
        _state.update { it.copy(statusFilter = filter) }
        refreshTasks()
    }

    private fun refreshTasks() {
        viewModelScope.launch {
            val all = taskDao.all()
            val filtered = when (_state.value.statusFilter) {
                "pending" -> all.filter { it.status != "done" }
                "done" -> all.filter { it.status == "done" }
                else -> all
            }
            _state.update { it.copy(tasks = filtered) }
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

    /**
     * Reopen a completed task — flip status back to "pending".
     * The "undo" for accidental mark-done taps.
     */
    fun reopenTask(id: String) {
        viewModelScope.launch {
            val task = taskDao.get(id) ?: return@launch
            taskDao.update(task.copy(status = "pending", completedAt = null))
            refreshTasks()
        }
    }

    /**
     * Update a task's editable fields. Used by the edit dialog.
     */
    fun updateTask(id: String, title: String, description: String, dueAt: Long?, priority: Int, tags: String) {
        viewModelScope.launch {
            val task = taskDao.get(id) ?: return@launch
            taskDao.update(task.copy(
                title = title.trim(),
                description = description.trim(),
                dueAt = dueAt,
                priority = priority.coerceIn(0, 3),
                tags = tags,
            ))
            refreshTasks()
        }
    }

    /**
     * Delete all completed tasks. Cleans up the graveyard of done
     * items that accumulate over weeks of use.
     */
    fun clearCompleted() {
        viewModelScope.launch {
            val done = taskDao.all().filter { it.status == "done" }
            for (task in done) {
                taskDao.delete(task.id)
            }
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
            WorkManager.getInstance(getApplication()).cancelWorkById(UUID.fromString(id))
        }
    }

    /**
     * Update an existing reminder's message and/or time. Cancels the
     * old WorkManager job and schedules a new one with the updated
     * values. The reminder ID stays the same — we update the Room row
     * and replace the WorkManager work.
     */
    fun updateReminder(id: String, message: String, triggerAt: Long) {
        viewModelScope.launch {
            // Cancel the old work
            WorkManager.getInstance(getApplication()).cancelWorkById(UUID.fromString(id))
            // Update the Room row
            val existing = reminderDao.get(id) ?: return@launch
            reminderDao.insert(existing.copy(message = message, triggerAt = triggerAt))
            // Schedule new work with the same ID
            val delayMs = (triggerAt - System.currentTimeMillis()).coerceAtLeast(0L)
            val notificationId = nextNotificationId.getAndIncrement()
            val work = androidx.work.OneTimeWorkRequestBuilder<ReminderWorker>()
                .setInitialDelay(delayMs, java.util.concurrent.TimeUnit.MILLISECONDS)
                .setInputData(
                    androidx.work.Data.Builder()
                        .putString("title", "⏰ Reminder")
                        .putString("body", message)
                        .putInt("notificationId", notificationId)
                        .build()
                )
                .addTag("reminder")
                .build()
            androidx.work.WorkManager.getInstance(getApplication())
                .enqueueUniqueWork("reminder-$id", androidx.work.ExistingWorkPolicy.REPLACE, work)
        }
    }

    /**
     * Create a reminder from the UI (AddReminderDialog). Same pattern as
     * SetReminderTool but called directly by the user instead of the agent.
     * Schedules a [ReminderWorker] via WorkManager and persists a
     * [ReminderEntity] so the reminder shows in the upcoming list.
     */
    fun createReminder(message: String, triggerAt: Long) {
        if (message.isBlank()) return
        viewModelScope.launch {
            val delayMs = (triggerAt - System.currentTimeMillis()).coerceAtLeast(0L)
            val notificationId = nextNotificationId.getAndIncrement()
            val work = androidx.work.OneTimeWorkRequestBuilder<ReminderWorker>()
                .setInitialDelay(delayMs, java.util.concurrent.TimeUnit.MILLISECONDS)
                .setInputData(
                    androidx.work.Data.Builder()
                        .putString("title", "⏰ Reminder")
                        .putString("body", message)
                        .putInt("notificationId", notificationId)
                        .build()
                )
                .addTag("reminder")
                .build()
            androidx.work.WorkManager.getInstance(getApplication())
                .enqueueUniqueWork("reminder-${work.id}", androidx.work.ExistingWorkPolicy.REPLACE, work)
            runCatching {
                reminderDao.insert(
                    ReminderEntity(
                        id = work.id.toString(),
                        message = message,
                        triggerAt = triggerAt,
                    )
                )
            }
        }
    }

    companion object {
        private val nextNotificationId = java.util.concurrent.atomic.AtomicInteger(1000)
    }
}
