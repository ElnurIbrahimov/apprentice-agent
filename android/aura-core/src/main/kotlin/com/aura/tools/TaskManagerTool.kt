package com.aura.tools

import android.content.Context
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import com.aura.tasks.TaskEntity
import com.aura.tasks.TaskDao
import dagger.hilt.android.qualifiers.ApplicationContext
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manage tasks: create one (optionally with a reminder), list, complete, or delete.
 * Mirrors aura/tools/task_manager.py + aura/tools/task_scheduler.py.
 * Risk: WRITE_LOCAL.
 */
@Singleton
class TaskManagerTool @Inject constructor(
    @ApplicationContext private val context: Context,
    private val taskDao: TaskDao,
) {
    fun definition() = ToolDefinition(
        name = "manage_tasks",
        description = "Create, list, complete, or delete tasks. Action is 'create'|'list'|'complete'|'delete'. For 'create', 'title' is required, 'when' is optional HH:mm for a reminder.",
        parameters = ToolParameters(
            properties = mapOf(
                "action" to ToolProperty(type = "string", description = "create|list|complete|delete"),
                "title" to ToolProperty(type = "string", description = "Task title (for create)"),
                "id" to ToolProperty(type = "string", description = "Task id (for complete/delete)"),
                "when" to ToolProperty(type = "string", description = "Optional reminder time HH:mm"),
            ),
            required = listOf("action"),
        ),
    )

    val tool = Tool(
        name = "manage_tasks",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            when (val action = (call.arguments["action"] as? String)?.lowercase()) {
                "create" -> {
                    val title = call.arguments["title"] as? String ?: return@Tool ToolResult.Error("missing 'title'", "bad_args")
                    val whenStr = call.arguments["when"] as? String
                    try {
                        val id = java.util.UUID.randomUUID().toString()
                        val triggerAt = whenStr?.let { parseTime(it) }
                        taskDao.insert(
                            TaskEntity(id = id, title = title, createdAt = System.currentTimeMillis(), dueAt = triggerAt, status = "pending")
                        )
                        if (triggerAt != null && triggerAt > System.currentTimeMillis()) {
                            val work = OneTimeWorkRequestBuilder<ReminderWorker>()
                                .setInitialDelay(triggerAt - System.currentTimeMillis(), TimeUnit.MILLISECONDS)
                                .setInputData(
                                    Data.Builder()
                                        .putString("title", "📋 $title")
                                        .putString("body", "Task reminder")
                                        .putString("task_id", id)
                                        .build()
                                )
                                .setConstraints(Constraints.NONE)
                                .addTag("task-$id")
                                .build()
                            WorkManager.getInstance(context).enqueueUniqueWork("task-$id", ExistingWorkPolicy.REPLACE, work)
                        }
                        val whenDisplay = triggerAt?.let { SimpleDateFormat("EEE MMM d, HH:mm", Locale.US).format(it) } ?: "(no reminder)"
                        ToolResult.Ok("Task created (id $id): $title — $whenDisplay")
                    } catch (e: Exception) {
                        ToolResult.Error("create failed: ${e.message}", "exception")
                    }
                }
                "list" -> {
                    try {
                        val all = taskDao.all()
                        if (all.isEmpty()) return@Tool ToolResult.Ok("No tasks.")
                        val text = all.take(20).mapIndexed { i, t ->
                            val due = t.dueAt?.let { SimpleDateFormat("MMM d HH:mm", Locale.US).format(it) } ?: "—"
                            "${i + 1}. [${t.status}] $t.title (due: $due)"
                        }.joinToString("\n")
                        ToolResult.Ok(text)
                    } catch (e: Exception) {
                        ToolResult.Error("list failed: ${e.message}", "exception")
                    }
                }
                "complete" -> {
                    val id = call.arguments["id"] as? String ?: return@Tool ToolResult.Error("missing 'id'", "bad_args")
                    try {
                        taskDao.markComplete(id, System.currentTimeMillis())
                        ToolResult.Ok("Task $id marked complete.")
                    } catch (e: Exception) {
                        ToolResult.Error("complete failed: ${e.message}", "exception")
                    }
                }
                "delete" -> {
                    val id = call.arguments["id"] as? String ?: return@Tool ToolResult.Error("missing 'id'", "bad_args")
                    try {
                        taskDao.delete(id)
                        WorkManager.getInstance(context).cancelUniqueWork("task-$id")
                        ToolResult.Ok("Task $id deleted.")
                    } catch (e: Exception) {
                        ToolResult.Error("delete failed: ${e.message}", "exception")
                    }
                }
                else -> ToolResult.Error("unknown action: $action", "bad_args")
            }
        },
    )

    private fun parseTime(s: String): Long? {
        return try {
            val parts = s.split(":")
            if (parts.size != 2) return null
            val cal = java.util.Calendar.getInstance()
            cal.set(java.util.Calendar.HOUR_OF_DAY, parts[0].toInt())
            cal.set(java.util.Calendar.MINUTE, parts[1].toInt())
            cal.set(java.util.Calendar.SECOND, 0)
            cal.set(java.util.Calendar.MILLISECOND, 0)
            if (cal.timeInMillis <= System.currentTimeMillis()) {
                cal.add(java.util.Calendar.DAY_OF_YEAR, 1)
            }
            cal.timeInMillis
        } catch (_: Exception) { null }
    }
}
