package com.aura.tasks

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import kotlinx.serialization.Serializable

@Entity(
    tableName = "tasks",
    indices = [
        Index(value = ["status"]),
        Index(value = ["status", "dueAt"]),
    ],
)
@Serializable
data class TaskEntity(
    @PrimaryKey val id: String,
    val title: String,
    val description: String = "",
    val createdAt: Long,
    val dueAt: Long? = null,
    val completedAt: Long? = null,
    val status: String = "pending", // pending | done | cancelled
    val priority: Int = 0, // 0..3
    val tags: String = "",
)
