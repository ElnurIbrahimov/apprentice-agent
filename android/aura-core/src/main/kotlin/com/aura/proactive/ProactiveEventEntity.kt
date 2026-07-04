package com.aura.proactive

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "proactive_events")
data class ProactiveEventEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val eventType: String,
    val title: String,
    val body: String,
    val timestamp: Long,
    val payload: String = "",
)
