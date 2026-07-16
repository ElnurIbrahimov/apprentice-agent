package com.aura.agentrun

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(
    entities = [
        AgentRunEntity::class,
        GoalEntity::class,
        StepEntity::class,
        AgentEventEntity::class,
        ApprovalRequestEntity::class,
        RunCheckpointEntity::class,
    ],
    version = 1,
    exportSchema = true,
)
abstract class AgentRunDatabase : RoomDatabase() {
    abstract fun agentRunDao(): AgentRunDao
    abstract fun goalDao(): GoalDao
    abstract fun stepDao(): StepDao
    abstract fun agentEventDao(): AgentEventDao
    abstract fun approvalRequestDao(): ApprovalRequestDao
    abstract fun runCheckpointDao(): RunCheckpointDao
}