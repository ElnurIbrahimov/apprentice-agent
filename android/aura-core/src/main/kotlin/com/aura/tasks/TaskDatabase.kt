package com.aura.tasks

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(entities = [TaskEntity::class, ReminderEntity::class], version = 2, exportSchema = false)
abstract class TaskDatabase : RoomDatabase() {
    abstract fun taskDao(): TaskDao
    abstract fun reminderDao(): ReminderDao
}
