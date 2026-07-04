package com.aura.proactive

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(entities = [ProactiveEventEntity::class], version = 2, exportSchema = false)
abstract class ProactiveEventDatabase : RoomDatabase() {
    abstract fun proactiveEventDao(): ProactiveEventDao
}
