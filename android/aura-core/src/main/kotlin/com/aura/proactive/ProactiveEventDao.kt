package com.aura.proactive

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

@Dao
interface ProactiveEventDao {
    @Insert
    suspend fun insert(event: ProactiveEventEntity): Long

    @Query("SELECT * FROM proactive_events ORDER BY timestamp DESC LIMIT :limit")
    suspend fun recent(limit: Int = 100): List<ProactiveEventEntity>
}
