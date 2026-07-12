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

    /**
     * Count of events with timestamp > [since]. Used by
     * [com.aura.proactive.ProactiveEvents.unreadCount] to drive the
     * Home-screen "📬 N today" badge. Pure SQL aggregate — the caller
     * doesn't need to load the full event rows, just the integer.
     */
    @Query("SELECT COUNT(*) FROM proactive_events WHERE timestamp > :since")
    suspend fun countSince(since: Long): Int

    /**
     * Delete all events older than [cutoff]. Called from
     * [com.aura.proactive.ProactiveEvents.init] with `now - 30 days`
     * so the table stays bounded — without this it grows forever and
     * the `recent()` query slows down as rows accumulate.
     */
    @Query("DELETE FROM proactive_events WHERE timestamp < :cutoff")
    suspend fun deleteOlderThan(cutoff: Long): Int
}