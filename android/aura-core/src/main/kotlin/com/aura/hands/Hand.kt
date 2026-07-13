package com.aura.hands

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * A "hand" is a user-defined automation macro — a named sequence of tool calls
 * that can be triggered by name or by a trigger phrase.
 */
@Entity(tableName = "hands")
data class Hand(
    @PrimaryKey val id: String,
    val name: String,
    val triggerPhrase: String = "",
    /** JSON array of {tool:String, args:Map<String,String>} */
    val steps: String = "[]",
    /** JSON object of template-variable defaults. */
    val variables: String = "{}",
    /** JSON array of [HandCondition] records. */
    val conditions: String = "[]",
    /** none, daily, weekdays, or weekly. */
    val scheduleType: String = "none",
    val scheduleHour: Int = 9,
    val scheduleMinute: Int = 0,
    /** java.time.DayOfWeek value (1 = Monday, 7 = Sunday). */
    val scheduleDayOfWeek: Int = 1,
    val enabled: Boolean = true,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = createdAt,
)
