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
    val enabled: Boolean = true,
    val createdAt: Long = System.currentTimeMillis(),
)
