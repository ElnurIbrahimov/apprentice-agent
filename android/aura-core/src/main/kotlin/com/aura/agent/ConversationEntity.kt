package com.aura.agent

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "conversations")
data class ConversationEntity(
    @PrimaryKey val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val systemPrompt: String?,
    val model: String?,
    val metadataJson: String = "{}",
    /** The full turn list serialized as JSON via kotlinx.serialization. */
    val turnsJson: String = "[]",
)
