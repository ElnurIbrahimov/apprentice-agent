package com.aura.creative

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "creative_projects",
    indices = [Index(value = ["updatedAt"]), Index(value = ["name"])],
)
data class CreativeProjectEntity(
    @PrimaryKey val id: String,
    val name: String,
    val description: String = "",
    val genre: String = "",
    val tone: String = "",
    val worldJson: String = "{}",
    val templateId: String = "",
    val metadataJson: String = "{}",
    val turnCount: Int = 0,
    val lastSessionEnded: Long = 0L,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
)

data class CreativeProject(
    val id: String,
    val name: String,
    val description: String,
    val genre: String,
    val tone: String,
    val world: WorldBible,
    val templateId: String,
    val turnCount: Int,
    val createdAt: Long,
    val updatedAt: Long,
)