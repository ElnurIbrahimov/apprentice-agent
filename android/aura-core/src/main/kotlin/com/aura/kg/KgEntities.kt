package com.aura.kg

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "kg_nodes",
    indices = [
        Index("label"),
        Index("type"),
        Index("label", "type", unique = true),
    ]
)
data class NodeEntity(
    @PrimaryKey val id: String,
    val label: String,
    val type: String,
    @ColumnInfo(name = "properties") val properties: String = "{}",
    @ColumnInfo(name = "confidence") val confidence: Float = 0.8f,
    @ColumnInfo(name = "sourceTurnId") val sourceTurnId: String = "",
    @ColumnInfo(name = "createdAt") val createdAt: Long = System.currentTimeMillis(),
    @ColumnInfo(name = "updatedAt") val updatedAt: Long = System.currentTimeMillis(),
    @ColumnInfo(name = "accessCount") val accessCount: Int = 0,
    @ColumnInfo(name = "lastAccessed") val lastAccessed: Long = System.currentTimeMillis(),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is NodeEntity) return false
        return id == other.id
    }
    override fun hashCode(): Int = id.hashCode()
}

@Entity(
    tableName = "kg_edges",
    foreignKeys = [
        ForeignKey(entity = NodeEntity::class, parentColumns = ["id"], childColumns = ["sourceId"], onDelete = ForeignKey.CASCADE),
        ForeignKey(entity = NodeEntity::class, parentColumns = ["id"], childColumns = ["targetId"], onDelete = ForeignKey.CASCADE),
    ],
    indices = [
        Index("sourceId"),
        Index("targetId"),
        Index("sourceId", "targetId", "type", unique = true),
    ]
)
data class EdgeEntity(
    @PrimaryKey val id: String,
    val type: String,
    @ColumnInfo(name = "sourceId") val sourceId: String,
    @ColumnInfo(name = "targetId") val targetId: String,
    @ColumnInfo(name = "weight") val weight: Float = 0.5f,
    @ColumnInfo(name = "properties") val properties: String = "{}",
    @ColumnInfo(name = "confidence") val confidence: Float = 0.8f,
    @ColumnInfo(name = "sourceTurnId") val sourceTurnId: String = "",
    @ColumnInfo(name = "createdAt") val createdAt: Long = System.currentTimeMillis(),
    @ColumnInfo(name = "lastReinforced") val lastReinforced: Long = System.currentTimeMillis(),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is EdgeEntity) return false
        return id == other.id
    }
    override fun hashCode(): Int = id.hashCode()
}
