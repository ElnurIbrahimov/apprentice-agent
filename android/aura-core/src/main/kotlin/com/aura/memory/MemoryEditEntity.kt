package com.aura.memory

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Audit trail entry for a memory edit. Created when [MemoryStore.update]
 * modifies a memory's content, category, or importance. Tracks the old
 * content so the user can see what changed and when.
 *
 * Foreign key to [MemoryEntity] with CASCADE delete — when a memory is
 * deleted, its edit history goes too.
 */
@Entity(
    tableName = "memory_edits",
    foreignKeys = [
        ForeignKey(
            entity = MemoryEntity::class,
            parentColumns = ["id"],
            childColumns = ["memoryId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("memoryId")],
)
data class MemoryEditEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "memoryId") val memoryId: String,
    @ColumnInfo(name = "oldContent") val oldContent: String,
    @ColumnInfo(name = "newContent") val newContent: String,
    @ColumnInfo(name = "oldCategory") val oldCategory: String,
    @ColumnInfo(name = "newCategory") val newCategory: String,
    @ColumnInfo(name = "editedAt") val editedAt: Long = System.currentTimeMillis(),
    /** "user" when edited via the UI, "agent" when edited by the agent loop. */
    @ColumnInfo(name = "editedBy") val editedBy: String = "user",
)