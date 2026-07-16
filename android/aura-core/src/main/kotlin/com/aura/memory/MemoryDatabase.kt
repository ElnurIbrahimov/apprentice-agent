package com.aura.memory

import androidx.room.Database
import androidx.room.RoomDatabase
import com.aura.creative.CreativeProjectDao
import com.aura.creative.CreativeProjectEntity
import com.aura.documents.DocumentDao
import com.aura.documents.DocumentEntity
import com.aura.documents.DocumentChunkDao
import com.aura.documents.DocumentChunkEntity
import com.aura.kg.EdgeEntity
import com.aura.kg.KnowledgeGraphDao
import com.aura.kg.NodeEntity

@Database(
    entities = [
        MemoryEntity::class,
        NodeEntity::class,
        EdgeEntity::class,
        MemoryEditEntity::class,
        DocumentEntity::class,
        CreativeProjectEntity::class,
        DocumentChunkEntity::class,
    ],
    version = 7,
    exportSchema = true,
)
abstract class MemoryDatabase : RoomDatabase() {
    abstract fun memoryDao(): MemoryDao
    abstract fun knowledgeGraphDao(): KnowledgeGraphDao
    abstract fun memoryEditDao(): MemoryEditDao
    abstract fun documentDao(): DocumentDao
    abstract fun creativeProjectDao(): CreativeProjectDao
    abstract fun documentChunkDao(): DocumentChunkDao
}
