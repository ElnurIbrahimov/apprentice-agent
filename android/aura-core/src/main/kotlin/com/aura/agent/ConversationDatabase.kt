package com.aura.agent

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(entities = [ConversationEntity::class], version = 2, exportSchema = true)
abstract class ConversationDatabase : RoomDatabase() {
    abstract fun conversationDao(): ConversationDao
}
