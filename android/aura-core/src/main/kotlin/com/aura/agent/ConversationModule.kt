package com.aura.agent

import android.content.Context
import androidx.room.Room
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object ConversationModule {
    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): ConversationDatabase =
        Room.databaseBuilder(context, ConversationDatabase::class.java, "aura-conversations.db")
            .build()

    @Provides
    fun provideConversationDao(db: ConversationDatabase): ConversationDao = db.conversationDao()
}
