package com.aura.agent

import android.content.Context
import com.aura.data.RoomConfig
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
        RoomConfig.builder(
            context,
            ConversationDatabase::class.java,
            "aura-conversations.db",
            migrations = emptyArray(),
        ).build()

    @Provides
    fun provideConversationDao(db: ConversationDatabase): ConversationDao = db.conversationDao()
}
