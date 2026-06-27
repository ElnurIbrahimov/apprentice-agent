package com.aura.hands

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
object HandsModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): HandDatabase =
        RoomConfig.builder(
            context,
            HandDatabase::class.java,
            "aura-hands.db",
            migrations = emptyArray(),
        ).build()

    @Provides
    fun provideHandDao(db: HandDatabase): HandDao = db.handDao()
}
