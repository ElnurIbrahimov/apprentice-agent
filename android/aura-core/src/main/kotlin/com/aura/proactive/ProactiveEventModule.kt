package com.aura.proactive

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
object ProactiveEventModule {
    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): ProactiveEventDatabase =
        RoomConfig.builder(
            context,
            ProactiveEventDatabase::class.java,
            "aura-proactive.db",
            migrations = emptyArray(),
        ).build()

    @Provides
    fun provideProactiveEventDao(db: ProactiveEventDatabase): ProactiveEventDao = db.proactiveEventDao()
}
