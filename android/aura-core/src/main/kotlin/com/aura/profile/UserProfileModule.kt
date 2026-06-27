package com.aura.profile

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
object UserProfileModule {
    @Provides @Singleton
    fun provideDb(@ApplicationContext ctx: Context): UserProfileDatabase =
        RoomConfig.builder(
            ctx,
            UserProfileDatabase::class.java,
            "aura-profile.db",
            migrations = emptyArray(),
        ).build()

    @Provides fun provideDao(db: UserProfileDatabase): UserProfileDao = db.userProfileDao()
}
