package com.aura.profile

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
object UserProfileModule {
    @Provides @Singleton
    fun provideDb(@ApplicationContext ctx: Context): UserProfileDatabase =
        Room.databaseBuilder(ctx, UserProfileDatabase::class.java, "aura-profile.db").build()

    @Provides fun provideDao(db: UserProfileDatabase): UserProfileDao = db.userProfileDao()
}
