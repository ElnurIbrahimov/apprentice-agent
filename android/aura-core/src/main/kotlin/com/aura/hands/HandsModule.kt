package com.aura.hands

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
object HandsModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): HandDatabase =
        Room.databaseBuilder(context, HandDatabase::class.java, "aura-hands.db")
            .fallbackToDestructiveMigration()
            .build()

    @Provides
    fun provideHandDao(db: HandDatabase): HandDao = db.handDao()
}
