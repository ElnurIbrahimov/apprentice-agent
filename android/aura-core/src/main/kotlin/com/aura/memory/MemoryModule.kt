package com.aura.memory

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
object MemoryModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): MemoryDatabase =
        Room.databaseBuilder(context, MemoryDatabase::class.java, "aura-memory.db")
            .fallbackToDestructiveMigration()
            .build()

    @Provides
    fun provideMemoryDao(db: MemoryDatabase): MemoryDao = db.memoryDao()

    @Provides
    @Singleton
    fun provideEmbedder(): Embedder = Embedder()

    @Provides
    @Singleton
    fun provideVectorIndex(): VectorIndex = VectorIndex()

    @Provides
    @Singleton
    fun provideWriteGate(): WriteGate = WriteGate()
}
