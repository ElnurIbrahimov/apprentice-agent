package com.aura.memory

import android.content.Context
import androidx.room.Room
import com.aura.providers.ProviderKeys
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
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
    fun provideLocalEmbedder(): LocalEmbedder = LocalEmbedder()

    @Provides
    @Singleton
    fun provideEmbedder(
        localEmbedder: LocalEmbedder,
        providerKeys: ProviderKeys,
        httpClient: OkHttpClient,
    ): Embedder = CloudEmbedder(localEmbedder, providerKeys, httpClient)

    @Provides
    @Singleton
    fun provideVectorIndex(): VectorIndex = VectorIndex()

    @Provides
    @Singleton
    fun provideWriteGate(): WriteGate = WriteGate()
}
