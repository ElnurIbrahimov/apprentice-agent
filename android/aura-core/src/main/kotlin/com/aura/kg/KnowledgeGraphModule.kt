package com.aura.kg

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object KnowledgeGraphModule {

    @Provides
    @Singleton
    fun provideKnowledgeGraphDao(db: com.aura.memory.MemoryDatabase): KnowledgeGraphDao =
        db.knowledgeGraphDao()

    @Provides
    @Singleton
    fun provideKnowledgeGraphRepository(dao: KnowledgeGraphDao): KnowledgeGraphRepository =
        KnowledgeGraphRepository(dao)
}
