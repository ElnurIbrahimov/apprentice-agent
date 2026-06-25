package com.aura.di

import com.aura.agent.ToolExecutor
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * App-level Hilt providers. ToolRegistry is provided by [com.aura.tools.ToolsModule]
 * in the aura-core module — don't redefine it here.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideToolExecutor(registry: com.aura.agent.ToolRegistry): ToolExecutor =
        ToolExecutor(registry)
}
