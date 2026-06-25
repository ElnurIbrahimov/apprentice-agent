package com.aura.di

import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.providers.OllamaCloudProvider
import com.aura.providers.Provider
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoMap
import dagger.multibindings.StringKey
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideToolRegistry(): ToolRegistry = ToolRegistry()

    @Provides
    @Singleton
    fun provideToolExecutor(registry: ToolRegistry): ToolExecutor = ToolExecutor(registry)
}
