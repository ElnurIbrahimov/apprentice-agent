package com.aura.di

import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

/**
 * App-level Hilt providers. ToolRegistry and ToolExecutor are provided by
 * [com.aura.tools.ToolsModule] in the aura-core module — both are @Inject
 * constructors and Hilt resolves them automatically. This file is kept as
 * a marker so future app-scoped providers have an obvious home.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule
