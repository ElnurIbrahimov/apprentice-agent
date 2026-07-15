package com.aura.skills

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

/**
 * Hilt module for the skills subsystem. SkillsStore is constructor-injected
 * with @Singleton so no explicit @Provides is needed, but the module is
 * kept around as the future home for cross-cutting skills concerns
 * (e.g. importer, sharing).
 */
@Module
@InstallIn(SingletonComponent::class)
object SkillsModule
