package com.aura.agentrun

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
object AgentRunModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AgentRunDatabase =
        RoomConfig.builder(
            context,
            AgentRunDatabase::class.java,
            "aura-agent-runs.db",
            migrations = arrayOf(),
        ).build()

    @Provides
    fun provideAgentRunDao(db: AgentRunDatabase): AgentRunDao = db.agentRunDao()

    @Provides
    fun provideGoalDao(db: AgentRunDatabase): GoalDao = db.goalDao()

    @Provides
    fun provideStepDao(db: AgentRunDatabase): StepDao = db.stepDao()

    @Provides
    fun provideAgentEventDao(db: AgentRunDatabase): AgentEventDao = db.agentEventDao()

    @Provides
    fun provideApprovalRequestDao(db: AgentRunDatabase): ApprovalRequestDao = db.approvalRequestDao()

    @Provides
    fun provideRunCheckpointDao(db: AgentRunDatabase): RunCheckpointDao = db.runCheckpointDao()
}