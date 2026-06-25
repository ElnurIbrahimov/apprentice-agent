package com.aura.tools

import com.aura.agent.ToolRegistry
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object ToolsModule {

    @Provides
    @Singleton
    fun provideToolRegistry(
        webSearch: WebSearchTool,
        notifications: NotificationsTool,
        location: LocationNowTool,
        share: ShareIntentTool,
        calendar: CalendarReadTool,
        setReminder: SetReminderTool,
        getCurrentTime: GetCurrentTimeTool,
        remember: RememberTool,
        recall: RecallTool,
    ): ToolRegistry {
        val registry = ToolRegistry()
        registry.register(webSearch.tool)
        registry.register(notifications.tool)
        registry.register(location.tool)
        registry.register(share.tool)
        registry.register(calendar.tool)
        registry.register(setReminder.tool)
        registry.register(getCurrentTime.tool)
        registry.register(remember.tool)
        registry.register(recall.tool)
        return registry
    }
}
