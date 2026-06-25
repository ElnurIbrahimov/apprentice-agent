package com.aura.tools

import com.aura.agent.ToolRegistry
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt module that wires every Tool into the singleton ToolRegistry.
 * Mirrors aura/toolsets.py — tools are grouped semantically but here
 * we register them all flat. Future: per-platform enable/disable.
 */
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
        calendarRead: CalendarReadTool,
        calendarWrite: CalendarWriteTool,
        contactsSearch: ContactsSearchTool,
        setReminder: SetReminderTool,
        getCurrentTime: GetCurrentTimeTool,
        remember: RememberTool,
        recall: RecallTool,
        appLauncher: AppLauncherTool,
        systemVolume: SystemVolumeTool,
        photoLibrary: PhotoLibraryTool,
        biometricPrompt: BiometricPromptTool,
        cameraCapture: CameraCaptureTool,
        batteryState: BatteryStateTool,
        networkState: NetworkStateTool,
        dndMode: DndModeTool,
        taskManager: TaskManagerTool,
        notificationList: NotificationListTool,
    ): ToolRegistry {
        val registry = ToolRegistry()
        registry.register(webSearch.tool)
        registry.register(notifications.tool)
        registry.register(location.tool)
        registry.register(share.tool)
        registry.register(calendarRead.tool)
        registry.register(calendarWrite.tool)
        registry.register(contactsSearch.tool)
        registry.register(setReminder.tool)
        registry.register(getCurrentTime.tool)
        registry.register(remember.tool)
        registry.register(recall.tool)
        registry.register(appLauncher.tool)
        registry.register(systemVolume.tool)
        registry.register(photoLibrary.tool)
        registry.register(biometricPrompt.tool)
        registry.register(cameraCapture.tool)
        registry.register(batteryState.tool)
        registry.register(networkState.tool)
        registry.register(dndMode.tool)
        registry.register(taskManager.tool)
        registry.register(notificationList.tool)
        return registry
    }
}
