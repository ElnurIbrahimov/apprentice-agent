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
        braveSearch: BraveSearchTool,
        tavilySearch: TavilySearchTool,
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
        imageInput: ImageInputTool,
        filePicker: FilePickerTool,
        firecrawlFetch: FirecrawlFetchTool,
        deepResearch: DeepResearchTool,
        vision: VisionTool,
        imageGen: ImageGenTool,
        transcription: TranscriptionTool,
    ): ToolRegistry {
        val registry = ToolRegistry()
        registry.register(webSearch.tool)
        registry.register(braveSearch.tool)
        registry.register(tavilySearch.tool)
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
        registry.register(imageInput.tool)
        registry.register(filePicker.tool)
        registry.register(firecrawlFetch.tool)
        registry.register(deepResearch.tool)
        registry.register(vision.tool)
        registry.register(imageGen.tool)
        registry.register(transcription.tool)
        return registry
    }
}
