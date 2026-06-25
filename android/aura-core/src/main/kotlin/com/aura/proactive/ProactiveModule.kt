package com.aura.proactive

import android.content.Context
import androidx.work.WorkerFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object ProactiveModule {

    @Provides
    @Singleton
    fun provideCalendarMonitor(
        @ApplicationContext context: Context,
        eventBus: ProactiveEventBus,
    ): CalendarMonitor = CalendarMonitor(context, eventBus)
}
