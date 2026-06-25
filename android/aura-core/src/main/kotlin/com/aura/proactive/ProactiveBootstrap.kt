package com.aura.proactive

import android.app.Application
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject

/**
 * Auto-starts the proactive layer when the app process starts. The
 * AuraApp Application class should call bootstrap() in onCreate.
 */
class ProactiveBootstrap @Inject constructor(
    private val scheduler: ProactiveScheduler,
    private val calendarMonitor: CalendarMonitor,
) {
    fun start() {
        scheduler.scheduleMorningBrief()
        calendarMonitor.start()
    }
}
