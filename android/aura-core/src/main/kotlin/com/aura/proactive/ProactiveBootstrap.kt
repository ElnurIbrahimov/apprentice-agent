package com.aura.proactive

import com.aura.memory.MemoryStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Auto-starts the proactive layer when the app process starts. The
 * AuraApp Application class calls bootstrap() in onCreate.
 *
 * Also runs a one-shot memory decay pass on startup. This is what the
 * Python codebase used to do via a daily cron; the Kotlin port skipped
 * it and [MemoryStore.runDecayPass] was orphan code for the v1 cut.
 * The cost is small (a single Room query of up to 10k rows + a few
 * in-memory computations) and the benefit is real (a memory that's
 * been unused for 60 days actually gets its score nudged down).
 */
@Singleton
class ProactiveBootstrap @Inject constructor(
    @ApplicationContext private val appContext: android.content.Context,
    private val scheduler: ProactiveScheduler,
    private val memoryStore: MemoryStore,
) {
    fun start() {
        scheduler.scheduleMorningBrief()
        scheduler.scheduleDecay()
        CalendarMonitorService.start(appContext)
        // One-shot decay pass on startup so any overdue decay is applied
        // immediately rather than waiting up to 6 hours. The periodic worker
        // handles the "app sat idle for days" case.
        kotlinx.coroutines.CoroutineScope(
            kotlinx.coroutines.SupervisorJob() + kotlinx.coroutines.Dispatchers.IO
        ).launch {
            runCatching { memoryStore.runDecayPass() }
        }
    }
}
