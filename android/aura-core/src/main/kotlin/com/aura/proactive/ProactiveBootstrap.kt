package com.aura.proactive

import android.content.Intent
import com.aura.data.UserPreferences
import com.aura.memory.MemoryStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Auto-starts the proactive layer when the app process starts. The
 * AuraApp Application class calls bootstrap() in onCreate.
 *
 * Each proactive worker is gated on a [UserPreferences] flag read
 * at bootstrap time. Default behavior is unchanged (both workers
 * are on for a fresh install) — the toggles are opt-out. When a
 * toggle flips to false, the matching cancel path runs so any
 * in-flight work is removed and the user-visible signal
 * (notification, foreground service) goes away.
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
    private val userPreferences: UserPreferences,
) {
    /**
     * Internal scope used to fire-and-forget the startup decay
     * pass and the async gate reads. Keeping it scoped to the
     * singleton (SupervisorJob on IO) means we don't leak a
     * coroutine when the process is about to die and we don't
     * block the main thread on the DataStore / Room reads.
     */
    private val scope = kotlinx.coroutines.CoroutineScope(
        kotlinx.coroutines.SupervisorJob() + kotlinx.coroutines.Dispatchers.IO
    )

    private var preferenceJob: kotlinx.coroutines.Job? = null

    fun start() {
        // Keep one long-lived reconciliation collector. DataStore flows emit
        // their persisted defaults immediately and every Settings mutation
        // thereafter, so schedules and the foreground service converge in
        // the active process instead of waiting for a restart.
        if (preferenceJob?.isActive != true) {
            preferenceJob = scope.launch {
                combine(
                    userPreferences.morningBriefEnabled,
                    userPreferences.calendarMonitorEnabled,
                    userPreferences.morningBriefHour,
                ) { morningBriefOn, calendarMonitorOn, briefHour ->
                    Triple(morningBriefOn, calendarMonitorOn, briefHour)
                }
                    .distinctUntilChanged()
                    .collect { (morningBriefOn, calendarMonitorOn, briefHour) ->
                        reconcile(morningBriefOn, calendarMonitorOn, briefHour)
                    }
            }
        }

        // Startup decay remains one-shot. It must not rerun when the user
        // merely changes a schedule toggle or time.
        scope.launch {
            runCatching { memoryStore.runDecayPass() }
                .onFailure { error ->
                    try {
                        android.util.Log.w(
                            "ProactiveBootstrap",
                            "startup decay pass failed: ${error.message}",
                        )
                    } catch (_: RuntimeException) {
                        // android.util.Log is unavailable in pure JVM tests.
                    }
                }
        }
    }

    private fun reconcile(
        morningBriefOn: Boolean,
        calendarMonitorOn: Boolean,
        briefHour: Int,
    ) {
        val decisions = applyGates(morningBriefOn, calendarMonitorOn, briefHour)
        try {
            if (decisions.calendarMonitorShouldRun) {
                CalendarMonitorService.start(appContext)
            } else {
                appContext.stopService(
                    Intent(appContext, CalendarMonitorService::class.java),
                )
            }
            val refresh = Intent(ACTION_REFRESH_WIDGET).apply {
                setPackage(appContext.packageName)
            }
            appContext.sendBroadcast(refresh)
        } catch (_: Throwable) {
            // Scheduling is best-effort; the next DataStore emission or
            // process launch reconciles it again.
        }
    }

    /**
     * Apply the morning-brief + calendar-monitor gates. Pure-Kotlin,
     * no Context, no Android framework — this is the seam the unit
     * tests exercise.
     *
     * Morning brief is scheduled (or cancelled) via the scheduler.
     * Calendar monitor is a foreground service, so its gating is a
     * decision rather than a call: the actual start/stop lives in
     * [start] because it needs a real Context. Returning the gate
     * decision lets the caller dispatch the FGS side effect without
     * duplicating the boolean math.
     */
    internal fun applyGates(morningBriefOn: Boolean, calendarMonitorOn: Boolean, briefHour: Int = 7): GatedDecisions {
        if (morningBriefOn) {
            scheduler.scheduleMorningBrief(briefHour)
            scheduler.scheduleDecay()
        } else {
            scheduler.cancelMorningBrief()
            scheduler.cancelDecay()
        }
        return GatedDecisions(
            morningBriefScheduled = morningBriefOn,
            calendarMonitorShouldRun = calendarMonitorOn,
        )
    }

    data class GatedDecisions(
        val morningBriefScheduled: Boolean,
        val calendarMonitorShouldRun: Boolean,
    )

    companion object {
        /** Custom broadcast action the [com.aura.widget.AskAuraWidget] listens for. */
        const val ACTION_REFRESH_WIDGET = "com.aura.action.REFRESH_WIDGET"
    }
}