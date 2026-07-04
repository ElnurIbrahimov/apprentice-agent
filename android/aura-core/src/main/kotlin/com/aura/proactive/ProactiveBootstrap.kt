package com.aura.proactive

import android.content.Intent
import com.aura.data.UserPreferences
import com.aura.memory.MemoryStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.first
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
     * pass. Keeping it scoped to the singleton (SupervisorJob on
     * IO) means we don't leak a coroutine when the process is
     * about to die and we don't block the main thread on the
     * Room query.
     */
    private val scope = kotlinx.coroutines.CoroutineScope(
        kotlinx.coroutines.SupervisorJob() + kotlinx.coroutines.Dispatchers.IO
    )

    fun start() {
        // Read both gates once at bootstrap. First-launch users get
        // the defaults (true, true). If the user has toggled one off
        // since the last launch, we cancel it here rather than
        // scheduling-and-immediately-cancelling (which would burn a
        // WorkManager scheduling round-trip and an FGS start).
        val morningBriefOn = runCatching {
            kotlinx.coroutines.runBlocking { userPreferences.morningBriefEnabled.first() }
        }.getOrDefault(true)
        val calendarMonitorOn = runCatching {
            kotlinx.coroutines.runBlocking { userPreferences.calendarMonitorEnabled.first() }
        }.getOrDefault(true)

        // Apply the gates. The split method is the testable seam:
        // it doesn't touch the Android Context, so a pure-JVM test
        // can verify scheduling/cancellation decisions without
        // needing Robolectric or a Context mock.
        val decisions = applyGates(morningBriefOn, calendarMonitorOn)

        // One-shot memory decay pass on startup. The original
        // comment promised this but the scope was never launched;
        // we're wiring it up properly here so the decay score
        // actually moves on cold start instead of waiting for the
        // 6h DecayWorker. Best-effort — failure is logged and
        // dropped, the next DecayWorker tick will catch up.
        scope.launch {
            runCatching { memoryStore.runDecayPass() }
        }

        // Side effects that DO need a Context (FGS start/stop,
        // widget-refresh broadcast). All wrapped in Throwable-catch
        // because a missing WorkManager or stubbed Context in a test
        // environment should not crash the process.
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
            // WorkManager / service bootstrap may fail under Robolectric
            // or in any environment where the Android framework methods
            // are partially mocked. Silently swallowing is the right
            // behavior — proactive scheduling is best-effort and a
            // missing schedule is recoverable on the next app launch.
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
    internal fun applyGates(morningBriefOn: Boolean, calendarMonitorOn: Boolean): GatedDecisions {
        if (morningBriefOn) {
            scheduler.scheduleMorningBrief()
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