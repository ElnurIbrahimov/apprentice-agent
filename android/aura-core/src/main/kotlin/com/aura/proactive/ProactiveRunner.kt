package com.aura.proactive

import com.aura.memory.MemoryStore
import dagger.Lazy
import javax.inject.Inject
import javax.inject.Singleton

/**
 * "Fire it now" entry points for the proactive loops. Backs the
 * three debug buttons on the Proactive history screen so the user
 * can verify each loop actually works without waiting for its
 * scheduled interval (7am for the brief, 6h for decay, 5min for
 * the calendar poll).
 *
 * The brief builder is injected via [Lazy] because the builder
 * itself depends on the LLM providers — if no provider is
 * configured, we still want the runner to construct (Hilt graph
 * should never fail just because the user hasn't set up a key
 * yet). The Lazy defers the work until the user actually taps
 * "fire now".
 */
@Singleton
class ProactiveRunner @Inject constructor(
    private val briefBuilder: Lazy<MorningBriefBuilder>,
    private val memoryStore: MemoryStore,
    private val calendarMonitor: CalendarMonitor,
) {
    /** Run a morning brief right now. Returns "ok" / "no-content" / error. */
    suspend fun fireMorningBrief(): RunResult = try {
        briefBuilder.get().runNow()
        RunResult.Ok("Brief fired")
    } catch (e: kotlinx.coroutines.CancellationException) {
        throw e
    } catch (e: Exception) {
        RunResult.Error("Brief failed: ${e.message ?: e.javaClass.simpleName}")
    }

    /** Run a memory decay pass right now. */
    suspend fun fireDecayPass(): RunResult = try {
        memoryStore.runDecayPass()
        RunResult.Ok("Decay pass fired")
    } catch (e: kotlinx.coroutines.CancellationException) {
        throw e
    } catch (e: Exception) {
        RunResult.Error("Decay failed: ${e.message ?: e.javaClass.simpleName}")
    }

    /** Force the calendar monitor to poll once. */
    suspend fun fireCalendarCheck(): RunResult = try {
        calendarMonitor.pollOnce()
        RunResult.Ok("Calendar poll fired")
    } catch (e: kotlinx.coroutines.CancellationException) {
        throw e
    } catch (e: Exception) {
        RunResult.Error("Calendar check failed: ${e.message ?: e.javaClass.simpleName}")
    }

    sealed class RunResult {
        data class Ok(val message: String) : RunResult()
        data class Error(val message: String) : RunResult()
    }
}
