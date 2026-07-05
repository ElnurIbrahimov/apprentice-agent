package com.aura.proactive

import android.content.Context
import com.aura.data.UserPreferences
import com.aura.memory.MemoryStore
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.flow.flowOf
import org.junit.Before
import org.junit.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests the [ProactiveBootstrap.applyGates] gating logic. The
 * bootstrap reads the morning-brief and calendar-monitor
 * preference gates and only schedules / cancels the matching
 * worker when the gate is on. When the gate is off, any
 * previously-scheduled worker is cancelled.
 *
 * The FGS start/stop and widget-refresh broadcast paths in
 * [ProactiveBootstrap.start] need a real Android Context and
 * are not unit-tested here — they're exercised in instrumented
 * tests or on-device. The pure-Kotlin gate decision is the
 * thing this PR adds, and that's what applyGates covers.
 */
class ProactiveBootstrapTest {

    private lateinit var context: Context
    private lateinit var scheduler: ProactiveScheduler
    private lateinit var memoryStore: MemoryStore
    private lateinit var userPreferences: UserPreferences

    @Before
    fun setUp() {
        // Context mock is required by the constructor but never
        // exercised through applyGates — relaxed so unused methods
        // don't blow up with AbstractMethodError.
        context = mockk(relaxed = true)
        scheduler = mockk(relaxed = true)
        memoryStore = mockk(relaxed = true)
        userPreferences = mockk(relaxed = true)
        coEvery { memoryStore.runDecayPass() } returns Unit

        // Default: morning brief on, calendar monitor on. Tests
        // override the flow per-case.
        every { userPreferences.morningBriefEnabled } returns flowOf(true)
        every { userPreferences.calendarMonitorEnabled } returns flowOf(true)
    }

    @Test
    fun `morning brief on schedules both workers`() {
        val bootstrap = ProactiveBootstrap(context, scheduler, memoryStore, userPreferences)
        bootstrap.applyGates(morningBriefOn = true, calendarMonitorOn = true)
        verify(exactly = 1) { scheduler.scheduleMorningBrief() }
        verify(exactly = 1) { scheduler.scheduleDecay() }
        verify(exactly = 0) { scheduler.cancelMorningBrief() }
        verify(exactly = 0) { scheduler.cancelDecay() }
    }

    @Test
    fun `morning brief off cancels both workers`() {
        every { userPreferences.morningBriefEnabled } returns flowOf(false)
        val bootstrap = ProactiveBootstrap(context, scheduler, memoryStore, userPreferences)
        bootstrap.applyGates(morningBriefOn = false, calendarMonitorOn = false)
        verify(exactly = 0) { scheduler.scheduleMorningBrief() }
        verify(exactly = 0) { scheduler.scheduleDecay() }
        verify(exactly = 1) { scheduler.cancelMorningBrief() }
        verify(exactly = 1) { scheduler.cancelDecay() }
    }

    @Test
    fun `applyGates returns the morning brief decision`() {
        val bootstrap = ProactiveBootstrap(context, scheduler, memoryStore, userPreferences)
        val decisions = bootstrap.applyGates(morningBriefOn = true, calendarMonitorOn = true)
        assertTrue(decisions.morningBriefScheduled)
        assertTrue(decisions.calendarMonitorShouldRun)
    }

    @Test
    fun `applyGates returns false when morning brief is off`() {
        every { userPreferences.morningBriefEnabled } returns flowOf(false)
        val bootstrap = ProactiveBootstrap(context, scheduler, memoryStore, userPreferences)
        val decisions = bootstrap.applyGates(morningBriefOn = false, calendarMonitorOn = true)
        assertFalse(decisions.morningBriefScheduled)
        assertTrue(decisions.calendarMonitorShouldRun, "calendar monitor is independent of morning brief")
    }

    @Test
    fun `applyGates returns false for calendar monitor when off`() {
        every { userPreferences.calendarMonitorEnabled } returns flowOf(false)
        val bootstrap = ProactiveBootstrap(context, scheduler, memoryStore, userPreferences)
        val decisions = bootstrap.applyGates(morningBriefOn = true, calendarMonitorOn = false)
        assertTrue(decisions.morningBriefScheduled, "morning brief is independent of calendar monitor")
        assertFalse(decisions.calendarMonitorShouldRun)
    }

    @Test
    fun `start reads both prefs and applies the gates`() {
        // End-to-end via start(), but with a fully-stubbed Context
        // so the broadcast / FGS calls don't trip AbstractMethodError.
        // The Throwable-catch in start() absorbs any stub blowups,
        // so the gate decision itself is what we verify.
        //
        // start() now launches its gate-read+apply on Dispatchers.IO
        // (previously used runBlocking on the calling thread). We poll
        // briefly for the scheduler call because the IO dispatcher
        // runs asynchronously from the test thread.
        val bootstrap = ProactiveBootstrap(context, scheduler, memoryStore, userPreferences)
        bootstrap.start()
        // Wait for the async gate read + apply to complete. The IO
        // dispatcher typically resolves this in <50ms; we poll up to
        // 2s to avoid flakiness on CI.
        val deadline = System.currentTimeMillis() + 2_000L
        var scheduled = false
        while (System.currentTimeMillis() < deadline) {
            try {
                verify(exactly = 1) { scheduler.scheduleMorningBrief() }
                scheduled = true
                break
            } catch (_: AssertionError) {
                Thread.sleep(20)
            }
        }
        assertTrue(scheduled, "scheduleMorningBrief was not called within 2s — the async gate read may not have completed")
        verify(exactly = 1) { scheduler.scheduleDecay() }
    }
}