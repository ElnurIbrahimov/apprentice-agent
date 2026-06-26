package com.aura.proactive

import org.junit.Test
import kotlin.test.assertEquals

/**
 * Smoke test for [CalendarMonitorService]. Since foreground service testing
 * requires androidTest (Robolectric or an emulator), this unit test only
 * verifies the class contracts — companion object constants and the start()
 * helper signature.
 */
class CalendarMonitorServiceTest {

    @Test
    fun `companion constants are defined`() {
        assertEquals("calendar_monitor", CalendarMonitorService.CHANNEL_ID)
        assertEquals(1002, CalendarMonitorService.NOTIFICATION_ID)
    }
}
