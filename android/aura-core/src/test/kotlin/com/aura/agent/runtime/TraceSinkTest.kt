package com.aura.agent.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TraceSinkTest {

    @Test
    fun emit_and_retrieve_event() {
        val sink = TraceSink()
        val event = sink.emit(
            runId = "run1",
            type = TraceEventType.RUN_STARTED,
        )
        assertEquals("run1", event.runId)
        assertEquals(TraceEventType.RUN_STARTED, event.type)
        assertEquals(1, sink.count())
    }

    @Test
    fun forRun_returns_only_matching_events() {
        val sink = TraceSink()
        sink.emit(runId = "run1", type = TraceEventType.RUN_STARTED)
        sink.emit(runId = "run2", type = TraceEventType.RUN_STARTED)
        sink.emit(runId = "run1", type = TraceEventType.STEP_COMPLETED, stepId = "s1")
        val run1Events = sink.forRun("run1")
        assertEquals(2, run1Events.size)
        assertTrue(run1Events.all { it.runId == "run1" })
    }

    @Test
    fun events_are_ordered_by_timestamp() {
        val sink = TraceSink()
        sink.emit(runId = "run1", type = TraceEventType.RUN_STARTED)
        Thread.sleep(2)
        sink.emit(runId = "run1", type = TraceEventType.STEP_STARTED, stepId = "s1")
        Thread.sleep(2)
        sink.emit(runId = "run1", type = TraceEventType.STEP_COMPLETED, stepId = "s1")
        val events = sink.forRun("run1")
        assertEquals(3, events.size)
        // Ordered by timestamp ascending
        assertTrue(events[0].timestamp <= events[1].timestamp)
        assertTrue(events[1].timestamp <= events[2].timestamp)
    }

    @Test
    fun ring_buffer_evicts_oldest_when_full() {
        val sink = TraceSink()
        // Fill with 10k + 100 events
        repeat(10100) { i ->
            sink.emit(runId = "run$i", type = TraceEventType.RUN_STARTED)
        }
        // Should be capped at ~10k
        assertTrue("Expected <= 10000 events, got ${sink.count()}", sink.count() <= 10000)
    }

    @Test
    fun clear_resets_all_events() {
        val sink = TraceSink()
        sink.emit(runId = "run1", type = TraceEventType.RUN_STARTED)
        sink.emit(runId = "run1", type = TraceEventType.RUN_COMPLETED)
        assertEquals(2, sink.count())
        sink.clear()
        assertEquals(0, sink.count())
    }

    @Test
    fun concurrent_emits_are_thread_safe() {
        val sink = TraceSink()
        val threads = (1..10).map { i ->
            Thread {
                repeat(100) {
                    sink.emit(runId = "run$i", type = TraceEventType.TOOL_CALL, toolName = "test")
                }
            }
        }
        threads.forEach(Thread::start)
        threads.forEach(Thread::join)
        assertEquals(1000, sink.count())
    }
}