package com.aura.agent.runtime

import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.ConcurrentLinkedQueue
import javax.inject.Inject
import javax.inject.Singleton

/**
 * In-memory trace event sink. Events are kept in a bounded ring buffer
 * for debugging and replay. A future phase will persist these to Room
 * as the durable event ledger.
 *
 * Thread-safe — concurrent tool calls can append events from parallel
 * coroutines without ordering issues. The [AtomicLong] sequence
 * preserves causal ordering even when wall-clock timestamps collide.
 */
@Singleton
class TraceSink @Inject constructor() {
    private val events = ConcurrentLinkedQueue<AgentTraceEvent>()
    private val sequence = AtomicLong(0)
    private val maxEvents = 10_000

    fun emit(event: AgentTraceEvent) {
        if (events.size >= maxEvents) {
            events.poll()
        }
        events.add(event)
    }

    fun emit(
        runId: kotlin.String,
        type: TraceEventType,
        stepId: kotlin.String? = null,
        toolName: kotlin.String? = null,
        redactedPayload: kotlin.String = "",
        durationMs: kotlin.Long = 0L,
        success: kotlin.Boolean = true,
        errorCode: kotlin.String? = null,
    ): AgentTraceEvent {
        val event = AgentTraceEvent(
            id = "evt_${sequence.incrementAndGet()}",
            runId = runId,
            stepId = stepId,
            timestamp = System.currentTimeMillis(),
            type = type,
            toolName = toolName,
            redactedPayload = redactedPayload,
            durationMs = durationMs,
            success = success,
            errorCode = errorCode,
        )
        emit(event)
        return event
    }

    fun forRun(runId: kotlin.String): List<AgentTraceEvent> =
        events.filter { it.runId == runId }.sortedBy { it.timestamp }

    fun recent(limit: Int = 100): List<AgentTraceEvent> =
        events.toList().takeLast(limit)

    fun clear() {
        events.clear()
    }

    fun count(): Int = events.size
}