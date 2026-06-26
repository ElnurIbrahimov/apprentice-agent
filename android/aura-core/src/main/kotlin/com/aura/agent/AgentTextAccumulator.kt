package com.aura.agent

/**
 * Accumulator for assistant text deltas. Used by the agent loop and the
 * TextToSpeech pipeline to avoid reading tool result noise aloud.
 *
 * The contract: only AgentEvent.TextDelta events contribute to the
 * accumulated text. Tool calls, tool results, errors, and finish events
 * MUST NOT change the accumulated text.
 *
 * Exposed as a pure function (not a class field) so the test can verify
 * the rule without spinning up a coroutine.
 */
object AgentTextAccumulator {
    fun apply(current: String, event: AgentEvent): String = when (event) {
        is AgentEvent.TextDelta -> current + event.text
        else -> current
    }
}
