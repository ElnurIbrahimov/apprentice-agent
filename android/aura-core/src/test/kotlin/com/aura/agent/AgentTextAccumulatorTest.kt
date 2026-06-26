package com.aura.agent

import org.junit.Test
import kotlin.test.assertEquals

class AgentTextAccumulatorTest {
    @Test fun `TextDelta appends to current`() {
        assertEquals("hello", AgentTextAccumulator.apply("", AgentEvent.TextDelta("hello")))
        assertEquals("hello world", AgentTextAccumulator.apply("hello ", AgentEvent.TextDelta("world")))
    }

    @Test fun `ToolCallStart does not change text`() {
        assertEquals("abc", AgentTextAccumulator.apply("abc", AgentEvent.ToolCallStart("id1", "echo")))
    }

    @Test fun `ToolExecuting does not change text`() {
        assertEquals("abc", AgentTextAccumulator.apply("abc", AgentEvent.ToolExecuting("id1", "echo", "{}")))
    }

    @Test fun `ToolCallEnd does not change text`() {
        assertEquals("abc", AgentTextAccumulator.apply("abc", AgentEvent.ToolCallEnd("id1", "echo", "{}")))
    }

    @Test fun `ToolResult does not change text`() {
        assertEquals("abc", AgentTextAccumulator.apply("abc", AgentEvent.ToolResult("id1", "echo", "echoed out")))
    }

    @Test fun `Error does not change text`() {
        assertEquals("abc", AgentTextAccumulator.apply("abc", AgentEvent.Error("500", "boom", retryable = true)))
    }

    @Test fun `PermissionGranted does not change text`() {
        assertEquals("abc", AgentTextAccumulator.apply("abc", AgentEvent.PermissionGranted("echo", "{}")))
    }

    @Test fun `Result does not change text`() {
        assertEquals("abc", AgentTextAccumulator.apply("abc", AgentEvent.Result(Conversation())))
    }

    @Test fun `Done does not change text`() {
        assertEquals("abc", AgentTextAccumulator.apply("abc", AgentEvent.Done))
    }
}
