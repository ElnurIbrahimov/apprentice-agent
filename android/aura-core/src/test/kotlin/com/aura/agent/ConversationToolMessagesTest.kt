package com.aura.agent

import com.aura.providers.ProviderMessage
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * Locks the provider-message contract for tool turns (bug C2).
 *
 * A turn that executed tools must serialize back to the provider as:
 *   1. an assistant message carrying the `tool_calls` (id/name/arguments), and
 *   2. one tool-role message per call carrying the matching `tool_call_id`.
 *
 * Without (1) the OpenAI-compatible providers reject the tool-role messages
 * with a 400 ("must be a response to a preceding message with 'tool_calls'").
 */
class ConversationToolMessagesTest {

    @Test
    fun `tool turn serializes assistant tool_calls followed by tool results`() {
        val conv = Conversation()
            .addUser("what's the weather?")
            .addToolCall(id = "call_1", name = "web_search", args = "{\"query\":\"weather\"}")
            .setToolResult(id = "call_1", result = "Sunny, 24C")

        val messages = conv.toMessages()

        // user, assistant(tool_calls), tool(result)
        assertEquals(3, messages.size, "expected user + assistant + tool, got: $messages")

        val user = messages[0]
        assertEquals(ProviderMessage.Role.user, user.role)

        val assistant = messages[1]
        assertEquals(ProviderMessage.Role.assistant, assistant.role)
        assertNotNull(assistant.toolCalls, "assistant message must carry tool_calls")
        assertEquals(1, assistant.toolCalls!!.size)
        assertEquals("call_1", assistant.toolCalls!![0].id)
        assertEquals("web_search", assistant.toolCalls!![0].name)
        assertEquals("{\"query\":\"weather\"}", assistant.toolCalls!![0].arguments)

        val toolResult = messages[2]
        assertEquals(ProviderMessage.Role.tool, toolResult.role)
        assertEquals("call_1", toolResult.toolCallId, "tool result must reference its call id")
        assertEquals("Sunny, 24C", toolResult.content)
    }

    @Test
    fun `plain assistant turn has no tool_calls`() {
        val conv = Conversation()
            .addUser("hi")
            .addAssistant("hello there")

        val messages = conv.toMessages()
        assertEquals(2, messages.size)
        assertEquals(ProviderMessage.Role.assistant, messages[1].role)
        assertNull(messages[1].toolCalls, "a plain reply must not carry tool_calls")
        assertEquals("hello there", messages[1].content)
    }

    @Test
    fun `assistant text and tool call ride on the same assistant message`() {
        val conv = Conversation()
            .addUser("remember I like tea, then confirm")
            .addAssistant("Sure, saving that.")
            .addToolCall(id = "call_2", name = "remember", args = "{\"fact\":\"likes tea\"}")
            .setToolResult(id = "call_2", result = "stored")

        val messages = conv.toMessages()
        val assistant = messages.first { it.role == ProviderMessage.Role.assistant }
        assertEquals("Sure, saving that.", assistant.content)
        assertNotNull(assistant.toolCalls)
        assertEquals("remember", assistant.toolCalls!![0].name)
    }
}
