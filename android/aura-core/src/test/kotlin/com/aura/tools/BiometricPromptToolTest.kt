package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.security.BiometricActivityHolder
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for [BiometricPromptTool] and its pure-Kotlin helper [BiometricAuthHandler].
 *
 * The tool's execute() is a suspend function, so we use [runTest] from
 * kotlinx-coroutines-test.
 */
class BiometricPromptToolTest {

    // -----------------------------------------------------------------
    // Tool-level: null activity → NeedsApproval
    // -----------------------------------------------------------------

    @Test
    fun `execute returns NeedsApproval when no activity attached`() = runTest {
        val holder = mockk<BiometricActivityHolder>()
        every { holder.activity } returns null

        val tool = BiometricPromptTool(holder)
        val result = tool.tool.execute(
            ToolCall(id = "1", name = "biometric_prompt", arguments = mapOf("title" to "Test")),
            ToolContext(conversationId = "conv-1"),
        )

        when (result) {
            is ToolResult.NeedsApproval -> {
                assertTrue(
                    result.rationale.contains("no foreground activity"),
                    "Rationale should mention missing activity: ${result.rationale}",
                )
            }
            else -> throw AssertionError("Expected NeedsApproval, got $result")
        }
    }

    @Test
    fun `execute returns Error when title is missing`() = runTest {
        val holder = mockk<BiometricActivityHolder>()
        // Even with an activity, missing title should bail early with Error
        every { holder.activity } returns mockk()

        val tool = BiometricPromptTool(holder)
        val result = tool.tool.execute(
            ToolCall(id = "1", name = "biometric_prompt", arguments = emptyMap()),
            ToolContext(conversationId = "conv-1"),
        )

        when (result) {
            is ToolResult.Error -> assertEquals("missing 'title'", result.message)
            else -> throw AssertionError("Expected Error, got $result")
        }
    }

    // -----------------------------------------------------------------
    // BiometricAuthHandler — pure-Kotlin callback routing (JVM-safe)
    // -----------------------------------------------------------------

    @Test
    fun `handler routes onAuthenticated to success`() = runTest {
        val handler = BiometricAuthHandler()
        handler.onAuthenticated()

        val result = handler.result.await()
        assertTrue(result.success, "Expected success = true")
    }

    @Test
    fun `handler routes onError with user-cancel code`() = runTest {
        val handler = BiometricAuthHandler()
        handler.onError(13, "User cancelled the operation")

        val result = handler.result.await()
        assertFalse(result.success, "Expected success = false")
        assertEquals(13, result.errorCode)
        assertEquals("User cancelled the operation", result.errorMessage)
    }

    @Test
    fun `handler routes onError with generic error`() = runTest {
        val handler = BiometricAuthHandler()
        handler.onError(7, "Locked out after too many attempts")

        val result = handler.result.await()
        assertFalse(result.success, "Expected success = false")
        assertEquals(7, result.errorCode)
        assertEquals("Locked out after too many attempts", result.errorMessage)
    }

    @Test
    fun `handler completes only once`() = runTest {
        val handler = BiometricAuthHandler()
        handler.onAuthenticated()

        val result = handler.result.await()
        assertTrue(result.success, "First completion should win")

        // Verify the deferred is done after the first completion
        assertTrue(handler.result.isCompleted, "Deferred should be completed after first call")
    }
}
