package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.notifications.CapturedNotification
import com.aura.notifications.NotificationCaptureStore
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertIs

class NotificationListToolTest {

    @Test
    fun `requires notification-listener access before reading`() = runTest {
        val tool = NotificationListTool(NotificationCaptureStore()).tool

        val result = tool.execute(ToolCall("1", "notification_list", emptyMap()), ToolContext(conversationId = "test"))

        val needed = assertIs<ToolResult.NeedsPermission>(result)
        assertEquals("android.permission.BIND_NOTIFICATION_LISTENER_SERVICE", needed.permission)
    }

    @Test
    fun `returns captured cross-app notifications newest first`() = runTest {
        val store = NotificationCaptureStore().apply {
            setConnected(true)
            upsert(CapturedNotification("one", "com.mail", "Older", "Message", 1L))
            upsert(CapturedNotification("two", "com.chat", "Newer", "Hello", 2L))
        }
        val tool = NotificationListTool(store).tool

        val result = tool.execute(
            ToolCall("1", "notification_list", mapOf("limit" to 1)),
            ToolContext(conversationId = "test"),
        )

        val output = assertIs<ToolResult.Ok>(result).output
        assertContains(output, "com.chat: Newer — Hello")
        assertEquals(false, output.contains("Older"))
    }
}
