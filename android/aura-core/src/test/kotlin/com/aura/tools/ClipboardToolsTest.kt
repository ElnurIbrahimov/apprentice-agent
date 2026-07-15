package com.aura.tools

import android.content.Context
import android.content.ClipData
import android.content.ClipboardManager
import androidx.test.core.app.ApplicationProvider
import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import kotlinx.coroutines.test.runTest
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [30])
class ClipboardToolsTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()

    @Test
    fun `clipboard_write copies text to clipboard`() = runTest {
        val tool = ClipboardWriteTool(context).tool
        val result = tool.execute(ToolCall("id-1", tool.name, mapOf("text" to "hello world")),
            ToolContext(conversationId = "test"))
        assertTrue(result is ToolResult.Ok)
        val manager = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        assertEquals("hello world", manager.primaryClip?.getItemAt(0)?.text)
    }

    @Test
    fun `clipboard_read returns empty when clipboard is empty`() = runTest {
        val tool = ClipboardReadTool(context).tool
        val result = tool.execute(ToolCall("id-2", tool.name, emptyMap()),
            ToolContext(conversationId = "test"))
        assertTrue(result is ToolResult.Ok)
    }

    @Test
    fun `clipboard_read returns text from clipboard`() = runTest {
        val manager = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        manager.setPrimaryClip(ClipData.newPlainText("test", "copied text"))
        val tool = ClipboardReadTool(context).tool
        val result = tool.execute(ToolCall("id-3", tool.name, emptyMap()),
            ToolContext(conversationId = "test"))
        assertTrue(result is ToolResult.Ok)
        assertEquals("copied text", (result as ToolResult.Ok).output)
    }
}
