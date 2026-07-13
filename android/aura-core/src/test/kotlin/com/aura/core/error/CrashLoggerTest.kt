package com.aura.core.error

import android.content.Context
import io.mockk.every
import io.mockk.mockk
import java.io.File
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CrashLoggerTest {

    private lateinit var directory: File
    private lateinit var context: Context
    private lateinit var logger: CrashLogger

    @BeforeTest
    fun setUp() {
        directory = Files.createTempDirectory("aura-crash-test").toFile()
        context = mockk()
        every { context.cacheDir } returns directory
        logger = CrashLogger(context)
    }

    @AfterTest
    fun tearDown() {
        directory.deleteRecursively()
    }

    @Test
    fun `structured entry round trips every diagnostic field newest first`() {
        logger.log(
            code = "provider_failure",
            message = "Primary model failed",
            stackTrace = "line one\nline two",
            threadName = "chat-worker",
            timestamp = 1_700_000_000_000L,
        )
        logger.log(
            code = "tool_error",
            message = "Calendar unavailable",
            threadName = "main",
            timestamp = 1_700_000_001_000L,
        )

        val entries = logger.entries()

        assertEquals(listOf("tool_error", "provider_failure"), entries.map { it.code })
        assertEquals("Calendar unavailable", entries[0].message)
        assertEquals("main", entries[0].threadName)
        assertEquals(1_700_000_001_000L, entries[0].timestamp)
        assertEquals("line one\nline two", entries[1].stackTrace)
    }

    @Test
    fun `legacy multiline entries remain readable`() {
        File(directory, "aura-error-log.txt").writeText(
            """
            [2026-07-13 10:15:30] http_500: Server failed
            java.lang.IllegalStateException: boom
                at com.aura.Test.run(Test.kt:7)
            [2026-07-13 10:16:00] tool_error: Tool failed
            """.trimIndent() + "\n",
        )

        val entries = logger.entries()

        assertEquals(2, entries.size)
        assertEquals("tool_error", entries[0].code)
        assertEquals("http_500", entries[1].code)
        assertTrue(entries[1].stackTrace?.contains("IllegalStateException") == true)
        assertTrue(entries[1].stackTrace?.contains("Test.kt:7") == true)
    }

    @Test
    fun `export is an exact copy and clear removes history`() {
        logger.log("test", "diagnostic", timestamp = 10L)

        val exported = logger.exportTo(directory, timestamp = 20L)

        assertTrue(exported.exists())
        assertEquals(logger.read(), exported.readText())
        logger.clear()
        assertTrue(logger.entries().isEmpty())
        assertEquals("", logger.read())
        assertFalse(exported.readText().isBlank())
    }

    @Test
    fun `rolling retains complete parseable newest entries under cap`() {
        repeat(180) { index ->
            logger.log(
                code = "error_$index",
                message = "x".repeat(900),
                stackTrace = "trace-$index\n${"s".repeat(300)}",
                timestamp = index.toLong(),
            )
        }

        val entries = logger.entries()
        val raw = logger.read()

        assertTrue(raw.toByteArray().size <= 100 * 1024)
        assertTrue(entries.isNotEmpty())
        assertEquals("error_179", entries.first().code)
        assertTrue(entries.none { it.code == "error_0" })
        assertTrue(raw.lineSequence().filter { it.isNotBlank() }.all { it.trimStart().startsWith("{") })
    }
}
