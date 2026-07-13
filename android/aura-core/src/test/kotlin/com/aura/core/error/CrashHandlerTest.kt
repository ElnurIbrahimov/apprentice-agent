package com.aura.core.error

import io.mockk.mockk
import io.mockk.verify
import kotlin.test.Test

class CrashHandlerTest {

    @Test
    fun `uncaught exception is logged once then delegated`() {
        val logger = mockk<CrashLogger>(relaxed = true)
        val delegate = mockk<Thread.UncaughtExceptionHandler>(relaxed = true)
        val handler = CrashHandler(logger, delegate)
        val thread = Thread.currentThread()
        val failure = IllegalStateException("fatal boom")

        handler.uncaughtException(thread, failure)

        verify(exactly = 1) {
            logger.logException(
                code = "uncaught_exception",
                throwable = failure,
                fatal = true,
                threadName = thread.name,
                timestamp = any(),
            )
        }
        verify(exactly = 1) { delegate.uncaughtException(thread, failure) }
    }

    @Test
    fun `handler never delegates to itself`() {
        val logger = mockk<CrashLogger>(relaxed = true)
        val handler = CrashHandler(logger, null)
        val thread = Thread.currentThread()
        val failure = RuntimeException("boom")

        handler.uncaughtException(thread, failure)

        verify(exactly = 1) { logger.logException(any(), failure, true, thread.name, any()) }
    }
}
