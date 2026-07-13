package com.aura.core.error

/**
 * Process-wide uncaught-exception boundary. It records the fatal diagnostic and
 * then delegates to Android's original handler so normal process termination,
 * tombstone generation, and system reporting semantics are preserved.
 */
class CrashHandler(
    private val logger: CrashLogger,
    private val delegate: Thread.UncaughtExceptionHandler?,
) : Thread.UncaughtExceptionHandler {

    override fun uncaughtException(thread: Thread, throwable: Throwable) {
        logger.logException(
            code = "uncaught_exception",
            throwable = throwable,
            fatal = true,
            threadName = thread.name,
        )
        val next = delegate
        if (next != null && next !== this) {
            next.uncaughtException(thread, throwable)
        }
    }

    companion object {
        /** Install once and return the active handler for diagnostics/tests. */
        fun install(logger: CrashLogger): CrashHandler {
            val current = Thread.getDefaultUncaughtExceptionHandler()
            if (current is CrashHandler) return current
            return CrashHandler(logger, current).also {
                Thread.setDefaultUncaughtExceptionHandler(it)
            }
        }
    }
}
