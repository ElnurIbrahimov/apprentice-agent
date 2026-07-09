package com.aura.core.error

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Persistent crash/error logger. Writes errors to a rolling file in
 * the app cache dir so they survive the 5-second UI auto-dismiss.
 *
 * The log file is capped at [MAX_LOG_BYTES] (100KB). When the cap is
 * exceeded, the oldest half is truncated — keeping the most recent
 * errors which are the ones the user is most likely to investigate.
 *
 * No telemetry — the log is local only, never sent anywhere. The
 * user can view it via Settings → Diagnostics → View error log.
 */
@Singleton
class CrashLogger @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
    private val logFile: File by lazy { File(context.cacheDir, LOG_FILENAME) }

    /**
     * Append an error entry to the log file.
     *
     * @param code Error code (e.g. "http_500", "tool_error")
     * @param message Human-readable error message
     * @param stackTrace Optional stack trace string
     */
    fun log(code: String, message: String, stackTrace: String? = null) {
        try {
            val timestamp = dateFormat.format(Date())
            val entry = buildString {
                append("[$timestamp] $code: $message")
                if (stackTrace != null) {
                    append("\n")
                    append(stackTrace.take(2000))  // cap trace length
                }
                append("\n")
            }
            synchronized(this) {
                logFile.appendText(entry)
                rollIfNeeded()
            }
        } catch (_: Exception) {
            // Best-effort logging — if the filesystem is full or
            // read-only, we can't do anything about it. Don't crash
            // the app trying to log an error.
        }
    }

    /**
     * Read the full log contents. Returns empty string if the log
     * file doesn't exist yet.
     */
    fun read(): String = try {
        if (logFile.exists()) logFile.readText() else ""
    } catch (_: Exception) {
        ""
    }

    /**
     * Delete the log file. Used by the "Clear log" button in Settings.
     */
    fun clear() {
        try {
            logFile.delete()
        } catch (_: Exception) {}
    }

    /**
     * If the log file exceeds [MAX_LOG_BYTES], truncate to the last
     * half so the most recent errors are preserved.
     */
    private fun rollIfNeeded() {
        if (!logFile.exists() || logFile.length() <= MAX_LOG_BYTES) return
        try {
            val content = logFile.readText()
            var keepFrom = content.length / 2
            // Advance to the next newline after the midpoint so we
            // don't split a log entry or a surrogate pair.
            val nextNewline = content.indexOf('\n', keepFrom)
            if (nextNewline != -1) keepFrom = nextNewline + 1
            val truncated = content.substring(keepFrom)
            logFile.writeText(truncated)
        } catch (_: Exception) {}
    }

    companion object {
        private const val LOG_FILENAME = "aura-error-log.txt"
        private const val MAX_LOG_BYTES = 100 * 1024L  // 100KB
    }
}