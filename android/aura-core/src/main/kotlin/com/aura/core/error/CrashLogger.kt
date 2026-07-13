package com.aura.core.error

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

@Serializable
data class CrashLogEntry(
    val timestamp: Long,
    val code: String,
    val message: String,
    val stackTrace: String? = null,
    val threadName: String = "",
    val fatal: Boolean = false,
)

/**
 * Local-only structured diagnostics logger. New entries use JSON Lines so one
 * physical line is one complete record; [entries] also understands the legacy
 * multiline format shipped before Phase 2.
 *
 * Nothing is uploaded. The only egress is an explicit user share of a file
 * returned by [exportTo].
 */
@Singleton
class CrashLogger @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val logFile: File by lazy { File(context.cacheDir, LOG_FILENAME) }

    /** Append one complete diagnostic record. Never throws. */
    fun log(
        code: String,
        message: String,
        stackTrace: String? = null,
        threadName: String = Thread.currentThread().name,
        timestamp: Long = System.currentTimeMillis(),
        fatal: Boolean = false,
    ) {
        try {
            val entry = CrashLogEntry(
                timestamp = timestamp,
                code = code.trim().ifBlank { "unknown_error" },
                message = message,
                stackTrace = stackTrace?.take(MAX_STACK_TRACE_CHARS),
                threadName = threadName,
                fatal = fatal,
            )
            synchronized(this) {
                logFile.parentFile?.mkdirs()
                logFile.appendText(json.encodeToString(entry) + "\n")
                rollIfNeeded()
            }
        } catch (_: Exception) {
            // Diagnostics must never become the crash.
        }
    }

    /** Convenience for process and worker exception boundaries. */
    fun logException(
        code: String,
        throwable: Throwable,
        fatal: Boolean = false,
        threadName: String = Thread.currentThread().name,
        timestamp: Long = System.currentTimeMillis(),
    ) {
        log(
            code = code,
            message = throwable.message ?: throwable.javaClass.simpleName,
            stackTrace = throwable.stackTraceToString(),
            threadName = threadName,
            timestamp = timestamp,
            fatal = fatal,
        )
    }

    /** Raw on-disk representation, used only for explicit export. */
    fun read(): String = synchronized(this) {
        try {
            if (logFile.exists()) logFile.readText() else ""
        } catch (_: Exception) {
            ""
        }
    }

    /** Parse all supported formats and return newest first. */
    fun entries(): List<CrashLogEntry> = synchronized(this) {
        try {
            if (!logFile.exists()) emptyList()
            else parseEntries(logFile.readText()).asReversed()
        } catch (_: Exception) {
            emptyList()
        }
    }

    /**
     * Copy the exact current log into [directory] for a FileProvider share.
     * Export survives a later [clear], which is deliberate: once the user has
     * asked to share a snapshot it should remain stable for the chooser.
     */
    fun exportTo(
        directory: File = context.cacheDir,
        timestamp: Long = System.currentTimeMillis(),
    ): File = synchronized(this) {
        directory.mkdirs()
        File(directory, "aura-diagnostics-$timestamp.jsonl").apply {
            writeText(if (logFile.exists()) logFile.readText() else "")
        }
    }

    fun clear() {
        synchronized(this) {
            try {
                logFile.delete()
            } catch (_: Exception) {
                // Best effort.
            }
        }
    }

    private fun parseEntries(content: String): List<CrashLogEntry> {
        if (content.isBlank()) return emptyList()
        val parsed = mutableListOf<CrashLogEntry>()
        var legacyHeader: MatchResult? = null
        val legacyStack = mutableListOf<String>()

        fun flushLegacy() {
            val header = legacyHeader ?: return
            val timestamp = runCatching {
                dateFormat.parse(header.groupValues[1])?.time ?: 0L
            }.getOrDefault(0L)
            parsed += CrashLogEntry(
                timestamp = timestamp,
                code = header.groupValues[2].trim(),
                message = header.groupValues[3],
                stackTrace = legacyStack.joinToString("\n").trim().ifBlank { null },
            )
            legacyHeader = null
            legacyStack.clear()
        }

        content.lineSequence().forEach { rawLine ->
            val line = rawLine.removeSuffix("\r")
            if (line.trimStart().startsWith("{")) {
                flushLegacy()
                runCatching { json.decodeFromString<CrashLogEntry>(line) }
                    .getOrNull()
                    ?.let(parsed::add)
                return@forEach
            }
            val header = LEGACY_HEADER.matchEntire(line)
            if (header != null) {
                flushLegacy()
                legacyHeader = header
            } else if (legacyHeader != null) {
                legacyStack += line
            }
        }
        flushLegacy()
        return parsed
    }

    /**
     * Keep complete newest entries under the byte cap. Re-encoding during a
     * roll also upgrades retained legacy multiline entries to JSONL, so a
     * midpoint can never strand half a stack trace at the beginning.
     */
    private fun rollIfNeeded() {
        if (!logFile.exists() || logFile.length() <= MAX_LOG_BYTES) return
        try {
            val newestFirst = parseEntries(logFile.readText()).asReversed()
            val keptNewestFirst = mutableListOf<String>()
            var bytes = 0
            for (entry in newestFirst) {
                val encoded = json.encodeToString(entry) + "\n"
                val encodedBytes = encoded.toByteArray().size
                if (keptNewestFirst.isNotEmpty() && bytes + encodedBytes > ROLL_TARGET_BYTES) break
                keptNewestFirst += encoded
                bytes += encodedBytes
            }
            logFile.writeText(keptNewestFirst.asReversed().joinToString(""))
        } catch (_: Exception) {
            // Leave the existing log intact if rolling fails.
        }
    }

    companion object {
        private const val LOG_FILENAME = "aura-error-log.txt"
        private const val MAX_LOG_BYTES = 100 * 1024L
        private const val ROLL_TARGET_BYTES = MAX_LOG_BYTES / 2
        private const val MAX_STACK_TRACE_CHARS = 8_000
        private val LEGACY_HEADER = Regex(
            """^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})]\s+([^:]+):\s?(.*)$""",
        )
    }
}
