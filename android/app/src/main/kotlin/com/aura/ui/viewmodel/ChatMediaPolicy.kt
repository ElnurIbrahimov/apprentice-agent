package com.aura.ui.viewmodel

import com.aura.agent.Conversation
import java.io.ByteArrayOutputStream
import java.io.InputStream

/** Attach a completed tool result with the same ID used for the call. */
internal fun Conversation.attachCompletedToolTurn(
    id: String,
    name: String,
    arguments: String,
    result: String,
): Conversation = addToolCall(id, name, arguments).setToolResult(id, result)

/**
 * Read at most [maxBytes] from [input]. Returns null as soon as byte
 * [maxBytes] + 1 is observed, so oversized content is never fully buffered.
 */
internal fun readStreamWithinLimit(input: InputStream, maxBytes: Int): ByteArray? {
    require(maxBytes >= 0) { "maxBytes must be non-negative" }
    val output = ByteArrayOutputStream(minOf(maxBytes, 64 * 1024))
    val buffer = ByteArray(8 * 1024)
    var remaining = maxBytes
    while (true) {
        val requested = minOf(buffer.size, remaining + 1)
        val read = input.read(buffer, 0, requested)
        if (read < 0) return output.toByteArray()
        if (read > remaining) return null
        output.write(buffer, 0, read)
        remaining -= read
    }
}

/** Power-of-two BitmapFactory sample size that keeps the largest side near target. */
internal fun calculateImageSampleSize(width: Int, height: Int, target: Int): Int {
    if (target <= 0) return 1
    val largest = maxOf(width, height).coerceAtLeast(1)
    var sample = 1
    while (largest / sample > target && sample <= Int.MAX_VALUE / 2) {
        sample *= 2
    }
    return sample
}
