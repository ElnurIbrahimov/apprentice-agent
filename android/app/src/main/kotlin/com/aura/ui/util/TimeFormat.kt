package com.aura.ui.util

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs

/**
 * Format a millisecond timestamp as a human-readable relative time
 * for the chat screen. The output tries to be brief:
 *
 * - < 1 minute ago       → "now"
 * - < 1 hour             → "5m"
 * - same day             → "3h"
 * - yesterday            → "yesterday"
 * - < 7 days             → "3d"
 * - same year            → "Mar 14"
 * - different year       → "Mar 14, 2024"
 */
fun formatRelativeTime(timestampMs: Long, now: Long = System.currentTimeMillis()): String {
    val diff = abs(now - timestampMs)
    return when {
        diff < 60_000L -> "now"
        diff < 3_600_000L -> "${diff / 60_000L}m"
        isSameDay(timestampMs, now) -> "${diff / 3_600_000L}h"
        isYesterday(timestampMs, now) -> "yesterday"
        diff < 7L * 86_400_000L -> "${diff / 86_400_000L}d"
        isSameYear(timestampMs, now) -> SimpleDateFormat("MMM d", Locale.US).format(Date(timestampMs))
        else -> SimpleDateFormat("MMM d, yyyy", Locale.US).format(Date(timestampMs))
    }
}

private fun isSameDay(a: Long, b: Long): Boolean {
    val da = Date(a); val db = Date(b)
    val fmt = SimpleDateFormat("yyyyMMdd", Locale.US)
    return fmt.format(da) == fmt.format(db)
}

private fun isYesterday(ts: Long, now: Long): Boolean {
    val yesterday = now - 86_400_000L
    return isSameDay(ts, yesterday)
}

private fun isSameYear(a: Long, b: Long): Boolean {
    val fmt = SimpleDateFormat("yyyy", Locale.US)
    return fmt.format(Date(a)) == fmt.format(Date(b))
}

/** Full clock time, e.g. "3:42 PM" — used for the timestamp tooltip. */
fun formatClockTime(timestampMs: Long): String =
    SimpleDateFormat("h:mm a", Locale.US).format(Date(timestampMs))