package com.aura.ui.util

import com.aura.proactive.BriefContext

/**
 * Render a [BriefContext] as a compact one-line-per-section summary.
 * Used by both the Home screen's proactive event card and the
 * ProactiveHistoryScreen's history card so the user sees what
 * actually changed today instead of a generic placeholder.
 */
fun BriefContext.toSummary(): String {
    val lines = mutableListOf<String>()
    if (decayedMemories.isNotEmpty()) {
        val n = decayedMemories.size
        val preview = decayedMemories.take(3)
            .joinToString(" \u00B7 ") { it.content.take(40) }
        lines += if (n == 1) "1 memory fading: $preview" else "$n memories fading: $preview"
    }
    if (newMemories.isNotEmpty()) {
        val n = newMemories.size
        lines += if (n == 1) "1 new thing you told me" else "$n new things you told me"
    }
    if (newKgNodes.isNotEmpty()) {
        val n = newKgNodes.size
        lines += if (n == 1) "1 fact learned" else "$n facts learned"
    }
    if (tasksDueToday.isNotEmpty()) {
        val n = tasksDueToday.size
        val titles = tasksDueToday.take(3).joinToString(" \u00B7 ") { it.title }
        lines += if (n == 1) "1 task due today: $titles" else "$n tasks due today: $titles"
    }
    if (calendarToday.isNotEmpty()) {
        lines += "Today: ${calendarToday.take(3).joinToString(" \u00B7 ")}"
    }
    return lines.joinToString("\n")
}