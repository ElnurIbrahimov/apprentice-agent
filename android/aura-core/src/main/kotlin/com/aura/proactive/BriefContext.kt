package com.aura.proactive

import com.aura.kg.KgNode
import com.aura.memory.MemoryEntity
import com.aura.tasks.TaskEntity
import kotlinx.serialization.Serializable

/**
 * Structured context for the morning brief.
 *
 * The old brief was a single LLM call against the user's last 10
 * memories — the model would produce a freeform 3-5 line paragraph
 * that rarely said anything concrete. This data class carries the
 * five sections the brief should actually cover, queried in
 * parallel and joined at render time:
 *
 *   - decayedMemories: memories whose [MemoryEntity.decayScore]
 *     has dropped below the threshold since the last brief. The
 *     brief says "X memories are fading; tap to review."
 *   - newMemories: memories created since the last brief (or in
 *     the last 24h). The brief says "Y new things you told me."
 *   - newKgNodes: knowledge-graph nodes updated since the last
 *     brief. The brief says "Z facts I learned."
 *   - tasksDueToday: pending tasks with dueAt in today's range.
 *     The brief says "N tasks due today."
 *   - calendarToday: formatted calendar lines for today (already
 *     computed by the home screen; brief re-uses the same query).
 *
 * Each list is bounded so a 5000-memory install doesn't blow up
 * the LLM prompt. The brief renders a one-line summary per
 * non-empty section, plus an LLM-written greeting on top.
 *
 * Marked [Serializable] so the event store can persist the
 * structured body as JSON in the `proactive_events` table.
 */
@Serializable
data class BriefContext(
    val decayedMemories: List<MemoryEntity> = emptyList(),
    val newMemories: List<MemoryEntity> = emptyList(),
    val newKgNodes: List<KgNode> = emptyList(),
    val tasksDueToday: List<TaskEntity> = emptyList(),
    val calendarToday: List<String> = emptyList(),
) {
    /** True if every section is empty — caller can skip the LLM call. */
    val isEmpty: Boolean
        get() = decayedMemories.isEmpty() &&
            newMemories.isEmpty() &&
            newKgNodes.isEmpty() &&
            tasksDueToday.isEmpty() &&
            calendarToday.isEmpty()
}
