package com.aura.memory

import kotlin.math.ln
import kotlin.math.pow

/**
 * 2-week half-life decay. Mirrors aura/memory/fade_mem.py.
 * After 14 days without access, decayScore halves. After 28 days, quarters.
 * After 56 days, 1/16th. Touched memories are bumped (handled in MemoryStore.touch).
 */
object FadeMem {
    const val HALF_LIFE_DAYS = 14.0

    fun compute(createdAt: Long, accessedAt: Long, now: Long = System.currentTimeMillis()): Float {
        val ageDays = (now - accessedAt).coerceAtLeast(0L) / 86_400_000.0
        // Geometric decay: e^(-age * ln(2) / halfLife)
        val decay = 0.5.pow(ageDays / HALF_LIFE_DAYS)
        // Floor based on absolute age: 2 years = 0
        val totalDays = (now - createdAt) / 86_400_000.0
        val floor = (1.0 - totalDays / 730.0).coerceIn(0.0, 1.0)
        return (decay * floor).toFloat().coerceIn(0f, 1f)
    }

    fun shouldKeep(memory: MemoryEntity, now: Long = System.currentTimeMillis()): Boolean {
        return compute(memory.createdAt, memory.accessedAt, now) > 0.01f
    }
}
