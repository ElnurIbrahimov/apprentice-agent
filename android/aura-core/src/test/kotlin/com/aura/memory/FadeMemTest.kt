package com.aura.memory

import org.junit.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FadeMemTest {

    @Test
    fun `shouldKeep returns true for fresh memory`() {
        val now = System.currentTimeMillis()
        val mem = MemoryEntity(
            id = "test",
            content = "fresh",
            source = "user",
            category = "fact",
            createdAt = now,
            accessedAt = now,
        )
        assertTrue(FadeMem.shouldKeep(mem, now))
    }

    @Test
    fun `shouldKeep returns false for very old memory`() {
        val now = System.currentTimeMillis()
        val old = now - 3650L * 86_400_000  // 10 years
        val mem = MemoryEntity(
            id = "test",
            content = "ancient",
            source = "user",
            category = "fact",
            createdAt = old,
            accessedAt = old,
        )
        assertFalse(FadeMem.shouldKeep(mem, now))
    }
}
