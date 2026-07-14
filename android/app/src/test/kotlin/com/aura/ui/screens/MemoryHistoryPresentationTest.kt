package com.aura.ui.screens

import com.aura.memory.MemoryEditEntity
import kotlin.test.Test
import kotlin.test.assertEquals

class MemoryHistoryPresentationTest {
    @Test
    fun `category change is the primary history label`() {
        val edit = MemoryEditEntity(
            memoryId = "m1",
            oldContent = "same",
            newContent = "same",
            oldCategory = "fact",
            newCategory = "preference",
        )
        assertEquals("fact → preference", memoryEditHeadline(edit))
    }

    @Test
    fun `content-only change gets a clear label`() {
        val edit = MemoryEditEntity(
            memoryId = "m1",
            oldContent = "old",
            newContent = "new",
            oldCategory = "fact",
            newCategory = "fact",
        )
        assertEquals("Content updated", memoryEditHeadline(edit))
    }
}