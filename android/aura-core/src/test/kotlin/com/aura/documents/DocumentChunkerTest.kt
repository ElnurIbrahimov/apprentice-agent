package com.aura.documents

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DocumentChunkerTest {
    @Test
    fun `short document stays in one clean chunk`() {
        val chunks = DocumentChunker.chunk("  First paragraph.\n\nSecond paragraph.  ")
        assertEquals(listOf("First paragraph.\n\nSecond paragraph."), chunks)
    }

    @Test
    fun `long document is bounded and retains overlap`() {
        val text = (1..160).joinToString(" ") { "word$it" }
        val chunks = DocumentChunker.chunk(text, maxChars = 220, overlapChars = 40)

        assertTrue(chunks.size > 1)
        assertTrue(chunks.all { it.length <= 220 })
        for (index in 0 until chunks.lastIndex) {
            val tailWords = chunks[index].takeLast(40).trim().split(Regex("\\s+")).takeLast(2)
            assertTrue(tailWords.any { it in chunks[index + 1] })
        }
    }

    @Test
    fun `empty and whitespace-only documents produce no chunks`() {
        assertTrue(DocumentChunker.chunk(" \n\t ").isEmpty())
    }
}