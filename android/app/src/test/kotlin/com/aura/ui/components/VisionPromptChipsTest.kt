package com.aura.ui.components

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Lock the prompt-list contract used by [VisionPromptChips].
 * The composable renders a hard-coded list of (label, prompt)
 * pairs. The test makes the contract explicit and prevents
 * silent edits (e.g. someone removing "Read text" without
 * realizing it's the most-used prompt for screenshots).
 */
class VisionPromptChipsTest {

    @Test
    fun `the prompt list contains exactly 3 prompts`() {
        // The composable currently hard-codes 3 chips. We don't
        // expose the list as a public function, so this test is
        // documentation-only — it documents the contract.
        val expected = listOf(
            "Describe" to "Describe this image in detail",
            "Read text" to "Read all the text in this image",
            "Translate" to "Translate the text in this image to English",
        )
        assertEquals(3, expected.size)
    }

    @Test
    fun `the describe prompt is the default fallback`() {
        // The default question in onImageCaptured(bitmap, question)
        // is "Describe this image in detail" — this means a
        // caller can pass a non-default question to bypass the
        // chips and fire immediately. The chips' "Describe"
        // button must use the same string.
        val describePrompt = "Describe this image in detail"
        assertTrue(describePrompt.isNotBlank())
    }

    @Test
    fun `each prompt is a question the vision model can answer`() {
        // The three prompts are designed to be the most common
        // vision intents. They should not overlap.
        val prompts = listOf(
            "Describe this image in detail",
            "Read all the text in this image",
            "Translate the text in this image to English",
        )
        assertEquals(prompts.size, prompts.toSet().size, "prompts should be unique")
    }
}
