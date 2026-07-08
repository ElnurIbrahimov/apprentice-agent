package com.aura.ui.components

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FollowUpSuggestionsTest {

    @Test
    fun `empty assistant text returns no suggestions`() {
        assertEquals(emptyList(), FollowUpSuggestions.suggest("", isCodey = false))
    }

    @Test
    fun `long prose response offers Summarize and More detail`() {
        val long = (1..20).joinToString(" ") { "word$it" } // 140+ chars, will exceed 800 once we add filler
        val response = long + " " + (1..80).joinToString(" ") { "filler$it" }
        val out = FollowUpSuggestions.suggest(response, isCodey = false)
        assertTrue("Summarize" in out, "got $out")
        assertTrue("More detail" in out, "got $out")
    }

    @Test
    fun `codey response offers Explain the code and Show an example`() {
        val response = "Here is a function that returns one:\n```kotlin\nfun foo() = 1\n```\nIt returns one."
        val out = FollowUpSuggestions.suggest(response, isCodey = true)
        assertTrue("Explain the code" in out, "got $out")
        assertTrue("Show an example" in out, "got $out")
    }

    @Test
    fun `response ending in unclosed code block gets Continue suggestion`() {
        // The text ends with ``` (the opening of a code block
        // with no closing fence). The endAbruptly check should
        // detect this.
        val response = "Here's the function:\n```"
        val out = FollowUpSuggestions.suggest(response, isCodey = true)
        assertTrue("Continue" in out, "got $out")
    }

    @Test
    fun `list response gets Pick the best option`() {
        val response = "1. Apple\n2. Banana\n3. Cherry"
        val out = FollowUpSuggestions.suggest(response, isCodey = false)
        assertTrue("Pick the best option" in out, "got $out")
    }

    @Test
    fun `question ending offers Yes and No, something else`() {
        val response = "Do you want me to continue?"
        val out = FollowUpSuggestions.suggest(response, isCodey = false)
        assertTrue("Yes" in out, "got $out")
        assertTrue("No, something else" in out, "got $out")
    }

    @Test
    fun `suggestions are capped at 3`() {
        // Force every heuristic to fire.
        val response = (1..100).joinToString(" ") { "word" } + "\n```\nDo you want me to continue?"
        val out = FollowUpSuggestions.suggest(response, isCodey = true)
        assertTrue(out.size <= 3, "expected <= 3 suggestions, got ${out.size}: $out")
    }

    @Test
    fun `suggestions are unique`() {
        val response = (1..100).joinToString(" ") { "word" }
        val out = FollowUpSuggestions.suggest(response, isCodey = false)
        assertEquals(out.size, out.toSet().size, "expected unique suggestions, got $out")
    }

    @Test
    fun `short response offers only More detail`() {
        val out = FollowUpSuggestions.suggest("here is a short answer about the topic", isCodey = false)
        // 41 chars is in the > 300 bucket only if we count … but
        // 41 chars < 300 so should be empty.
        assertEquals(emptyList(), out, "got $out")
    }
}
