package com.aura.profile

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests for [UserProfileStore] — load default, merge updates, system
 * prompt generation, and DAO round-trip. The profile store is the
 * single source of truth for user facts extracted by the agent loop;
 * a regression here means the model forgets who it's talking to.
 */
class UserProfileStoreTest {

    private val dispatcher = StandardTestDispatcher()

    @Test
    fun `default profile has null name and empty collections`() {
        val profile = UserProfile()
        assertNull(profile.name)
        assertTrue(profile.traits.isEmpty())
        assertTrue(profile.preferences.isEmpty())
        assertTrue(profile.facts.isEmpty())
    }

    @Test
    fun `update merges name without overwriting existing traits`() = runTest(dispatcher) {
        val dao = mockk<UserProfileDao>(relaxed = true)
        coEvery { dao.get() } returns null
        val store = UserProfileStore(dao)
        advanceUntilIdle()

        store.update(name = "Elnur")
        advanceUntilIdle()

        assertEquals("Elnur", store.profile.value.name)
        assertTrue(store.profile.value.traits.isEmpty())

        // Update traits — name should persist
        store.update(traits = listOf("developer", "night owl"))
        advanceUntilIdle()

        assertEquals("Elnur", store.profile.value.name)
        assertEquals(listOf("developer", "night owl"), store.profile.value.traits)
    }

    @Test
    fun `update merges preferences additively not replacing`() = runTest(dispatcher) {
        val dao = mockk<UserProfileDao>(relaxed = true)
        coEvery { dao.get() } returns null
        val store = UserProfileStore(dao)
        advanceUntilIdle()

        store.update(preferences = mapOf("theme" to "dark"))
        advanceUntilIdle()
        assertEquals("dark", store.profile.value.preferences["theme"])

        // Add a different preference — the first one should persist
        store.update(preferences = mapOf("language" to "en"))
        advanceUntilIdle()
        assertEquals("dark", store.profile.value.preferences["theme"])
        assertEquals("en", store.profile.value.preferences["language"])
    }

    @Test
    fun `explicit update replaces facts list for profile editing`() = runTest(dispatcher) {
        val dao = mockk<UserProfileDao>(relaxed = true)
        coEvery { dao.get() } returns null
        val store = UserProfileStore(dao)
        advanceUntilIdle()

        store.update(facts = listOf("likes tea", "has a cat"))
        advanceUntilIdle()
        assertEquals(listOf("likes tea", "has a cat"), store.profile.value.facts)

        // Explicit profile editing retains replacement semantics.
        store.update(facts = listOf("likes coffee"))
        advanceUntilIdle()
        assertEquals(listOf("likes coffee"), store.profile.value.facts)
    }

    @Test
    fun `mergeFacts preserves prior facts and deduplicates case-insensitively`() = runTest(dispatcher) {
        val dao = mockk<UserProfileDao>(relaxed = true)
        coEvery { dao.get() } returns null
        val store = UserProfileStore(dao)

        store.mergeFacts(listOf("Lives in Baku"))
        store.mergeFacts(listOf("Prefers tea", "lives in baku", "  "))

        assertEquals(listOf("Lives in Baku", "Prefers tea"), store.profile.value.facts)
    }

    @Test
    fun `update waits for initial load and preserves persisted fields`() = runTest(dispatcher) {
        val dao = mockk<UserProfileDao>(relaxed = true)
        coEvery { dao.get() } coAnswers {
            kotlinx.coroutines.delay(50)
            UserProfileEntity(name = "Persisted", traitsJson = "[\"curious\"]")
        }
        val store = UserProfileStore(dao)

        store.update(preferences = mapOf("theme" to "dark"))

        assertEquals("Persisted", store.profile.value.name)
        assertEquals(listOf("curious"), store.profile.value.traits)
        assertEquals("dark", store.profile.value.preferences["theme"])
    }

    @Test
    fun `update persists to DAO via upsert`() = runTest(dispatcher) {
        val dao = mockk<UserProfileDao>(relaxed = true)
        coEvery { dao.get() } returns null
        val store = UserProfileStore(dao)
        advanceUntilIdle()

        store.update(name = "Test", traits = listOf("curious"))
        advanceUntilIdle()

        coVerify { dao.upsert(any()) }
    }

    @Test
    fun `loads persisted profile from DAO on init`() = runTest(dispatcher) {
        val entity = UserProfileEntity(
            id = 1,
            name = "Loaded",
            traitsJson = """["smart"]""",
            preferencesJson = """{"editor":"vim"}""",
            factsJson = """["likes sushi"]""",
            lastUpdated = 123L,
        )
        val dao = mockk<UserProfileDao>(relaxed = true)
        coEvery { dao.get() } returns entity
        val store = UserProfileStore(dao)
        store.awaitLoaded()

        assertEquals("Loaded", store.profile.value.name)
        assertEquals(listOf("smart"), store.profile.value.traits)
        assertEquals("vim", store.profile.value.preferences["editor"])
        assertEquals(listOf("likes sushi"), store.profile.value.facts)
    }

    @Test
    fun `toSystemPrompt includes all fields when populated`() {
        val profile = UserProfile(
            name = "Alice",
            traits = listOf("developer", "curious"),
            preferences = mapOf("theme" to "dark", "editor" to "vim"),
            facts = listOf("owns a dog", "speaks Japanese"),
        )
        val prompt = profile.toSystemPrompt()
        assertTrue(prompt.contains("Name: Alice"))
        assertTrue(prompt.contains("Traits: developer, curious"))
        assertTrue(prompt.contains("Prefers theme: dark"))
        assertTrue(prompt.contains("Prefers editor: vim"))
        assertTrue(prompt.contains("Facts about the user:"))
        assertTrue(prompt.contains("- owns a dog"))
        assertTrue(prompt.contains("- speaks Japanese"))
    }

    @Test
    fun `toSystemPrompt returns minimal header for empty profile`() {
        val profile = UserProfile()
        val prompt = profile.toSystemPrompt()
        assertTrue(prompt.contains("## User Profile"))
        assertFalse(prompt.contains("Name:"))
        assertFalse(prompt.contains("Traits:"))
        assertFalse(prompt.contains("Prefers"))
        assertFalse(prompt.contains("Facts"))
    }

    @Test
    fun `entity round-trip preserves all fields`() {
        val original = UserProfile(
            name = "Roundtrip",
            traits = listOf("a", "b"),
            preferences = mapOf("x" to "1"),
            facts = listOf("fact1"),
        )
        val entity = UserProfile.toEntity(original)
        val restored = UserProfile.fromEntity(entity)
        assertEquals(original.name, restored.name)
        assertEquals(original.traits, restored.traits)
        assertEquals(original.preferences, restored.preferences)
        assertEquals(original.facts, restored.facts)
    }

    @Test
    fun `fromEntity handles corrupt JSON gracefully`() {
        val entity = UserProfileEntity(
            id = 1,
            name = "Corrupt",
            traitsJson = "not valid json",
            preferencesJson = "also not json",
            factsJson = "definitely not json",
        )
        val profile = UserProfile.fromEntity(entity)
        assertEquals("Corrupt", profile.name)
        assertTrue(profile.traits.isEmpty())
        assertTrue(profile.preferences.isEmpty())
        assertTrue(profile.facts.isEmpty())
    }

    @Test
    fun `getSystemPrompt reflects live profile state`() = runTest(dispatcher) {
        val dao = mockk<UserProfileDao>(relaxed = true)
        coEvery { dao.get() } returns null
        val store = UserProfileStore(dao)
        advanceUntilIdle()

        // Before update — minimal prompt
        assertFalse(store.getSystemPrompt().contains("Name:"))

        store.update(name = "Live")
        advanceUntilIdle()

        assertTrue(store.getSystemPrompt().contains("Name: Live"))
    }
}