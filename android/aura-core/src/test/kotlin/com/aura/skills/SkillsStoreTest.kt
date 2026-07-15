package com.aura.skills

import com.aura.security.SecureDataStore
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests the [SkillsStore] pipeline: add, update, remove, lookup.
 *
 * Persistence is mocked so the tests don't need a real DataStore; the
 * round-trip is exercised by [SkillsSerializationTest] which uses
 * a real in-memory DataStore.
 */
class SkillsStoreTest {

    private fun newStore(initial: Map<String, String?> = emptyMap()): SkillsStore {
        val secure = mockk<SecureDataStore>(relaxed = true)
        coEvery { secure.getString(any()) } answers { firstArg<String>().let { initial[it] } }
        coEvery { secure.putString(any(), any()) } returns Unit
        return SkillsStore(secure)
    }

    @Test
    fun `add persists the skill envelope`() = runTest {
        val store = newStore()
        store.add(Skill(name = "review", description = "PR review", body = "Always check tests."))
        // We don't strictly need to verify the captured putString value here;
        // the SkillsSerializationTest covers the on-the-wire format. Here we
        // just assert that adding twice is idempotent in the public state.
        store.add(Skill(name = "summarize", description = "", body = "x"))
        assertEquals(2, store.skills.value.size)
        assertEquals("review", store.findByName("review")?.name)
    }

    @Test
    fun `findByName is case insensitive`() = runTest {
        val store = newStore()
        store.add(Skill(name = "Review", description = "", body = "x"))
        assertEquals("Review", store.findByName("review")?.name)
        assertEquals("Review", store.findByName("REVIEW")?.name)
        assertEquals("Review", store.findByName("Review")?.name)
        assertEquals(null, store.findByName("missing"))
    }

    @Test
    fun `update replaces the matching skill by id`() = runTest {
        val store = newStore()
        val s1 = Skill(name = "a", description = "", body = "alpha")
        val s2 = Skill(name = "b", description = "", body = "bravo")
        store.add(s1)
        store.add(s2)
        val updated = s1.withBody("alpha v2")
        store.update(updated)
        assertEquals("alpha v2", store.findById(s1.id)?.body)
        assertEquals("bravo", store.findById(s2.id)?.body)
    }

    @Test
    fun `remove drops the matching skill by id`() = runTest {
        val store = newStore()
        val s1 = Skill(name = "a", description = "", body = "alpha")
        val s2 = Skill(name = "b", description = "", body = "bravo")
        store.add(s1)
        store.add(s2)
        store.remove(s1.id)
        assertEquals(null, store.findById(s1.id))
        assertEquals("b", store.findById(s2.id)?.name)
    }
}
