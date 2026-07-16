package com.aura.taste

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TasteEngineTest {

    private val signalDao = mockk<PreferenceSignalDao>(relaxed = true)
    private val profileDao = mockk<StyleProfileDao>(relaxed = true)
    private val routingDao = mockk<RoutingOutcomeDao>(relaxed = true)
    private val engine = TasteEngine(signalDao, profileDao, routingDao)

    @Test
    fun recordSignal_persists_signal_with_weight() = runTest {
        val signalSlot = slot<PreferenceSignalEntity>()
        coEvery { signalDao.upsert(capture(signalSlot)) } returns Unit

        engine.recordSignal(
            projectId = "p1",
            signalType = "accept",
            category = "text_style",
            attributes = mapOf("tone" to "formal"),
            weight = 1.0f,
        )

        assertEquals("p1", signalSlot.captured.projectId)
        assertEquals("accept", signalSlot.captured.signalType)
        assertEquals("text_style", signalSlot.captured.category)
        assertEquals(1.0f, signalSlot.captured.weight, 0.01f)
        assertTrue(signalSlot.captured.attributesJson.contains("formal"))
    }

    @Test
    fun recordReaction_positive_uses_positive_weight() = runTest {
        val signalSlot = slot<PreferenceSignalEntity>()
        coEvery { signalDao.upsert(capture(signalSlot)) } returns Unit

        engine.recordReaction("p1", "a1", "thumbs_up", true)

        assertTrue(signalSlot.captured.weight > 0)
    }

    @Test
    fun recordReaction_negative_uses_negative_weight() = runTest {
        val signalSlot = slot<PreferenceSignalEntity>()
        coEvery { signalDao.upsert(capture(signalSlot)) } returns Unit

        engine.recordReaction("p1", "a1", "thumbs_down", false)

        assertTrue(signalSlot.captured.weight < 0)
    }

    @Test
    fun recordEdit_uses_negative_weight() = runTest {
        val signalSlot = slot<PreferenceSignalEntity>()
        coEvery { signalDao.upsert(capture(signalSlot)) } returns Unit

        engine.recordEdit("p1", "a1", "text")

        assertTrue(signalSlot.captured.weight < 0)
        assertEquals("edit", signalSlot.captured.signalType)
    }

    @Test
    fun recomputeProfile_creates_profile_from_signals() = runTest {
        val signals = listOf(
            PreferenceSignalEntity(id = "s1", projectId = "p1", signalType = "accept", category = "tone", attributesJson = """{"formal":1}""", weight = 1.0f),
            PreferenceSignalEntity(id = "s2", projectId = "p1", signalType = "accept", category = "tone", attributesJson = """{"formal":1}""", weight = 1.0f),
            PreferenceSignalEntity(id = "s3", projectId = "p1", signalType = "reject", category = "tone", attributesJson = """{"casual":1}""", weight = -1.0f),
        )
        coEvery { signalDao.forProject("p1", 500) } returns signals
        coEvery { profileDao.forProject("p1") } returns null

        engine.recomputeProfile("p1")

        coVerify { profileDao.upsert(any()) }
    }

    @Test
    fun recomputeProfile_skips_when_no_signals() = runTest {
        coEvery { signalDao.global(500) } returns emptyList()

        engine.recomputeProfile("")

        coVerify(exactly = 0) { profileDao.upsert(any()) }
    }

    @Test
    fun getProfile_falls_back_to_global() = runTest {
        val globalProfile = StyleProfileEntity(id = "g1", projectId = "", attributesJson = "{}", signalCount = 5)
        coEvery { profileDao.forProject("p1") } returns null
        coEvery { profileDao.global() } returns globalProfile

        val result = engine.getProfile("p1")
        assertNotNull(result)
        assertEquals("g1", result!!.id)
    }

    @Test
    fun bestModelForRole_returns_null_when_no_data() = runTest {
        coEvery { routingDao.statsForRole("WRITER") } returns emptyList()

        assertNull(engine.bestModelForRole("WRITER"))
    }

    @Test
    fun bestModelForRole_returns_model_with_best_success_rate() = runTest {
        coEvery { routingDao.statsForRole("WRITER") } returns listOf(
            RoutingStats("model-a", count = 5, successes = 4),
            RoutingStats("model-b", count = 3, successes = 1),
        )

        val best = engine.bestModelForRole("WRITER")
        assertEquals("model-a", best)
    }

    @Test
    fun clearSignals_global_calls_deleteGlobal() = runTest {
        engine.clearSignals("")
        coVerify { signalDao.deleteGlobal() }
    }

    @Test
    fun clearSignals_project_calls_deleteForProject() = runTest {
        engine.clearSignals("p1")
        coVerify { signalDao.deleteForProject("p1") }
    }
}