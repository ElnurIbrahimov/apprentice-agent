package com.aura.agent

import android.content.Context
import com.aura.data.UserPreferences
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import java.io.ByteArrayInputStream
import java.io.File
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Locks the single-source-of-truth identity contract. */
class BrainIdentityResolutionTest {
    @get:Rule val tmp = TemporaryFolder()

    private data class Fixture(
        val brain: Brain,
        val store: IdentityStore,
        val preferences: UserPreferences,
        val legacyFile: File,
    )

    private fun fixture(
        storedIdentity: String = "",
        legacyIdentity: String? = null,
        assetText: String? = "BUNDLED SOUL.md",
    ): Fixture {
        val context = mockk<Context>(relaxed = true)
        every { context.filesDir } returns tmp.root
        if (assetText != null) {
            every { context.assets.open(Brain.IDENTITY_ASSET_FILENAME) } returns
                ByteArrayInputStream(assetText.toByteArray())
        } else {
            every { context.assets.open(Brain.IDENTITY_ASSET_FILENAME) } throws
                java.io.FileNotFoundException("SOUL.md not in assets")
        }
        val preferences = mockk<UserPreferences>(relaxed = true)
        every { preferences.customIdentity } returns flowOf(storedIdentity)
        coEvery { preferences.setCustomIdentity(any()) } returns Unit
        val legacyFile = File(tmp.root, Brain.IDENTITY_OVERRIDE_FILENAME)
        legacyIdentity?.let(legacyFile::writeText)
        val store = IdentityStore(context, preferences)
        return Fixture(
            brain = Brain(
                providerRegistry = mockk(relaxed = true),
                identityStore = store,
            ),
            store = store,
            preferences = preferences,
            legacyFile = legacyFile,
        )
    }

    @Test
    fun `stored custom identity is the live brain prompt`() = runTest {
        val fixture = fixture(storedIdentity = "USER OVERRIDE TEXT")

        assertEquals("USER OVERRIDE TEXT", fixture.brain.resolvedIdentity())
        assertEquals("USER OVERRIDE TEXT", fixture.brain.identity.first())
    }

    @Test
    fun `legacy identity file migrates once into DataStore`() = runTest {
        val fixture = fixture(legacyIdentity = "LEGACY FILE IDENTITY")

        assertEquals("LEGACY FILE IDENTITY", fixture.brain.resolvedIdentity())
        coVerify(exactly = 1) {
            fixture.preferences.setCustomIdentity("LEGACY FILE IDENTITY")
        }
        assertFalse(fixture.legacyFile.exists())
    }

    @Test
    fun `falls back to bundled asset when no custom identity exists`() = runTest {
        val fixture = fixture(assetText = "BUNDLED SOUL.md")

        assertEquals("BUNDLED SOUL.md", fixture.brain.resolvedIdentity())
        assertFalse(fixture.store.hasOverride())
    }

    @Test
    fun `falls back to hardcoded constant when bundled asset is missing`() = runTest {
        val fixture = fixture(assetText = null)

        assertEquals(Brain.IDENTITY_FALLBACK.trim(), fixture.brain.resolvedIdentity())
    }

    @Test
    fun `save and reset use the backed up custom identity preference`() = runTest {
        val fixture = fixture()

        assertTrue(fixture.store.save("  CUSTOM PERSONA  "))
        coVerify(exactly = 1) { fixture.preferences.setCustomIdentity("CUSTOM PERSONA") }

        assertTrue(fixture.store.resetToDefault())
        coVerify(exactly = 1) { fixture.preferences.setCustomIdentity("") }
    }

    @Test
    fun `identity filenames and fallback remain stable`() {
        assertEquals("SOUL.md", Brain.IDENTITY_ASSET_FILENAME)
        assertEquals("identity.md", Brain.IDENTITY_OVERRIDE_FILENAME)
        assertTrue(Brain.IDENTITY_FALLBACK.isNotBlank())
    }
}
