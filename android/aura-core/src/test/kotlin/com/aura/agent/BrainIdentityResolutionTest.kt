package com.aura.agent

import android.content.Context
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Lock the [Brain.identity] file-resolution rules. The brain
 * reads identity from one of three sources in priority order:
 *
 *   1. User override at `filesDir/identity.md` (if present)
 *   2. Bundled asset at `assets/SOUL.md` (if present)
 *   3. Hardcoded fallback [Brain.IDENTITY_FALLBACK]
 *
 * Tests use a real TemporaryFolder (not a mock) for filesDir so
 * Java's [File] ctor doesn't NPE on a null parent path. The
 * `Context.assets` stream is mocked with mockk.
 */
class BrainIdentityResolutionTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun brainWith(
        overrideFile: File? = null,
        assetText: String? = "BUNDLED SOUL.md",
    ): Brain {
        val ctx = mockk<Context>(relaxed = true)
        every { ctx.filesDir } returns tmp.root
        if (assetText != null) {
            every { ctx.assets.open(Brain.IDENTITY_ASSET_FILENAME) } returns
                ByteArrayInputStream(assetText.toByteArray())
        } else {
            every { ctx.assets.open(Brain.IDENTITY_ASSET_FILENAME) } throws
                java.io.FileNotFoundException("SOUL.md not in assets")
        }
        // Pre-create the override file if requested
        if (overrideFile != null) {
            val real = File(tmp.root, Brain.IDENTITY_OVERRIDE_FILENAME)
            real.writeText(overrideFile.readText())
        }
        return Brain(
            context = ctx,
            providerRegistry = mockk(relaxed = true),
            userPreferences = mockk(relaxed = true),
        )
    }

    @Test
    fun `prefers user override over bundled asset`() = runTest {
        val override = File(tmp.root, "staging-override.txt")
        override.writeText("USER OVERRIDE TEXT")
        val brain = brainWith(overrideFile = override)
        val text = brain.identity.first()
        assertEquals("USER OVERRIDE TEXT", text)
    }

    @Test
    fun `falls back to bundled asset when no override`() = runTest {
        val brain = brainWith(overrideFile = null, assetText = "BUNDLED SOUL.md")
        val text = brain.identity.first()
        assertEquals("BUNDLED SOUL.md", text)
    }

    @Test
    fun `falls back to hardcoded constant when both files missing`() = runTest {
        val brain = brainWith(overrideFile = null, assetText = null)
        val text = brain.identity.first()
        assertEquals(Brain.IDENTITY_FALLBACK.trim(), text)
    }

    @Test
    fun `empty override file falls through to bundled asset`() = runTest {
        val override = File(tmp.root, "staging-override.txt")
        override.writeText("")
        val brain = brainWith(overrideFile = override, assetText = "BUNDLED SOUL.md")
        val text = brain.identity.first()
        // A 0-byte override is treated as "no override" — use the asset.
        assertEquals("BUNDLED SOUL.md", text)
    }

    @Test
    fun `asset filename is SOUL_md and override filename is identity_md`() {
        // Lock the filenames so a refactor can't accidentally
        // rename the user file (would silently lose user edits).
        assertEquals("SOUL.md", Brain.IDENTITY_ASSET_FILENAME)
        assertEquals("identity.md", Brain.IDENTITY_OVERRIDE_FILENAME)
        assertTrue(Brain.IDENTITY_FALLBACK.isNotBlank())
    }
}
