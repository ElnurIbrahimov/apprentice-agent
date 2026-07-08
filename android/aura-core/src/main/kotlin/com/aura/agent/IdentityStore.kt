package com.aura.agent

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Reads and writes the user-editable identity file.
 *
 * The user's edits live at `filesDir/identity.md`. The bundled
 * asset at `assets/SOUL.md` is the source of "default" — used
 * by [resetToDefault] and by [Brain] as the fallback.
 *
 * All file I/O is dispatched to [Dispatchers.IO] so the UI
 * thread is never blocked, even though the files are small
 * (a few KB at most).
 */
@Singleton
class IdentityStore @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    /** Path of the user-editable file. Created on first write. */
    val overrideFile: File
        get() = File(context.filesDir, Brain.IDENTITY_OVERRIDE_FILENAME)

    /**
     * Returns the user's current identity text. If the user
     * override file exists and is non-empty, that's returned.
     * Otherwise the bundled asset is read. Synchronous I/O — fine
     * because the file is a few KB; callers should call this from
     * a coroutine on Dispatchers.IO if running on the main thread.
     */
    fun readCurrent(): String {
        val f = overrideFile
        if (f.exists() && f.length() > 0) {
            return f.readText().trim()
        }
        return readAsset()
    }

    /**
     * Returns the bundled asset text (the "default"). This is what
     * [readCurrent] falls back to, and what [resetToDefault] writes
     * to the override file.
     */
    fun readAsset(): String = context.assets
        .open(Brain.IDENTITY_ASSET_FILENAME)
        .bufferedReader()
        .use { it.readText() }
        .trim()

    /**
     * Persist the user's edits to `filesDir/identity.md`. Returns
     * true if the file was written successfully, false otherwise
     * (e.g. disk full, permission denied).
     */
    suspend fun save(text: String): Boolean = withContext(Dispatchers.IO) {
        try {
            overrideFile.writeText(text.trim())
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Delete the user override file, falling back to the bundled
     * asset on next read. Idempotent: deleting a non-existent
     * file is a no-op.
     */
    suspend fun resetToDefault(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (overrideFile.exists()) {
                overrideFile.delete()
            }
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * True if the user has a non-empty override file in place.
     * Settings shows a "Customized" badge when this returns true.
     */
    fun hasOverride(): Boolean =
        overrideFile.exists() && overrideFile.length() > 0
}
