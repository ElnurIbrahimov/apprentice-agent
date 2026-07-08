package com.aura.proactive

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.aura.memory.MemoryStore
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject

/**
 * WorkManager periodic worker that runs a single decay pass over all memories.
 * Scheduled every 6 hours via [ProactiveScheduler.scheduleDecay].
 *
 * The actual work is one call to [MemoryStore.runDecayPass] — no need
 * to extract a builder class. The ProactiveRunner uses the same
 * MemoryStore directly when the user taps "fire decay pass now".
 */
@HiltWorker
class DecayWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val memoryStore: MemoryStore,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result = runNow()

    /**
     * Run a single decay pass. Returns a [Result] the same way
     * [doWork] does so the ProactiveRunner can surface
     * success / retry / failure to the UI.
     */
    suspend fun runNow(): Result = try {
        memoryStore.runDecayPass()
        Result.success()
    } catch (e: kotlinx.coroutines.CancellationException) {
        throw e
    } catch (e: Exception) {
        try {
            android.util.Log.w("DecayWorker", "decay pass failed: ${e.message}")
        } catch (_: RuntimeException) {}
        Result.retry()
    }

    companion object {
        const val UNIQUE_NAME = "memory-decay-periodic"
    }
}
