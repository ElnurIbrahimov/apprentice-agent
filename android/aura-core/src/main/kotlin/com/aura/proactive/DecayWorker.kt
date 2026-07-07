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
 * The cost is O(n) over the memory table (typically hundreds to a few thousand
 * rows) and completes in milliseconds, so overlapping a chat is not a concern.
 */
@HiltWorker
class DecayWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val memoryStore: MemoryStore,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        return try {
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
    }

    companion object {
        const val UNIQUE_NAME = "memory-decay-periodic"
    }
}
