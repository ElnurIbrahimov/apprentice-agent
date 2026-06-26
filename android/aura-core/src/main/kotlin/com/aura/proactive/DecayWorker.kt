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
        runCatching { memoryStore.runDecayPass() }
        return Result.success()
    }

    companion object {
        const val UNIQUE_NAME = "memory-decay-periodic"
    }
}
