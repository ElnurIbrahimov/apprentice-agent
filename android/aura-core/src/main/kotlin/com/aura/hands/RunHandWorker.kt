package com.aura.hands

import android.content.Context
import android.util.Log
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.tools.NotificationsTool
import dagger.Lazy
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject

/**
 * WorkManager worker that looks up a saved hand by name and executes it.
 * Shows a notification when done or on failure.
 *
 * Lazy<ToolExecutor> is used to break the Dagger cycle with ToolRegistry.
 */
@HiltWorker
class RunHandWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val repository: HandRepository,
    private val executor: Lazy<ToolExecutor>,
    private val notifications: NotificationsTool,
) : CoroutineWorker(appContext, params) {

    companion object {
        const val KEY_HAND_NAME = "hand_name"
        private const val TAG = "RunHandWorker"
    }

    override suspend fun doWork(): Result {
        val handName = inputData.getString(KEY_HAND_NAME) ?: run {
            Log.e(TAG, "Missing hand_name input")
            return Result.failure()
        }

        Log.d(TAG, "Executing hand: $handName")

        val hand = repository.getByName(handName)
        if (hand == null) {
            notifications.post("Aura Hand", "Hand '$handName' not found")
            return Result.failure()
        }

        val ctx = ToolContext(
            conversationId = "hand:${hand.name}",
            timeout = 120_000L,
        )

        val result = repository.run(hand, executor.get(), ctx)

        return when (result) {
            is com.aura.agent.ToolResult.Ok -> {
                notifications.post("Aura Hand: ${hand.name}", result.output.take(200))
                Result.success()
            }
            is com.aura.agent.ToolResult.Error -> {
                notifications.post("Aura Hand Failed: ${hand.name}", result.message.take(200))
                Result.failure()
            }
            is com.aura.agent.ToolResult.NeedsPermission -> {
                notifications.post("Aura Hand Needs Permission", result.rationale.take(200))
                Result.failure()
            }
            is com.aura.agent.ToolResult.NeedsApproval -> {
                notifications.post("Aura Hand Needs Approval", result.rationale.take(200))
                Result.failure()
            }
        }
    }
}
