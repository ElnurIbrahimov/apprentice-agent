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
import kotlinx.coroutines.CancellationException

/**
 * WorkManager worker that looks up a saved hand by stable ID and executes it.
 * Legacy name input remains readable for already-enqueued v1 work.
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
    private val scheduler: HandScheduler,
) : CoroutineWorker(appContext, params) {

    companion object {
        const val KEY_HAND_ID = "hand_id"
        const val KEY_HAND_NAME = "hand_name"
        const val KEY_TRIGGER = "trigger"
        private const val TAG = "RunHandWorker"
    }

    override suspend fun doWork(): Result {
        val handId = inputData.getString(KEY_HAND_ID)
        val legacyHandName = inputData.getString(KEY_HAND_NAME)
        if (handId == null && legacyHandName == null) {
            Log.e(TAG, "Missing hand_id input")
            return Result.failure()
        }
        val trigger = inputData.getString(KEY_TRIGGER) ?: HandRunTrigger.SCHEDULE.value
        val hand = if (handId != null) repository.getById(handId) else repository.getByName(legacyHandName!!)
        if (hand == null) {
            val label = handId ?: legacyHandName.orEmpty()
            notifications.post("Aura Hand", "Hand '$label' not found")
            return Result.success()
        }

        suspend fun scheduleLatest() {
            val latest = repository.getById(hand.id)
            if (latest == null) scheduler.cancel(hand.id) else scheduler.scheduleNextAfterRun(latest)
        }

        Log.d(TAG, "Executing hand: ${hand.name}")

        val ctx = ToolContext(
            conversationId = "hand:${hand.name}",
            timeout = 120_000L,
        )

        val result = try {
            repository.run(hand, executor.get(), ctx, trigger = trigger)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.e(TAG, "Unexpected failure while executing ${hand.name}", error)
            // A malformed tool or infrastructure failure must not silently kill
            // an otherwise recurring hand. Keep the next occurrence alive.
            runCatching { scheduleLatest() }
                .onFailure { Log.e(TAG, "Could not reschedule ${hand.name}", it) }
            runCatching {
                notifications.post(
                    "Aura Hand Failed: ${hand.name}",
                    (error.message ?: "Unexpected runtime failure").take(200),
                )
            }
            return Result.success()
        }
        // One-time WorkRequests avoid periodic-work drift. Every terminal run
        // computes the next local-time occurrence; disabling/unscheduling cancels it.
        runCatching { scheduleLatest() }
            .onFailure { Log.e(TAG, "Could not reschedule ${hand.name}", it) }

        return when (result) {
            is com.aura.agent.ToolResult.Ok -> {
                notifications.post("Aura Hand: ${hand.name}", result.output.take(200))
                Result.success()
            }
            is com.aura.agent.ToolResult.Error -> {
                notifications.post("Aura Hand Failed: ${hand.name}", result.message.take(200))
                Result.success()
            }
            is com.aura.agent.ToolResult.NeedsPermission -> {
                notifications.post("Aura Hand Needs Permission", result.rationale.take(200))
                Result.success()
            }
            is com.aura.agent.ToolResult.NeedsApproval -> {
                notifications.post("Aura Hand Needs Approval", result.rationale.take(200))
                Result.success()
            }
        }
    }
}
