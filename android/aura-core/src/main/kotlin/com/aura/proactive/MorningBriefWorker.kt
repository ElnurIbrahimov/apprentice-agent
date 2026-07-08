package com.aura.proactive

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject

/**
 * WorkManager job that runs at ~7am local time. The actual
 * brief-building logic lives in [MorningBriefBuilder] so the
 * ProactiveRunner can also call it on demand ("fire now"
 * button on the Proactive history screen).
 *
 * Notification actions (Tell me more, Snooze 1h) are wired
 * by [MorningBriefBuilder] when it posts the notification.
 */
@HiltWorker
class MorningBriefWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val builder: MorningBriefBuilder,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result = builder.runNow()

    companion object {
        const val MORNING_BRIEF_ID = 1001
        const val UNIQUE_NAME = "morning-brief-daily"
        const val ACTION_SNOOZE = "com.aura.MORNING_BRIEF_SNOOZE"
        const val REQUEST_CODE_CHAT = 2001
        const val REQUEST_CODE_SNOOZE = 2002
    }
}
