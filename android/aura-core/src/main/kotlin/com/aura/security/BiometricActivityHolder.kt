package com.aura.security

import androidx.fragment.app.FragmentActivity
import java.lang.ref.WeakReference
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Holds a [WeakReference] to the current foreground [FragmentActivity] so
 * that tools (e.g. [BiometricPrompt][androidx.biometric.BiometricPrompt]) that
 * need a FragmentActivity reference can obtain it without the agent loop
 * having to thread the activity through every tool invocation.
 *
 * The activity sets itself in [FragmentActivity.onCreate] via
 * `holder.activity = this` and clears in [FragmentActivity.onDestroy] via
 * `holder.activity = null`.
 */
@Singleton
class BiometricActivityHolder @Inject constructor() {

    @Volatile
    private var activityRef: WeakReference<FragmentActivity>? = null

    /**
     * The current foreground [FragmentActivity], or `null` if no activity is alive.
     */
    var activity: FragmentActivity?
        get() = activityRef?.get()
        set(value) {
            activityRef = if (value != null) WeakReference(value) else null
        }
}
