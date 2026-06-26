package com.aura.tools

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred

/**
 * Result of a biometric authentication attempt, produced by [BiometricAuthHandler].
 * Pure Kotlin — no Android dependencies, testable on JVM.
 *
 * @property success `true` when the user was authenticated successfully.
 * @property errorCode The Android [BiometricPrompt][androidx.biometric.BiometricPrompt]
 *   error code, or `null` on success.
 * @property errorMessage Human-readable error string, or `null` on success.
 */
data class BiometricAuthResult(
    val success: Boolean,
    val errorCode: Int? = null,
    val errorMessage: String? = null,
)

/**
 * Pure-Kotlin bridge between [BiometricPrompt.AuthenticationCallback]
 * and a [CompletableDeferred].  Instantiate this handler, pass its
 * [result] deferred to a waiting coroutine, then call the lifecycle
 * methods from the callback.
 *
 * ```kotlin
 * val handler = BiometricAuthHandler()
 * val callback = object : BiometricPrompt.AuthenticationCallback() {
 *     override fun onAuthenticationSucceeded(r) { handler.onAuthenticated() }
 *     override fun onAuthenticationError(c, s)  { handler.onError(c, s.toString()) }
 * }
 * // … later …
 * val outcome = handler.result.await()
 * ```
 */
class BiometricAuthHandler {
    private val deferred = CompletableDeferred<BiometricAuthResult>()

    /** Deferred result that completes when [onAuthenticated] or [onError] is called. */
    val result: Deferred<BiometricAuthResult> = deferred

    /** Call from [BiometricPrompt.AuthenticationCallback.onAuthenticationSucceeded]. */
    fun onAuthenticated() {
        deferred.complete(BiometricAuthResult(success = true))
    }

    /**
     * Call from [BiometricPrompt.AuthenticationCallback.onAuthenticationError].
     *
     * @param code  Android biometric error code (e.g. 13 = user cancelled).
     * @param message Human-readable error description.
     */
    fun onError(code: Int, message: String) {
        deferred.complete(BiometricAuthResult(success = false, errorCode = code, errorMessage = message))
    }
}
