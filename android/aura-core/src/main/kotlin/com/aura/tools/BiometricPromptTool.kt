package com.aura.tools

import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.fragment.app.FragmentActivity
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import com.aura.security.BiometricActivityHolder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.util.concurrent.Executors
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Prompts the user for biometric authentication (fingerprint, face, etc.)
 * using the real [BiometricPrompt] API.
 *
 * The tool reads the current foreground [FragmentActivity] from
 * [BiometricActivityHolder].  If no activity is attached (e.g. the app is
 * in the background), it returns [ToolResult.NeedsApproval] so the model
 * can explain the situation to the user.
 *
 * The actual BiometricPrompt call is bridged to a coroutine via
 * [BiometricAuthHandler] + [withTimeout] (60 s).
 *
 * Risk: WRITE_LOCAL (security gate).
 */
@Singleton
class BiometricPromptTool @Inject constructor(
    private val holder: BiometricActivityHolder,
) {
    fun definition() = ToolDefinition(
        name = "biometric_prompt",
        description = "Prompt the user for biometric authentication (fingerprint, face, etc.) and return the result.",
        parameters = ToolParameters(
            properties = mapOf(
                "title" to ToolProperty(type = "string", description = "Prompt title (e.g. 'Confirm action')"),
                "subtitle" to ToolProperty(type = "string", description = "Optional subtitle"),
                "reason" to ToolProperty(type = "string", description = "Why this prompt is being shown (shown to user)"),
            ),
            required = listOf("title"),
        ),
    )

    val tool = Tool(
        name = "biometric_prompt",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        requiredPermissions = listOf("android.permission.USE_BIOMETRIC"),
        parameters = definition().parameters,
        execute = { call, _ ->
            // --- Guard: no foreground activity ---------------------------------
            val activity: FragmentActivity = holder.activity
                ?: return@Tool ToolResult.NeedsApproval("no foreground activity to display the biometric prompt")

            // --- Parse arguments ------------------------------------------------
            val title = call.arguments["title"] as? String
                ?: return@Tool ToolResult.Error("missing 'title'", "bad_args")
            val subtitle = call.arguments["subtitle"] as? String
            val reason = call.arguments["reason"] as? String

            // --- Wire up the callback bridge ------------------------------------
            val handler = BiometricAuthHandler()
            val executor = Executors.newSingleThreadExecutor()

            // BiometricPrompt must be created on the main thread
            withContext(Dispatchers.Main) {
                val callback = object : BiometricPrompt.AuthenticationCallback() {
                    override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                        handler.onAuthenticated()
                    }

                    override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                        handler.onError(errorCode, errString.toString())
                    }
                }

                val prompt = BiometricPrompt(activity, executor, callback)
                val promptInfo = BiometricPrompt.PromptInfo.Builder()
                    .setTitle(title)
                    .apply { if (subtitle != null) setSubtitle(subtitle) }
                    .apply { if (reason != null) setDescription(reason) }
                    .setAllowedAuthenticators(BiometricManager.Authenticators.BIOMETRIC_WEAK)
                    .build()
                prompt.authenticate(promptInfo)
            }

            // --- Await the result (with 60 s timeout) --------------------------
            val authResult = try {
                withTimeout(60_000L) { handler.result.await() }
            } catch (e: TimeoutCancellationException) {
                executor.shutdownNow()
                return@Tool ToolResult.Error("timeout", "auth_timeout")
            }
            executor.shutdownNow()

            // --- Map to ToolResult ---------------------------------------------
            when {
                authResult.success -> ToolResult.Ok("authenticated")
                authResult.errorCode == BiometricPrompt.ERROR_USER_CANCELED ->
                    ToolResult.Error("user cancelled", "user_cancelled")
                else -> ToolResult.Error(
                    "Authentication failed: ${authResult.errorMessage}",
                    "auth_failed:${authResult.errorCode}",
                )
            }
        },
    )
}
