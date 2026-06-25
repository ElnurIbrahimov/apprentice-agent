package com.aura.tools

import android.content.Context
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.suspendCancellableCoroutine
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.resume
/**
 * Prompt the user for biometric authentication (face/fingerprint/credential).
 * Useful as a high-risk tool gate (e.g. before making a purchase, sending an email,
 * or performing other destructive actions). Mirrors aura/security/audit_chain.py
 * + the BiometricPrompt AndroidX flow.
 * Risk: WRITE_LOCAL (UI prompt).
 */
@Singleton
class BiometricPromptTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "biometric_prompt",
        description = "Prompt the user for biometric authentication. Returns 'authenticated' on success, error message on failure or cancel. Use as a safety gate before sensitive actions.",
        parameters = ToolParameters(
            properties = mapOf(
                "title" to ToolProperty(type = "string", description = "Prompt title (e.g. 'Confirm action')"),
                "subtitle" to ToolProperty(type = "string", description = "Optional subtitle"),
                "reason" to ToolProperty(type = "string", description = "Why is this prompt being shown (shown to user)"),
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
        execute = { call, ctx ->
            val title = call.arguments["title"] as? String ?: return@Tool ToolResult.Error("missing 'title'", "bad_args")
            val subtitle = call.arguments["subtitle"] as? String
            val reason = call.arguments["reason"] as? String

            val activity = (context as? FragmentActivity)
                ?: return@Tool ToolResult.Error("biometric_prompt requires FragmentActivity context", "context_error")

            val mgr = BiometricManager.from(context)
            val canAuth = mgr.canAuthenticate(BiometricManager.Authenticators.BIOMETRIC_WEAK or BiometricManager.Authenticators.DEVICE_CREDENTIAL)
            if (canAuth != BiometricManager.BIOMETRIC_SUCCESS) {
                return@Tool ToolResult.Error("biometric unavailable: ${canAuth}", "no_biometric")
            }

            try {
                val result = promptSync(activity, title, subtitle, reason)
                when (result) {
                    is PromptResult.Success -> ToolResult.Ok("authenticated")
                    is PromptResult.Error -> ToolResult.Error(result.message, result.code)
                    is PromptResult.UserCancel -> ToolResult.Error("user cancelled", "cancelled")
                }
            } catch (e: Exception) {
                ToolResult.Error("biometric failed: ${e.message}", "exception")
            }
        },
    )

    private sealed class PromptResult {
        data object Success : PromptResult()
        data class Error(val code: String, val message: String) : PromptResult()
        data object UserCancel : PromptResult()
    }

    private suspend fun promptSync(
        activity: FragmentActivity,
        title: String,
        subtitle: String?,
        reason: String?,
    ): PromptResult = suspendCancellableCoroutine { cont ->
        val executor = ContextCompat.getMainExecutor(context)
        val prompt = BiometricPrompt(activity, executor, object : BiometricPrompt.AuthenticationCallback() {
            override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                if (cont.isActive) cont.resume(PromptResult.Success)
            }
            override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                if (cont.isActive) {
                    val code = when (errorCode) {
                        BiometricPrompt.ERROR_USER_CANCELED, BiometricPrompt.ERROR_NEGATIVE_BUTTON, BiometricPrompt.ERROR_CANCELED -> {
                            cont.resume(PromptResult.UserCancel)
                            return
                        }
                        else -> "auth_error"
                    }
                    cont.resume(PromptResult.Error(code, errString.toString()))
                }
            }
            override fun onAuthenticationFailed() { /* user can retry, don't resume */ }
        })
        val info = BiometricPrompt.PromptInfo.Builder()
            .setTitle(title)
            .apply { if (subtitle != null) setSubtitle(subtitle) }
            .apply { if (reason != null) setDescription(reason) }
            .setAllowedAuthenticators(BiometricManager.Authenticators.BIOMETRIC_WEAK or BiometricManager.Authenticators.DEVICE_CREDENTIAL)
            .build()
        prompt.authenticate(info)
    }
}
