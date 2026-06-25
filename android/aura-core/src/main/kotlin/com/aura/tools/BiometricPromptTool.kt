package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Marks a sensitive operation as needing biometric confirmation by the user.
 *
 * The previous implementation tried to call BiometricPrompt from the application
 * context, but BiometricPrompt requires a FragmentActivity — and the only
 * context the tool has access to is the application context. The cast
 * `context as? FragmentActivity` was always null, so the tool always
 * returned an error and was effectively dead code.
 *
 * Rather than restructure the agent loop to thread an activity reference
 * through the tool system, we now treat biometric confirmation as a
 * first-class approval signal: the tool returns NeedsApproval with a
 * deterministic rationale. The activity's chat loop is expected to
 * surface this as a confirmation prompt in a future revision; for v1
 * the model can explain to the user that the action requires explicit
 * confirmation outside the model flow.
 *
 * Risk: WRITE_LOCAL (security gate).
 */
@Singleton
class BiometricPromptTool @Inject constructor() {
    fun definition() = ToolDefinition(
        name = "biometric_prompt",
        description = "Mark a sensitive operation as requiring biometric confirmation. " +
            "Returns NeedsApproval. The user must explicitly confirm outside the model flow.",
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
            val reason = (call.arguments["reason"] as? String)?.let { ": $it" } ?: ""
            ToolResult.NeedsApproval("Biometric confirmation required for: $title$reason")
        },
    )
}
