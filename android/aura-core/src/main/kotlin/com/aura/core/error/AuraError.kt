package com.aura.core.error

import kotlinx.serialization.Serializable

/**
 * Strongly-typed domain error model. Replacing string `(message, code)` pairs
 * lets callers render UX, decide retry, log, and report with exhaustiveness.
 */
sealed class AuraError(
    open val message: String,
    open val retryable: Boolean = false,
) {
    abstract val code: String

    /** Network or HTTP failures. */
    data class Network(
        override val message: String,
        val statusCode: Int? = null,
        override val retryable: Boolean = true,
    ) : AuraError(message, retryable) {
        override val code: String = "network"
    }

    /** Provider-side failure (auth, rate limit, content filter, model error). */
    data class Provider(
        override val message: String,
        val providerCode: String? = null,
        override val retryable: Boolean = false,
    ) : AuraError(message, retryable) {
        override val code: String = "provider"
    }

    /** Authentication / missing API key / unauthorized. */
    data class Auth(
        override val message: String,
        val providerId: String? = null,
    ) : AuraError(message, retryable = false) {
        override val code: String = "auth"
    }

    /** Rate-limited by an upstream provider. */
    data class RateLimited(
        override val message: String,
        val retryAfterMs: Long? = null,
        override val retryable: Boolean = true,
    ) : AuraError(message, retryable) {
        override val code: String = "rate_limited"
    }

    /** Tool execution failure. */
    data class Tool(
        override val message: String,
        val toolName: String? = null,
        override val retryable: Boolean = false,
    ) : AuraError(message, retryable) {
        override val code: String = "tool"
    }

    /** Tool needs a runtime permission the user hasn't granted. */
    data class NeedsPermission(
        val permission: String,
        val rationale: String,
    ) : AuraError("Permission required: $permission", retryable = true) {
        override val code: String = "needs_permission"
    }

    /** Tool needs explicit user approval because of risk/destructiveness. */
    data class NeedsApproval(
        val rationale: String,
    ) : AuraError("Approval required: $rationale", retryable = true) {
        override val code: String = "needs_approval"
    }

    /** Invalid arguments supplied by the model or caller. */
    data class BadArguments(
        override val message: String,
    ) : AuraError(message, retryable = false) {
        override val code: String = "bad_arguments"
    }

    /** Local database / storage failure. */
    data class Storage(
        override val message: String,
        override val retryable: Boolean = true,
    ) : AuraError(message, retryable) {
        override val code: String = "storage"
    }

    /** Safety guard tripped (SSRF, private IP, etc.). */
    data class Security(
        override val message: String,
    ) : AuraError(message, retryable = false) {
        override val code: String = "security"
    }

    /** Catch-all for unexpected failures. */
    data class Unknown(
        override val message: String,
        val cause: Throwable? = null,
        override val retryable: Boolean = false,
    ) : AuraError(message, retryable) {
        override val code: String = "unknown"
    }

    /** User-readable summary with an optional retry hint. */
    fun formatUserMessage(): String = buildString {
        append(message)
        if (retryable) append(" (tap to retry)")
    }
}

/**
 * Convert the serializable provider error DTO into a typed [AuraError].
 */
fun com.aura.providers.ProviderError.toAuraError(providerId: String? = null): AuraError = when (code.lowercase()) {
    "auth", "unauthorized", "invalid_api_key", "api_key" -> AuraError.Auth(message, providerId)
    "rate_limited", "rate_limit", "429", "quota" -> AuraError.RateLimited(message, retryAfterMs = null, retryable = retryable)
    "network", "timeout", "connect", "dns_error" -> AuraError.Network(message, retryable = retryable)
    "bad_arguments", "bad_args", "invalid_request" -> AuraError.BadArguments(message)
    "tool", "tool_error" -> AuraError.Tool(message, retryable = retryable)
    "storage" -> AuraError.Storage(message, retryable = retryable)
    else -> AuraError.Provider(message, providerCode = code, retryable = retryable)
}

/**
 * Convert a [Throwable] to a typed [AuraError].
 */
fun Throwable.toAuraError(): AuraError = AuraError.Unknown(
    message = message ?: "Unexpected error",
    cause = this,
)
