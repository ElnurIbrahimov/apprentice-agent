package com.aura.providers

import kotlinx.coroutines.CancellationException

/**
 * Typed exceptions raised by [Provider.listModels] when model catalog
 * discovery fails. Every subtype is a proper [Exception] so callers can
 * catch them with standard Kotlin `try/catch` without relying on
 * unstructured string matching.
 *
 * Messages are **sanitised** — they never include raw API keys, secret
 * response bodies, or internal implementation details. Use the typed
 * properties (e.g. [statusCode]) for structured logging/reporting.
 */
sealed class ProviderCatalogException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause) {

    /**
     * 401 Unauthorised — the API key was missing, invalid, or expired.
     * The raw response body is **never** included in the message to
     * avoid leaking secrets through error UX or logs.
     */
    class AuthenticationException
        internal constructor(
            message: String = "Authentication failed. Check your API key.",
            cause: Throwable? = null,
        ) : ProviderCatalogException(message, cause)

    /**
     * 429 Too Many Requests — the upstream provider is rate-limiting us.
     * [retryAfterMs] is the provider's suggested wait (may be null).
     */
    class RateLimitedException
        internal constructor(
            message: String = "Rate limited by the provider. Try again later.",
            val retryAfterMs: Long? = null,
            cause: Throwable? = null,
        ) : ProviderCatalogException(message, cause)

    /**
     * The /models endpoint returned a response that could not be parsed
     * or had an unexpected structure. The sanitised description hints at
     * what went wrong without echoing the raw payload.
     */
    class MalformedResponseException
        internal constructor(
            message: String,
            cause: Throwable? = null,
        ) : ProviderCatalogException(message, cause)

    /**
     * The /models endpoint responded successfully but returned zero
     * models. This is distinct from a parse failure — the JSON was
     * valid, it just had no entries.
     */
    class EmptyCatalogException
        internal constructor(
            message: String = "Provider returned an empty model catalog.",
            cause: Throwable? = null,
        ) : ProviderCatalogException(message, cause)

    /**
     * A network, I/O, or transport-level failure (timeout, DNS, connection
     * reset). [statusCode] is set when we have an HTTP status (e.g. 5xx).
     */
    class NetworkException
        internal constructor(
            message: String = "Network error while fetching models.",
            val statusCode: Int? = null,
            cause: Throwable? = null,
        ) : ProviderCatalogException(message, cause)

    /**
     * NOTE: [kotlinx.coroutines.CancellationException] must always be
     * rethrown as-is — it is NOT caught or wrapped by any subtype
     * here. Each [Provider.listModels] implementation is responsible
     * for the `catch (e: CancellationException) { throw e }` pattern.
     *
     * Example:
     * ```
     * override suspend fun listModels(): List<String> {
     *     return try { … } catch (e: CancellationException) { throw e }
     *                   catch (e: IOException) { throw NetworkException(cause = e) }
     * }
     * ```
     */
}
