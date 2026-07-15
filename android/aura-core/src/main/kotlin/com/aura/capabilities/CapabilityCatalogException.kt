package com.aura.capabilities

/**
 * Error types capability providers raise. Mirrors [com.aura.providers.ProviderCatalogException]
 * but lives in the capabilities package so chat-completions code doesn't depend on capability
 * code (and vice versa).
 */
sealed class CapabilityCatalogException(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class AuthenticationException : CapabilityCatalogException("Provider rejected the API key.")
    class RateLimitedException(val retryAfterMs: Long? = null) :
        CapabilityCatalogException("Provider rate limited the request.")
    class NetworkException(message: String, val statusCode: Int? = null, cause: Throwable? = null) :
        CapabilityCatalogException(message, cause)
    class MalformedResponseException(message: String, cause: Throwable? = null) :
        CapabilityCatalogException(message, cause)
    class EmptyCatalogException : CapabilityCatalogException("Provider returned no models.")
    class CancelledException : CapabilityCatalogException("Capability request was cancelled.")
    class MissingAssetException(message: String) : CapabilityCatalogException(message)
}
