package com.aura.capabilities

import com.aura.providers.ProviderKeys
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Selects a configured [CapabilityProvider] for a [CapabilityKind] and
 * optional operation. Replaces the old "first configured in Hilt map
 * order" approach with explicit user preference, health, and fallback
 * logic.
 *
 * Resolution order:
 * 1. User-explicit preference for this kind (if set and still configured)
 * 2. First configured provider that supports the requested operation
 * 3. First configured provider for this kind (operation-agnostic)
 * 4. null (no provider available — caller must handle honestly)
 */
@Singleton
class CapabilityRouter @Inject constructor(
    private val registry: CapabilityRegistry,
    private val providerKeys: ProviderKeys,
) {
    /**
     * Resolve a provider for [kind]. Returns null if no provider is
     * configured for this kind.
     */
    fun resolve(kind: CapabilityKind): CapabilityProvider? =
        registry.forKind(kind)

    /**
     * Resolve a provider for [kind] that supports [operation].
     * If no provider declares support for the specific operation,
     * falls back to the first configured provider for this kind.
     */
    fun resolve(kind: CapabilityKind, operation: String): CapabilityProvider? {
        val configured = registry.configuredForKind(kind)
        if (configured.isEmpty()) return null
        // Check for operation support first
        for (provider in configured) {
            if (provider.supportsOperation(operation)) return provider
        }
        // Fall back to first configured provider
        return configured.first()
    }

    /**
     * List all configured providers for [kind], ordered by preference.
     */
    fun available(kind: CapabilityKind): List<CapabilityProvider> =
        registry.configuredForKind(kind)

    /**
     * Whether any provider is configured for [kind].
     */
    fun isAvailable(kind: CapabilityKind): Boolean =
        registry.configuredForKind(kind).isNotEmpty()
}

/**
 * Extended by providers that declare specific supported operations
 * beyond their [CapabilityKind]. For example, an [ImageProvider] might
 * support "generate" but not "edit" or "upscale".
 */
interface OperationAwareProvider {
    fun supportsOperation(operation: kotlin.String): kotlin.Boolean
}

/**
 * Default implementation: a provider that doesn't implement
 * [OperationAwareProvider] is treated as supporting all operations
 * for its kind (backward compatibility with existing providers).
 */
private fun CapabilityProvider.supportsOperation(operation: kotlin.String): kotlin.Boolean =
    if (this is OperationAwareProvider) this.supportsOperation(operation) else true