package com.aura.capabilities

import com.aura.providers.ProviderKeys
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Registry of every [CapabilityProvider] registered in the Hilt graph.
 *
 * The Hilt multibinding collects all providers; [forKind] returns the
 * first configured provider for a [CapabilityKind] so callers don't
 * have to know which specific backend is in use.
 *
 * Providers that the user has not configured (no API key) are filtered
 * out — [forKind] returns null if no provider for a kind is configured.
 */
@Singleton
class CapabilityRegistry @Inject constructor(
    private val providers: Map<String, @JvmSuppressWildcards CapabilityProvider>,
    private val providerKeys: ProviderKeys,
) {
    fun all(): List<CapabilityProvider> = providers.values.toList()
    fun byPrefix(prefix: String): CapabilityProvider? = providers[prefix]

    /**
     * Return the first configured provider for [kind], or null if none.
     * Iteration order follows Hilt's multibinding insertion order so
     * providers added in [com.aura.capabilities.di.CapabilityModule] are
     * tried in declaration order.
     */
    fun forKind(kind: CapabilityKind): CapabilityProvider? =
        providers.values.firstOrNull { it.kind == kind && it.isConfigured() }

    fun configuredForKind(kind: CapabilityKind): List<CapabilityProvider> =
        providers.values.filter { it.kind == kind && it.isConfigured() }
}
