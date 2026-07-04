package com.aura.agent

import com.aura.core.error.AuraError

/**
 * Convert a raw provider error DTO into the rich [AuraError] domain model.
 */
fun com.aura.providers.ProviderError.toAuraError(providerId: String? = null): AuraError =
    this.toAuraError(providerId)

/**
 * Convert an agent/tool failure into a typed [AuraError] for the UI.
 */
fun Throwable.toAuraError(): AuraError =
    this.toAuraError()
