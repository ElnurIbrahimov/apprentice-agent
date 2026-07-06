package com.aura.agent

import com.aura.core.error.AuraError
import com.aura.providers.ProviderError

/**
 * Convert an agent/tool failure into a typed [AuraError] for the UI.
 */
fun Throwable.toAuraError(): AuraError = AuraError.Unknown(
    message = message ?: "Unexpected error",
    cause = this,
)
