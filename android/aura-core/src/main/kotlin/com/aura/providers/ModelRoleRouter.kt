package com.aura.providers

import com.aura.data.UserPreferences
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Task roles that map to user-configured models. Each role has a
 * [UserPreferences] preference key and a fallback chain.
 *
 * The router never hardcodes a model ID. If the user hasn't configured
 * a role-specific model, it falls back to the default model.
 */
enum class ModelRole(val key: kotlin.String, val displayName: kotlin.String) {
    CONVERSATION("default_model", "Conversation"),
    BACKGROUND("background_model", "Background"),
    DEEP_RESEARCH("deep_mode_model", "Deep Research"),
    CREATIVE_DRAFT("creative_draft_model", "Creative Draft"),
    CREATIVE_CRITIC("creative_critic_model", "Creative Critic"),
    PLANNER("planner_model", "Planner"),
    VERIFIER("verifier_model", "Verifier"),
    EMBEDDING("embedding_model", "Embedding"),
    ;

    companion object {
        /**
         * Roles the user can explicitly configure in Settings.
         * EMBEDDING is managed separately (it's a capability, not a chat model).
         */
        val configurable: List<ModelRole> get() = listOf(
            CONVERSATION, BACKGROUND, DEEP_RESEARCH, CREATIVE_DRAFT, CREATIVE_CRITIC, PLANNER, VERIFIER,
        )
    }
}

/**
 * Resolves a model ID for a [ModelRole] from user preferences, with
 * fallback to the default conversation model. Never invents or
 * hardcodes a model ID.
 */
@Singleton
class ModelRoleRouter @Inject constructor(
    private val userPreferences: UserPreferences,
    private val providerRegistry: ProviderRegistry,
) {
    /**
     * Returns the user's preferred model for [role], or the default
     * model if no role-specific preference is set.
     *
     * Returns null if no model is configured at all — the caller must
     * handle this honestly (show "configure a model" prompt).
     */
    suspend fun resolve(role: ModelRole): kotlin.String? {
        // Role-specific preference
        val roleModel = userPreferences.forRole(role).first()
        if (!roleModel.isNullOrBlank()) return roleModel
        // Fallback: conversation default
        if (role != ModelRole.CONVERSATION) {
            val default = userPreferences.defaultModel.first()
            if (!default.isNullOrBlank()) return default
        }
        return null
    }

    /**
     * Observable flow of the model for [role]. Emits the role-specific
     * model, or the default model if not set.
     */
    fun observe(role: ModelRole): Flow<kotlin.String?> =
        userPreferences.forRole(role).map { roleModel ->
            roleModel.takeIf { !it.isNullOrBlank() }
        }

    /**
     * Returns true if any model is configured for [role] (either
     * role-specific or the default fallback).
     */
    suspend fun isConfigured(role: ModelRole): kotlin.Boolean {
        return resolve(role) != null
    }

    /**
     * Lists all models available from configured providers that could
     * serve [role]. Uses the live provider catalog, not hardcoded lists.
     */
    suspend fun availableModels(role: ModelRole): List<kotlin.String> {
        val configured = providerRegistry.configured()
        return configured.flatMap { provider ->
            runCatching { provider.listModels() }.getOrDefault(emptyList())
        }.distinct()
    }
}