package com.aura.usage

import android.content.Context
import android.content.SharedPreferences
import com.aura.providers.Usage
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.ceil

@Serializable
data class ModelUsage(
    val modelId: String,
    val promptTokens: Long = 0,
    val completionTokens: Long = 0,
    val calls: Long = 0,
    /** True when at least one call used Aura's chars/4 fallback. */
    val estimated: Boolean = false,
)

@Serializable
data class UsageSnapshot(
    val promptTokens: Long = 0,
    val completionTokens: Long = 0,
    val calls: Long = 0,
    val toolResultChars: Long = 0,
    val models: List<ModelUsage> = emptyList(),
) {
    val totalTokens: Long get() = promptTokens + completionTokens
}

/**
 * Persistent, provider-central usage ledger. Exact token metadata is used when
 * a provider reports it; otherwise counts are explicitly marked as estimated.
 * Pricing is deliberately not hardcoded because model identifiers and prices
 * change independently of the app.
 */
@Singleton
class UsageTracker private constructor(
    private val preferences: SharedPreferences?,
) {
    @Inject
    constructor(@ApplicationContext context: Context) : this(
        context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE),
    )

    /** In-memory constructor for deterministic unit tests. */
    constructor() : this(null)

    private val json = Json { ignoreUnknownKeys = true }
    private val _snapshot = MutableStateFlow(load())
    val snapshot: StateFlow<UsageSnapshot> = _snapshot.asStateFlow()

    @Synchronized
    fun recordLlmCall(
        modelId: String,
        inputChars: Int,
        outputChars: Int,
        reportedUsage: Usage? = null,
    ) {
        val hasExactBreakdown = reportedUsage != null &&
            (reportedUsage.promptTokens > 0 || reportedUsage.completionTokens > 0)
        val prompt = if (hasExactBreakdown) {
            reportedUsage!!.promptTokens.toLong()
        } else {
            estimateTokens(inputChars)
        }
        val completion = if (hasExactBreakdown) {
            reportedUsage!!.completionTokens.toLong()
        } else {
            estimateTokens(outputChars)
        }
        val estimated = !hasExactBreakdown
        val current = _snapshot.value
        val existing = current.models.firstOrNull { it.modelId == modelId }
            ?: ModelUsage(modelId = modelId)
        val updatedModel = existing.copy(
            promptTokens = existing.promptTokens + prompt,
            completionTokens = existing.completionTokens + completion,
            calls = existing.calls + 1,
            estimated = existing.estimated || estimated,
        )
        update(
            current.copy(
                promptTokens = current.promptTokens + prompt,
                completionTokens = current.completionTokens + completion,
                calls = current.calls + 1,
                models = (current.models.filterNot { it.modelId == modelId } + updatedModel)
                    .sortedByDescending { it.promptTokens + it.completionTokens },
            ),
        )
    }

    @Synchronized
    fun recordToolResult(resultChars: Int) {
        if (resultChars <= 0) return
        update(_snapshot.value.copy(toolResultChars = _snapshot.value.toolResultChars + resultChars))
    }

    @Synchronized
    fun reset() {
        update(UsageSnapshot())
    }

    fun summary(): String {
        val value = _snapshot.value
        return "${formatCount(value.totalTokens)} tokens · ${value.calls} calls"
    }

    private fun update(value: UsageSnapshot) {
        _snapshot.value = value
        preferences?.edit()?.putString(KEY_LEDGER, json.encodeToString(value))?.apply()
    }

    private fun load(): UsageSnapshot {
        val encoded = preferences?.getString(KEY_LEDGER, null) ?: return UsageSnapshot()
        return runCatching { json.decodeFromString<UsageSnapshot>(encoded) }
            .getOrDefault(UsageSnapshot())
    }

    private fun estimateTokens(chars: Int): Long =
        if (chars <= 0) 0 else ceil(chars / 4.0).toLong()

    private fun formatCount(value: Long): String = when {
        value >= 1_000_000 -> "%.1fM".format(value / 1_000_000.0)
        value >= 1_000 -> "%.1fK".format(value / 1_000.0)
        else -> value.toString()
    }

    private companion object {
        const val PREFERENCES_NAME = "aura_usage"
        const val KEY_LEDGER = "ledger_v1"
    }
}
