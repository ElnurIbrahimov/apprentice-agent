package com.aura.creative

import com.aura.data.UserPreferences
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton

enum class CreativeMode(val label: String, val instruction: String, val temperature: Double) {
    BRAINSTORM(
        "Brainstorm",
        "Generate several distinct, specific possibilities. Explain the dramatic trade-off of each without declaring one canon.",
        0.95,
    ),
    OUTLINE(
        "Outline",
        "Create a structured sequence of story beats with escalation, reversals, setup/payoff, and an earned ending.",
        0.75,
    ),
    DRAFT(
        "Draft",
        "Write polished prose or screenplay material in the project's established voice. Do not contradict canon.",
        0.85,
    ),
    REWRITE(
        "Rewrite",
        "Rewrite the supplied text while preserving its intent and the user's voice. Improve specificity, rhythm, subtext, and clarity.",
        0.75,
    ),
    SIMULATE(
        "Simulate",
        "Run this as a non-canon what-if simulation. Trace plausible decisions, reactions, second-order consequences, and the resulting world state.",
        0.9,
    ),
    CONTINUITY(
        "Continuity",
        "Audit the supplied text against canon. List concrete contradictions, timeline problems, unexplained changes, and possible repairs. Do not invent problems.",
        0.2,
    ),
}

@Singleton
class CreativeEngine @Inject constructor(
    private val providerRegistry: ProviderRegistry,
    private val userPreferences: UserPreferences,
    private val projectStore: CreativeProjectStore,
) {
    fun generate(
        projectId: String,
        mode: CreativeMode,
        input: String,
        perspective: String = "",
    ): Flow<String> = flow {
        require(input.isNotBlank()) { "A writing prompt is required." }
        val project = projectStore.get(projectId)
            ?: throw IllegalArgumentException("Creative project not found.")
        val model = resolveModel()
        val template = WritingTemplates.byId(project.templateId)
        val systemPrompt = buildString {
            appendLine("You are Aura's Creative Studio engine working inside one durable project.")
            appendLine(mode.instruction)
            appendLine("Preserve the user's authorship: assist and extend; never claim generated exploration is canon.")
            template?.let { appendLine("FORM: ${it.prompt}") }
            appendLine()
            append(buildProjectContext(project))
        }
        val userPrompt = buildString {
            if (perspective.isNotBlank()) appendLine("Perspective: $perspective")
            append(input.trim())
        }
        val output = StringBuilder()
        providerRegistry.chat(
            model,
            listOf(
                ProviderMessage(ProviderMessage.Role.system, systemPrompt),
                ProviderMessage(ProviderMessage.Role.user, userPrompt),
            ),
            ChatOptions(temperature = mode.temperature, maxTokens = 4_096),
        ).collect { chunk ->
            chunk.error?.let { throw IllegalStateException(it.message) }
            chunk.text?.takeIf(String::isNotEmpty)?.let { text ->
                output.append(text)
                emit(text)
            }
        }
        if (output.isNotBlank()) {
            projectStore.incrementTurn(projectId)
            if (mode == CreativeMode.SIMULATE) {
                projectStore.recordSimulation(
                    projectId,
                    SimulationRecord(
                        premise = input.trim(),
                        outcome = output.toString(),
                        perspective = perspective.trim(),
                    ),
                )
            }
        }
    }

    internal suspend fun resolveModel(): String {
        userPreferences.defaultModel.first()?.takeIf(String::isNotBlank)?.let { return it }
        for (provider in providerRegistry.configured()) {
            val model = runCatching { provider.listModels().firstOrNull() }.getOrNull()
            if (!model.isNullOrBlank()) return "${provider.prefix}:$model"
        }
        throw IllegalStateException("Configure an LLM provider and choose a default model before using Creative Studio.")
    }

    fun buildProjectContext(project: CreativeProject): String {
        val world = project.world
        val text = buildString {
            appendLine("PROJECT: ${project.name}")
            appendLine("DESCRIPTION: ${project.description.ifBlank { "Not set" }}")
            appendLine("GENRE: ${project.genre.ifBlank { "Not set" }}")
            appendLine("TONE: ${project.tone.ifBlank { "Not set" }}")
            appendLine("WORLD OVERVIEW: ${world.overview.ifBlank { "Not set" }}")
            if (world.characters.isNotEmpty()) {
                appendLine("CHARACTERS:")
                world.characters.forEach { appendLine("- ${it.name}; role=${it.role}; traits=${it.traits.joinToString()}; backstory=${it.backstory}; motivation=${it.motivation}; arc=${it.arc}") }
            }
            if (world.locations.isNotEmpty()) {
                appendLine("LOCATIONS:")
                world.locations.forEach { appendLine("- ${it.name}; type=${it.type}; ${it.description}; significance=${it.significance}") }
            }
            if (world.factions.isNotEmpty()) {
                appendLine("FACTIONS:")
                world.factions.forEach { appendLine("- ${it.name}; ideology=${it.ideology}; members=${it.members.joinToString()}; rivals=${it.rivals.joinToString()}") }
            }
            if (world.rules.isNotEmpty()) {
                appendLine("WORLD RULES:")
                world.rules.forEach { appendLine("- ${it.name} [${it.category}]: ${it.description}; impact=${it.impact}") }
            }
            if (world.timeline.isNotEmpty()) {
                appendLine("TIMELINE:")
                world.timeline.forEach { appendLine("- ${it.date} ${it.title}: ${it.description}") }
            }
            if (world.outline.isNotEmpty()) {
                appendLine("OUTLINE:")
                world.outline.forEachIndexed { i, it -> appendLine("${i + 1}. ${it.title} [${it.status}]: ${it.summary}") }
            }
            if (world.notes.isNotBlank()) appendLine("NOTES: ${world.notes}")
        }
        return text.take(MAX_CONTEXT_CHARS)
    }

    companion object {
        const val MAX_CONTEXT_CHARS = 24_000
    }
}