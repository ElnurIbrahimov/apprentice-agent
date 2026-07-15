package com.aura.creative

import kotlinx.serialization.Serializable
import java.util.UUID

@Serializable
data class WorldBible(
    val overview: String = "",
    val characters: List<WorldCharacter> = emptyList(),
    val locations: List<WorldLocation> = emptyList(),
    val factions: List<WorldFaction> = emptyList(),
    val rules: List<WorldRule> = emptyList(),
    val timeline: List<WorldEvent> = emptyList(),
    val outline: List<StoryBeat> = emptyList(),
    val simulations: List<SimulationRecord> = emptyList(),
    val continuityNotes: List<ContinuityIssue> = emptyList(),
    val notes: String = "",
)

@Serializable
data class WorldCharacter(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val role: String = "",
    val traits: List<String> = emptyList(),
    val backstory: String = "",
    val motivation: String = "",
    val arc: String = "",
)

@Serializable
data class WorldLocation(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val description: String = "",
    val type: String = "",
    val significance: String = "",
)

@Serializable
data class WorldFaction(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val ideology: String = "",
    val members: List<String> = emptyList(),
    val rivals: List<String> = emptyList(),
)

@Serializable
data class WorldRule(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val description: String,
    val category: String = "world",
    val impact: String = "",
)

@Serializable
data class WorldEvent(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val date: String = "",
    val description: String = "",
    val participants: List<String> = emptyList(),
)

@Serializable
data class StoryBeat(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val summary: String = "",
    val status: String = "planned",
)

@Serializable
data class SimulationRecord(
    val id: String = UUID.randomUUID().toString(),
    val premise: String,
    val outcome: String,
    val perspective: String = "",
    val createdAt: Long = System.currentTimeMillis(),
    val canonized: Boolean = false,
)

@Serializable
data class ContinuityIssue(
    val id: String = UUID.randomUUID().toString(),
    val severity: String = "warning",
    val category: String = "continuity",
    val description: String,
    val affectedEntity: String = "",
)

@Serializable
data class WritingTemplate(
    val id: String,
    val name: String,
    val description: String,
    val icon: String,
    val prompt: String,
)

object WritingTemplates {
    val all = listOf(
        WritingTemplate(
            id = "novel",
            name = "Novel",
            description = "Long-form fiction with acts, character arcs, and scene continuity",
            icon = "📖",
            prompt = "Develop this as a novel. Track character arcs, setup/payoff, pacing, and scene-level continuity.",
        ),
        WritingTemplate(
            id = "short-story",
            name = "Short story",
            description = "A focused narrative built around one transformation",
            icon = "✒️",
            prompt = "Shape this as a short story with a sharp premise, economical scenes, and a resonant ending.",
        ),
        WritingTemplate(
            id = "screenplay",
            name = "Screenplay",
            description = "Visual scenes, dialogue, action lines, and dramatic beats",
            icon = "🎬",
            prompt = "Write in screenplay-minded scenes: visual action, playable dialogue, subtext, and clean dramatic turns.",
        ),
        WritingTemplate(
            id = "rpg-world",
            name = "RPG world",
            description = "Lore, factions, locations, conflicts, quests, and simulation-ready rules",
            icon = "🗺️",
            prompt = "Build this as a simulation-ready RPG world with factions, locations, rules, conflicts, and consequence chains.",
        ),
        WritingTemplate(
            id = "character-study",
            name = "Character study",
            description = "Voice, psychology, relationships, contradictions, and internal change",
            icon = "🎭",
            prompt = "Center this work on psychology, voice, contradictory motives, relationships, and earned internal change.",
        ),
    )

    fun byId(id: String): WritingTemplate? = all.find { it.id == id }
}