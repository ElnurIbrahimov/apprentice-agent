package com.aura.pipeline

import kotlinx.serialization.Serializable

/**
 * A reusable creative production pipeline. Defines a sequence of
 * stages that produce a complete creative work — novel, screenplay,
 * comic, short film, trailer, podcast, RPG campaign, brand campaign.
 *
 * Each stage has:
 * - A role (which council member or tool executes it)
 * - Inputs (artifact references or free text)
 * - Output artifact type
 * - Dependencies (which stages must complete first)
 *
 * Pipelines are inspectable and editable — the user can skip stages,
 * reorder, or add custom stages.
 */
@Serializable
data class PipelineStage(
    val id: kotlin.String,
    val name: kotlin.String,
    val description: kotlin.String,
    /** Council role or tool name that executes this stage. */
    val executor: kotlin.String,
    /** Output artifact type: "text", "image", "audio", "video", "3d", "document", "storyboard". */
    val outputType: kotlin.String = "text",
    /** Stage IDs that must complete before this stage. */
    val dependsOn: List<kotlin.String> = emptyList(),
    /** Whether this stage is optional. */
    val optional: kotlin.Boolean = false,
    /** Prompt template with {brief} and {context} placeholders. */
    val promptTemplate: kotlin.String = "",
)

/**
 * A complete production pipeline definition.
 */
@Serializable
data class ProductionPipeline(
    val id: kotlin.String,
    val name: kotlin.String,
    val description: kotlin.String,
    val stages: List<PipelineStage>,
    /** Default council roles to activate for this pipeline. */
    val defaultRoles: List<kotlin.String> = emptyList(),
    /** Estimated total time in minutes. */
    val estimatedMinutes: Int = 30,
) {
    companion object {
        /** Novel pipeline: concept → outline → characters → world bible → chapter drafts → continuity review → final */
        val novel = ProductionPipeline(
            id = "novel",
            name = "Novel",
            description = "Full novel production from concept to final manuscript.",
            defaultRoles = listOf("WRITER", "STORY_EDITOR", "CONTINUITY_EDITOR", "DIRECTOR"),
            estimatedMinutes = 120,
            stages = listOf(
                PipelineStage("concept", "Concept", "Define the core concept and premise.", "DIRECTOR", "text"),
                PipelineStage("outline", "Outline", "Create chapter-by-chapter outline.", "WRITER", "text", listOf("concept")),
                PipelineStage("characters", "Characters", "Develop character profiles.", "WRITER", "text", listOf("concept")),
                PipelineStage("world", "World Bible", "Build the world bible.", "WORLD_SIMULATOR", "text", listOf("concept")),
                PipelineStage("chapters", "Chapter Drafts", "Draft each chapter.", "WRITER", "text", listOf("outline", "characters", "world")),
                PipelineStage("continuity", "Continuity Review", "Check for contradictions.", "CONTINUITY_EDITOR", "text", listOf("chapters")),
                PipelineStage("final", "Final Polish", "Director synthesizes final manuscript.", "DIRECTOR", "text", listOf("continuity")),
            ),
        )

        /** Screenplay pipeline */
        val screenplay = ProductionPipeline(
            id = "screenplay",
            name = "Screenplay",
            description = "Screenplay from logline to final draft with shot breakdown.",
            defaultRoles = listOf("WRITER", "STORY_EDITOR", "CINEMATOGRAPHER", "DIRECTOR"),
            estimatedMinutes = 90,
            stages = listOf(
                PipelineStage("logline", "Logline", "One-sentence pitch.", "DIRECTOR", "text"),
                PipelineStage("treatment", "Treatment", "Full story treatment.", "WRITER", "text", listOf("logline")),
                PipelineStage("characters", "Characters", "Character bios and arcs.", "WRITER", "text", listOf("logline")),
                PipelineStage("scenes", "Scene Breakdown", "Scene-by-scene breakdown.", "WRITER", "text", listOf("treatment", "characters")),
                PipelineStage("dialogue", "Dialogue Draft", "Draft dialogue for all scenes.", "WRITER", "text", listOf("scenes")),
                PipelineStage("shots", "Shot List", "Camera angles and shots per scene.", "CINEMATOGRAPHER", "text", listOf("dialogue")),
                PipelineStage("final", "Final Draft", "Director finalizes screenplay.", "DIRECTOR", "text", listOf("shots")),
            ),
        )

        /** Short film pipeline: concept → storyboard → shots → audio direction → rough cut */
        val shortFilm = ProductionPipeline(
            id = "short_film",
            name = "Short Film",
            description = "Short film from concept to rough cut with storyboard.",
            defaultRoles = listOf("DIRECTOR", "WRITER", "ART_DIRECTOR", "CINEMATOGRAPHER", "SOUND_DESIGNER"),
            estimatedMinutes = 180,
            stages = listOf(
                PipelineStage("concept", "Concept", "Film concept and visual direction.", "DIRECTOR", "text"),
                PipelineStage("storyboard", "Storyboard", "12-frame storyboard.", "ART_DIRECTOR", "image", listOf("concept")),
                PipelineStage("shots", "Shot Prompts", "Video shot prompts per frame.", "CINEMATOGRAPHER", "text", listOf("storyboard")),
                PipelineStage("voice", "Voice Direction", "Voice casting and line reads.", "SOUND_DESIGNER", "text", listOf("concept")),
                PipelineStage("music", "Music Direction", "Music motifs and score direction.", "SOUND_DESIGNER", "text", listOf("concept")),
                PipelineStage("rough", "Rough Cut", "Timeline assembly and rough cut.", "DIRECTOR", "text", listOf("shots", "voice", "music")),
            ),
        )

        /** Trailer pipeline: brief → concepts → beat sheet → storyboard → shot prompts → voice → music → rough cut */
        val trailer = ProductionPipeline(
            id = "trailer",
            name = "Trailer",
            description = "Trailer production from brief to rough cut timeline.",
            defaultRoles = listOf("DIRECTOR", "WRITER", "ART_DIRECTOR", "CINEMATOGRAPHER", "SOUND_DESIGNER"),
            estimatedMinutes = 60,
            stages = listOf(
                PipelineStage("brief", "Creative Brief", "Parse the brief and define goals.", "DIRECTOR", "text"),
                PipelineStage("concepts", "Three Concepts", "Generate three narrative concepts.", "WRITER", "text", listOf("brief")),
                PipelineStage("beats", "Beat Sheet", "Select concept and create beat sheet.", "DIRECTOR", "text", listOf("concepts")),
                PipelineStage("storyboard", "Storyboard", "12-frame storyboard.", "ART_DIRECTOR", "image", listOf("beats")),
                PipelineStage("shots", "Shot Prompts", "Video shot prompts per frame.", "CINEMATOGRAPHER", "text", listOf("storyboard")),
                PipelineStage("voice", "Voice Casting", "Voice direction and casting.", "SOUND_DESIGNER", "text", listOf("beats")),
                PipelineStage("music", "Music Direction", "Music and SFX direction.", "SOUND_DESIGNER", "text", listOf("beats")),
                PipelineStage("rough", "Rough Cut", "Timeline assembly.", "DIRECTOR", "text", listOf("shots", "voice", "music")),
            ),
        )

        /** Podcast drama pipeline */
        val podcastDrama = ProductionPipeline(
            id = "podcast_drama",
            name = "Podcast Drama",
            description = "Audio drama from concept to final script with voice direction.",
            defaultRoles = listOf("DIRECTOR", "WRITER", "SOUND_DESIGNER"),
            estimatedMinutes = 45,
            stages = listOf(
                PipelineStage("concept", "Concept", "Premise and tone.", "DIRECTOR", "text"),
                PipelineStage("script", "Script", "Full episode script.", "WRITER", "text", listOf("concept")),
                PipelineStage("voices", "Voice Direction", "Character voice direction.", "SOUND_DESIGNER", "text", listOf("script")),
                PipelineStage("sfx", "SFX Plan", "Sound effects plan per scene.", "SOUND_DESIGNER", "text", listOf("script")),
                PipelineStage("final", "Final Script", "Director finalizes.", "DIRECTOR", "text", listOf("voices", "sfx")),
            ),
        )

        /** RPG campaign pipeline */
        val rpgCampaign = ProductionPipeline(
            id = "rpg_campaign",
            name = "RPG Campaign",
            description = "Tabletop RPG campaign from world concept to encounter set.",
            defaultRoles = listOf("DIRECTOR", "WRITER", "WORLD_SIMULATOR"),
            estimatedMinutes = 60,
            stages = listOf(
                PipelineStage("world", "World Concept", "Campaign world and setting.", "WORLD_SIMULATOR", "text"),
                PipelineStage("factions", "Factions", "Major factions and relationships.", "WORLD_SIMULATOR", "text", listOf("world")),
                PipelineStage("plot", "Campaign Arc", "Main plot arc and milestones.", "WRITER", "text", listOf("world", "factions")),
                PipelineStage("encounters", "Encounter Set", "Key encounters and challenges.", "WRITER", "text", listOf("plot")),
                PipelineStage("final", "Campaign Bible", "Director compiles campaign bible.", "DIRECTOR", "text", listOf("encounters")),
            ),
        )

        /** All built-in pipelines. */
        val all: List<ProductionPipeline> = listOf(
            novel, screenplay, shortFilm, trailer, podcastDrama, rpgCampaign,
        )

        /** Get a pipeline by ID. */
        fun byId(id: kotlin.String): ProductionPipeline? = all.firstOrNull { it.id == id }
    }
}