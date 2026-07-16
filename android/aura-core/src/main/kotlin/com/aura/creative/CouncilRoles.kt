package com.aura.creative

import com.aura.agents.SubagentSpec
import com.aura.agents.SubagentResult
import kotlinx.serialization.Serializable

/**
 * Creative Council member roles. Each role has a specific responsibility,
 * model role, tool allowlist, and expected output shape.
 *
 * The Director synthesizes proposals from other members into a
 * cohesive final output. Critics review and flag issues. Producers
 * create draft content. Researchers gather context.
 */
@Serializable
enum class CouncilRole(
    val displayName: kotlin.String,
    val modelRole: kotlin.String,
    val description: kotlin.String,
    val defaultToolAllowlist: List<kotlin.String>,
) {
    DIRECTOR(
        "Director",
        "CREATIVE_CRITIC",
        "Synthesizes proposals from all members into a cohesive final output. Does not produce drafts — chooses and merges.",
        listOf("creative_read_project", "creative_add_world_item"),
    ),
    WRITER(
        "Writer",
        "CREATIVE_DRAFT",
        "Produces draft text: scenes, chapters, dialogue, descriptions.",
        listOf("creative_read_project", "creative_add_world_item", "web_search_capability", "recall"),
    ),
    STORY_EDITOR(
        "Story Editor",
        "CREATIVE_CRITIC",
        "Reviews story structure, pacing, and narrative arc. Flags structural issues.",
        listOf("creative_read_project", "recall"),
    ),
    CONTINUITY_EDITOR(
        "Continuity Editor",
        "CREATIVE_CRITIC",
        "Checks for canon contradictions, timeline errors, and knowledge-chronology violations.",
        listOf("creative_read_project", "recall"),
    ),
    WORLD_SIMULATOR(
        "World Simulator",
        "CREATIVE_DRAFT",
        "Simulates world-state changes and predicts consequences. Produces state deltas.",
        listOf("creative_read_project", "creative_add_world_item"),
    ),
    RESEARCHER(
        "Researcher",
        "DEEP_RESEARCH",
        "Gathers external context: sources, references, real-world parallels.",
        listOf("web_search_capability", "recall", "deep_research"),
    ),
    ART_DIRECTOR(
        "Art Director",
        "CREATIVE_DRAFT",
        "Defines visual language: palette, composition, style references for image generation.",
        listOf("creative_read_project", "image_generate"),
    ),
    CINEMATOGRAPHER(
        "Cinematographer",
        "CREATIVE_DRAFT",
        "Defines shot composition, camera angles, lighting for storyboard and video.",
        listOf("creative_read_project", "image_generate"),
    ),
    SOUND_DESIGNER(
        "Sound Designer",
        "CREATIVE_DRAFT",
        "Defines audio direction: voice casting, music motifs, SFX choices.",
        listOf("creative_read_project", "tts_speak"),
    ),
    AUDIENCE_CRITIC(
        "Audience Critic",
        "CREATIVE_CRITIC",
        "Reviews from the audience perspective: engagement, clarity, emotional impact.",
        listOf("creative_read_project", "recall"),
    );

    companion object {
        /**
         * Roles that produce proposals (run in parallel first).
         * The Director runs last to synthesize.
         */
        val producers: List<CouncilRole> get() = listOf(
            WRITER, RESEARCHER, WORLD_SIMULATOR, ART_DIRECTOR, CINEMATOGRAPHER, SOUND_DESIGNER,
        )

        /**
         * Roles that review/critique (run after producers, before Director).
         */
        val critics: List<CouncilRole> get() = listOf(
            STORY_EDITOR, CONTINUITY_EDITOR, AUDIENCE_CRITIC,
        )

        /**
         * Full council: producers -> critics -> director.
         */
        val full: List<CouncilRole> get() = producers + critics + listOf(DIRECTOR)
    }
}

/**
 * A council session request. Defines which roles to activate,
 * the brief, and the context.
 */
@Serializable
data class CouncilSessionRequest(
    val projectId: kotlin.String,
    val brief: kotlin.String,
    val roles: List<CouncilRole> = CouncilRole.full,
    val contextArtifactIds: List<kotlin.String> = emptyList(),
    val budgetMs: kotlin.Long = 120_000L,
    val maxToolCallsPerMember: Int = 10,
)

/**
 * A proposal from a single council member. Contains the member's
 * output, rationale, and any artifacts it references or created.
 */
@Serializable
data class CouncilProposal(
    val role: CouncilRole,
    val content: kotlin.String,
    val rationale: kotlin.String,
    val confidence: Float = 0.7f,
    val issues: List<kotlin.String> = emptyList(),
    val createdArtifactIds: List<kotlin.String> = emptyList(),
    val success: kotlin.Boolean = true,
    val error: kotlin.String = "",
)

/**
 * The synthesized result from a council session. Contains the
 * Director's final output, all member proposals, and metadata.
 */
@Serializable
data class CouncilResult(
    val projectId: kotlin.String,
    val brief: kotlin.String,
    val directorOutput: kotlin.String,
    val proposals: List<CouncilProposal>,
    val totalDurationMs: kotlin.Long = 0L,
    val success: kotlin.Boolean = true,
    val error: kotlin.String = "",
)

/**
 * Convert a [CouncilRole] and session request into a [SubagentSpec].
 */
fun CouncilRole.toSubagentSpec(
    request: CouncilSessionRequest,
): SubagentSpec = SubagentSpec(
    role = displayName,
    objective = when (this) {
        CouncilRole.DIRECTOR -> "Synthesize the best elements from all proposals into a final cohesive output. Brief: ${request.brief}"
        else -> "$displayName perspective on: ${request.brief}"
    },
    contextArtifactIds = request.contextArtifactIds,
    modelRole = modelRole,
    toolAllowlist = defaultToolAllowlist,
    budgetMs = request.budgetMs / request.roles.size.coerceAtLeast(1),
    maxToolCalls = request.maxToolCallsPerMember,
)

/**
 * Convert a [SubagentResult] into a [CouncilProposal] for a given role.
 */
fun SubagentResult.toProposal(role: CouncilRole): CouncilProposal = CouncilProposal(
    role = role,
    content = output,
    rationale = rationale,
    createdArtifactIds = createdArtifactIds,
    success = success,
    error = error,
    confidence = if (success) 0.7f else 0.0f,
)