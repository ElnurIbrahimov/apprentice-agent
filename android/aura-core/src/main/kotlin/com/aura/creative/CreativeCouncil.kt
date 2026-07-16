package com.aura.creative

import com.aura.agents.SubagentManager
import com.aura.agents.SubagentResult
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope

/**
 * Orchestrates a Creative Council session. Producers run in parallel
 * first, then critics review the proposals, and finally the Director
 * synthesizes a cohesive output.
 *
 * Pattern: producers -> critics -> director synthesis -> user selection.
 * Never exposes raw chain-of-thought — only concise rationale and
 * structured proposals.
 */
@Singleton
class CreativeCouncil @Inject constructor(
    private val subagentManager: SubagentManager,
) {
    /**
     * Run a full council session. Returns the synthesized result
     * with all member proposals.
     */
    suspend fun run(
        request: CouncilSessionRequest,
        executor: suspend (com.aura.agents.SubagentTask) -> SubagentResult,
    ): CouncilResult {
        val startTime = System.currentTimeMillis()
        val roles = request.roles.ifEmpty { CouncilRole.full }

        try {
            // Phase 1: Producers run in parallel
            val producerRoles = roles.filter { it in CouncilRole.producers }
            val producerProposals = if (producerRoles.isNotEmpty()) {
                runMembers(producerRoles, request, executor)
            } else emptyList()

            // Phase 2: Critics review producer proposals
            val criticRoles = roles.filter { it in CouncilRole.critics }
            val criticContext = producerProposals.map { "${it.role.displayName}: ${it.content}" }.joinToString("\n---\n")
            val criticProposals = if (criticRoles.isNotEmpty()) {
                runMembers(criticRoles, request.copy(brief = "${request.brief}\n\nProducer proposals:\n$criticContext"), executor)
            } else emptyList()

            // Phase 3: Director synthesizes
            val directorRole = roles.firstOrNull { it == CouncilRole.DIRECTOR }
            val allProposals = producerProposals + criticProposals
            val directorProposal = if (directorRole != null) {
                val synthesisContext = allProposals.map { "${it.role.displayName}: ${it.content}" }.joinToString("\n---\n")
                val directorRequest = request.copy(
                    brief = "Synthesize the following proposals into a final cohesive output.\n\nOriginal brief: ${request.brief}\n\nProposals:\n$synthesisContext",
                    roles = listOf(directorRole),
                )
                runMembers(listOf(directorRole), directorRequest, executor)
            } else emptyList()

            val directorOutput = directorProposal.firstOrNull()?.content ?: allProposals.firstOrNull()?.content ?: ""
            val allResults = allProposals + directorProposal

            return CouncilResult(
                projectId = request.projectId,
                brief = request.brief,
                directorOutput = directorOutput,
                proposals = allResults,
                totalDurationMs = System.currentTimeMillis() - startTime,
            )
        } catch (e: Exception) {
            return CouncilResult(
                projectId = request.projectId,
                brief = request.brief,
                directorOutput = "",
                proposals = emptyList(),
                totalDurationMs = System.currentTimeMillis() - startTime,
                success = false,
                error = e.message ?: "Council session failed",
            )
        }
    }

    private suspend fun runMembers(
        roles: List<CouncilRole>,
        request: CouncilSessionRequest,
        executor: suspend (com.aura.agents.SubagentTask) -> SubagentResult,
    ): List<CouncilProposal> {
        val tasks = roles.map { role ->
            subagentManager.createTask(role.toSubagentSpec(request), "council:${request.projectId}")
        }
        val results = subagentManager.spawnAll(tasks, executor)
        return results.zip(roles).map { (result, role) ->
            result.toProposal(role)
        }
    }
}