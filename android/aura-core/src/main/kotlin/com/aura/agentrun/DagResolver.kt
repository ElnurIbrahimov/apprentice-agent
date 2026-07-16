package com.aura.agentrun

import javax.inject.Inject
import javax.inject.Singleton

/**
 * Resolves DAG dependencies between [StepEntity] records. A step is
 * ready to execute when all its [StepEntity.dependsOn] step IDs have
 * status=SUCCESS. Cycles are detected and rejected.
 */
@Singleton
class DagResolver @Inject constructor() {

    /**
     * Returns steps that are ready to execute (all dependencies satisfied).
     */
    fun readySteps(steps: List<StepEntity>): List<StepEntity> {
        val stepMap = steps.associateBy { it.id }
        return steps.filter { step ->
            step.status == "PENDING" && dependenciesSatisfied(step, stepMap)
        }
    }

    /**
     * Topological sort of steps. Returns batches of steps that can
     * execute in parallel. Throws if a cycle is detected.
     */
    fun topologicalBatches(steps: List<StepEntity>): List<List<StepEntity>> {
        val stepMap = steps.associateBy { it.id }
        val depthCache = mutableMapOf<kotlin.String, Int>()
        val inProgress = mutableSetOf<kotlin.String>()

        fun depthOf(step: StepEntity): Int {
            if (step.id in depthCache) return depthCache[step.id]!!
            if (step.id in inProgress) throw IllegalStateException("Cycle detected at step ${step.id}")
            inProgress.add(step.id)

            val depIds = parseDependsOn(step.dependsOn)
            val maxDepDepth = depIds.mapNotNull { depId ->
                stepMap[depId]?.let { depthOf(it) + 1 }
            }.maxOrNull() ?: 0

            inProgress.remove(step.id)
            depthCache[step.id] = maxDepDepth
            return maxDepDepth
        }

        // Compute depth for each step
        for (step in steps) depthOf(step)

        // Group by depth
        val maxDepth = depthCache.values.maxOrNull() ?: 0
        val batches = (0..maxDepth).map { mutableListOf<StepEntity>() }
        for (step in steps) {
            val d = depthCache[step.id]!!
            batches[d].add(step)
        }
        return batches
    }

    /**
     * Returns true if all dependencies of [step] have status=SUCCESS.
     */
    private fun dependenciesSatisfied(step: StepEntity, stepMap: Map<kotlin.String, StepEntity>): kotlin.Boolean {
        val depIds = parseDependsOn(step.dependsOn)
        if (depIds.isEmpty()) return true
        return depIds.all { depId ->
            stepMap[depId]?.status == "SUCCESS"
        }
    }

    /**
     * Returns true if a cycle exists in the dependency graph.
     */
    fun hasCycle(steps: List<StepEntity>): kotlin.Boolean {
        return try {
            topologicalBatches(steps)
            false
        } catch (e: IllegalStateException) {
            true
        }
    }

    private fun parseDependsOn(json: kotlin.String): List<kotlin.String> {
        if (json.isBlank() || json == "[]") return emptyList()
        return json.trim()
            .removeSurrounding("[", "]")
            .split(",")
            .map { it.trim().trim('"') }
            .filter { it.isNotBlank() }
    }
}