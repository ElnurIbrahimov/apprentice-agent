package com.aura.kg

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Best-effort extractor that runs after an assistant turn completes.
 *
 * It takes the last N turns of the conversation, asks a cloud LLM to extract
 * a knowledge graph, and saves the result to [KnowledgeGraphRepository].
 * Failures are swallowed — the chat stream must not be interrupted.
 */
@Singleton
class ConversationKgExtractor @Inject constructor(
    private val knowledgeGraphTool: com.aura.tools.KnowledgeGraphTool,
    private val repository: KnowledgeGraphRepository,
) {
    private val scope = CoroutineScope(SupervisorJob())

    fun extract(turnText: String) {
        scope.launch {
            try {
                val (nodes, edges) = knowledgeGraphTool.extract(turnText)
                if (nodes.isNotEmpty() || edges.isNotEmpty()) {
                    repository.saveGraph(nodes, edges, "turn-${System.currentTimeMillis()}")
                }
            } catch (_: Exception) {
                // Best effort: do not crash the chat stream.
            }
        }
    }
}
