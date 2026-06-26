package com.aura.kg

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Best-effort extractor that runs after an assistant turn completes.
 *
 * Debounced: multiple extract() calls within [DEBOUNCE_MS] collapse to one
 * run on the most recent text. A long run (e.g. an active extraction job)
 * is left to finish — we just don't queue another until it's done.
 *
 * Failures are swallowed — the chat stream must not be interrupted.
 */
@Singleton
class ConversationKgExtractor @Inject constructor(
    private val knowledgeGraphTool: com.aura.tools.KnowledgeGraphTool,
    private val repository: KnowledgeGraphRepository,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    @Volatile private var pendingText: String? = null
    @Volatile private var debounceJob: Job? = null
    @Volatile private var running: Boolean = false

    fun extract(turnText: String) {
        if (turnText.isBlank()) return
        pendingText = turnText
        // Cancel any pending debounce, start a new one
        debounceJob?.cancel()
        debounceJob = scope.launch {
            delay(DEBOUNCE_MS)
            val text = pendingText ?: return@launch
            pendingText = null
            if (running) return@launch
            running = true
            try {
                val (nodes, edges) = knowledgeGraphTool.extract(text)
                if (nodes.isNotEmpty() || edges.isNotEmpty()) {
                    repository.saveGraph(nodes, edges, "turn-${System.currentTimeMillis()}")
                }
            } catch (_: Exception) {
                // Best effort: do not crash the chat stream.
            } finally {
                running = false
            }
        }
    }

    companion object {
        private const val DEBOUNCE_MS = 5_000L
    }
}
