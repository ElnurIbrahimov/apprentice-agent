package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.kg.KnowledgeGraphRepository
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.coroutines.withTimeout
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Tool that queries the knowledge graph repository with a natural-language query.
 *
 * Heuristic dispatch:
 *   - "path between X and Y"  -> findPath
 *   - "what do I know about X" or "show X" -> search by X
 *   - else -> search and return top results
 *
 * Risk: READ_ONLY (reads local database only).
 */
@Singleton
class KgQueryTool @Inject constructor(
    private val repository: KnowledgeGraphRepository,
) {
    // "path between X and Y" — X and Y are arbitrary labels, which can
    // themselves contain the word "and" (e.g. "research and development").
    // The non-greedy `(.+?)\s+and\s+` capture for the source leaves the
    // destination to match greedily to the end. This means a query like
    // "path between research and development and shipping" routes as
    // fromLabel="research and development", toLabel="shipping" — which is
    // the most useful interpretation. A query like "path between A and B
    // and C" routes as fromLabel="A", toLabel="B and C" — also a sane
    // choice, and one we can't disambiguate without knowing the user's
    // intent.
    private val pathPattern = Regex(
        """path\s+between\s+(.+?)\s+and\s+(.+)""",
        RegexOption.IGNORE_CASE,
    )

    fun definition() = ToolParameters(
        properties = mapOf(
            "query" to ToolProperty(
                type = "string",
                description = "Query to search the knowledge graph, e.g. 'what do I know about Kotlin' or 'path between Aura and Android'",
            ),
        ),
        required = listOf("query"),
    )

    val tool = Tool(
        name = "kg_query",
        description = "Query the knowledge graph. Returns relevant nodes, edges, or paths as markdown.",
        risk = ToolRisk.READ_ONLY,
        parameters = definition(),
        execute = { call, _ ->
            val query = call.arguments["query"] as? String
                ?: return@Tool ToolResult.Error("missing 'query' argument", "bad_args")

            try {
                val result = withTimeout(15_000L) { runQuery(query) }
                ToolResult.Ok(result)
            } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
                ToolResult.Error("Query timed out", "timeout")
            } catch (e: Exception) {
                ToolResult.Error("Query failed: ${e.message}", "query_error")
            }
        },
    category = "knowledge")
    private suspend fun runQuery(query: String): String {
        // 1. Check for "path between X and Y"
        val pathMatch = pathPattern.find(query.trim())
        if (pathMatch != null) {
            val fromLabel = pathMatch.groupValues[1].trim()
            val toLabel = pathMatch.groupValues[2].trim()
            return queryPath(fromLabel, toLabel)
        }

        // 2. Check for "what do I know about X" or "show X"
        val knownAboutMatch = Regex(
            """(?:what\s+do\s+I\s+know\s+about|show|tell\s+me\s+about|search\s+for?)\s+(.+)""",
            RegexOption.IGNORE_CASE,
        ).find(query.trim())

        val searchTerm = if (knownAboutMatch != null) {
            knownAboutMatch.groupValues[1].trim()
        } else {
            query.trim()
        }

        return querySearch(searchTerm)
    }

    private suspend fun queryPath(fromLabel: String, toLabel: String): String {
        val fromNode = repository.getNodeByLabel(fromLabel)
        val toNode = repository.getNodeByLabel(toLabel)

        if (fromNode == null && toNode == null) {
            return "I couldn't find nodes matching **\"$fromLabel\"** or **\"$toLabel\"** in the knowledge graph."
        }
        if (fromNode == null) {
            return "I couldn't find a node matching **\"$fromLabel\"** in the knowledge graph."
        }
        if (toNode == null) {
            return "I couldn't find a node matching **\"$toLabel\"** in the knowledge graph."
        }

        val pathIds = repository.findPath(fromNode.id, toNode.id)
        if (pathIds.isEmpty()) {
            return "No path found between **${fromNode.label}** and **${toNode.label}**."
        }

        val pathLabels = pathIds.map { id ->
            val node = if (id == fromNode.id) fromNode
            else if (id == toNode.id) toNode
            else repository.getNode(id)
            node?.label ?: id.take(8)
        }

        val sb = StringBuilder()
        sb.appendLine("### Path from **${fromNode.label}** to **${toNode.label}**")
        sb.appendLine()
        for (i in pathLabels.indices) {
            sb.append("${i + 1}. ${pathLabels[i]}")
            if (i < pathLabels.size - 1) {
                // Look up edge between this node and next
                val nextId = pathIds[i + 1]
                val edges = repository.getNeighbors(pathIds[i])
                val connectingEdge = edges.outgoing.firstOrNull { it.targetId == nextId }
                if (connectingEdge != null) {
                    sb.append("  ──(${connectingEdge.type.name.lowercase()})──▶  ")
                } else {
                    sb.append("  ──▶  ")
                }
            } else {
                sb.appendLine()
            }
        }
        return sb.toString()
    }

    private suspend fun querySearch(searchTerm: String): String {
        // Try exact label match first
        val exactNode = repository.getNodeByLabel(searchTerm)
        if (exactNode != null) {
            val neighbors = repository.getNeighbors(exactNode.id)
            return formatNodeDetail(exactNode, neighbors)
        }

        // Fall back to search
        val results = repository.search(searchTerm)
        if (results.isEmpty()) {
            return "No results found in the knowledge graph for **\"$searchTerm\"**."
        }

        val sb = StringBuilder()
        sb.appendLine("### Search results for **\"$searchTerm\"**")
        sb.appendLine()
        sb.appendLine("| # | Label | Type |")
        sb.appendLine("|---|-------|------|")
        results.take(20).forEachIndexed { i, node ->
            sb.appendLine("| ${i + 1} | ${node.label} | ${node.type.name.lowercase()} |")
        }
        return sb.toString()
    }

    private suspend fun formatNodeDetail(
        node: com.aura.kg.KgNode,
        neighbors: KnowledgeGraphRepository.Neighbors,
    ): String {
        val sb = StringBuilder()
        sb.appendLine("### ${node.label}")
        sb.appendLine()
        sb.appendLine("- **Type:** ${node.type.name.lowercase()}")
        sb.appendLine("- **ID:** `${node.id.take(12)}...`")
        if (node.properties.isNotEmpty()) {
            sb.appendLine("- **Properties:**")
            for ((k, v) in node.properties) {
                sb.appendLine("  - $k: ${v}")
            }
        }
        sb.appendLine()

        if (neighbors.outgoing.isNotEmpty()) {
            sb.appendLine("#### Outgoing edges")
            sb.appendLine()
            sb.appendLine("| Type | Target | Weight |")
            sb.appendLine("|------|--------|--------|")
            for (edge in neighbors.outgoing) {
                val targetNode = try { repository.getNode(edge.targetId) } catch (_: Exception) { null }
                val targetLabel = targetNode?.label ?: edge.targetId.take(8)
                sb.appendLine("| ${edge.type.name.lowercase()} | $targetLabel | ${edge.weight} |")
            }
            sb.appendLine()
        }

        if (neighbors.incoming.isNotEmpty()) {
            sb.appendLine("#### Incoming edges")
            sb.appendLine()
            sb.appendLine("| Type | Source | Weight |")
            sb.appendLine("|------|--------|--------|")
            for (edge in neighbors.incoming) {
                val sourceNode = try { repository.getNode(edge.sourceId) } catch (_: Exception) { null }
                val sourceLabel = sourceNode?.label ?: edge.sourceId.take(8)
                sb.appendLine("| ${edge.type.name.lowercase()} | $sourceLabel | ${edge.weight} |")
            }
            sb.appendLine()
        }

        return sb.toString()
    }
}
