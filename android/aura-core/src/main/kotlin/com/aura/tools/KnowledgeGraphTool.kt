package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.kg.EdgeType
import com.aura.kg.KgEdge
import com.aura.kg.KgId
import com.aura.kg.KgNode
import com.aura.kg.NodeType
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.ResponseFormat
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.coroutines.flow.toList
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Tool that extracts a knowledge graph (nodes + edges) from unstructured text
 * by calling a cloud LLM with a structured JSON prompt.
 *
 * Risk: READ_ONLY (network egress only).
 */
@Singleton
class KnowledgeGraphTool @Inject constructor(
    private val providerRegistry: ProviderRegistry,
) {
    private val json = Json { ignoreUnknownKeys = true }

    private val allowedNodeTypes = NodeType.entries.map { it.name.lowercase() }
    private val allowedEdgeTypes = EdgeType.entries.map { it.name.lowercase() }

    fun definition() = ToolParameters(
        properties = mapOf(
            "text" to ToolProperty(
                type = "string",
                description = "The text to extract knowledge graph nodes and edges from",
            ),
        ),
        required = listOf("text"),
    )

    val tool = Tool(
        name = "knowledge_graph_extract",
        description = "Extract a knowledge graph (nodes and edges) from unstructured text. Returns JSON with nodes and edges.",
        risk = ToolRisk.READ_ONLY,
        parameters = definition(),
        execute = { call, _ ->
            val text = call.arguments["text"] as? String
                ?: return@Tool ToolResult.Error("missing 'text' argument", "bad_args")

            try {
                val (nodes, edges) = extract(text)
                ToolResult.Ok(buildJsonResult(nodes, edges))
            } catch (e: Exception) {
                ToolResult.Error("Knowledge graph extraction failed: ${e.message}", "extraction_error")
            }
        },
    category = "knowledge")
    suspend fun extract(text: String): Pair<List<KgNode>, List<KgEdge>> {
        val response = callLlm(text)
        return parseResponse(response) ?: Pair(emptyList(), emptyList())
    }

    fun buildJsonResult(nodes: List<KgNode>, edges: List<KgEdge>): String {
        val nodesJson = nodes.joinToString(",", "[", "]") { node -> formatNodeJson(node) }
        val edgesJson = edges.joinToString(",", "[", "]") { edge -> formatEdgeJson(edge) }
        return """{"nodes":$nodesJson,"edges":$edgesJson}"""
    }

    /**
     * Build the extraction prompt and call the provider.
     */
    private suspend fun callLlm(text: String): String {
        val systemPrompt = buildString {
            appendLine("You are a knowledge graph extraction assistant.")
            appendLine("Extract nodes and edges from the given text as a JSON object.")
            appendLine()
            appendLine("Allowed node types: ${allowedNodeTypes.joinToString(", ")}")
            appendLine("Allowed edge types: ${allowedEdgeTypes.joinToString(", ")}")
            appendLine()
            appendLine("Return ONLY valid JSON with this exact structure:")
            appendLine("""{"nodes":[{"label":"...","type":"...","properties":{...}}],"edges":[{"type":"...","source_label":"...","target_label":"...","weight":0.5,"properties":{...}}]}""")
            appendLine()
            appendLine("Rules:")
            appendLine("- 'label' is a concise name for the node.")
            appendLine("- 'type' must be one of the allowed node types (default to 'unknown' if unsure).")
            appendLine("- 'properties' is optional (omit for empty).")
            appendLine("- 'source_label' and 'target_label' reference node labels from the nodes array.")
            appendLine("- 'type' for edges must be one of the allowed edge types (default to 'relates_to' if unsure).")
            appendLine("- 'weight' is optional (float 0.0-1.0, omit for default).")
            appendLine("- If no entities are found, return {\"nodes\":[],\"edges\":[]}.")
            appendLine("- Do NOT wrap in markdown code fences.")
        }

        val userPrompt = "Extract a knowledge graph from this text:\n\n$text"

        val messages = listOf(
            ProviderMessage(role = ProviderMessage.Role.system, content = systemPrompt),
            ProviderMessage(role = ProviderMessage.Role.user, content = userPrompt),
        )

        val options = ChatOptions(
            temperature = 0.2,
            maxTokens = 4096,
            responseFormat = ResponseFormat.JSON,
        )

        // Pick a model to drive the extraction. We try the user's default
        // model first (resolved by ProviderRegistry's parse("default")
        // which routes to the first configured provider's first model).
        // If that throws (no configured providers at all), we fall back
        // to any configured provider so the tool still works once the
        // user has set up at least one key. We do NOT hardcode a
        // specific provider:model — that was the 2026-07-07 bug where
        // ollama:deepseek-v4-pro was baked in and crashed on users with
        // no Ollama key.
        val flow = runCatching {
            providerRegistry.chat("default", messages, options)
        }.getOrElse {
            val fallback = providerRegistry.configured()
                .firstOrNull()
                ?.let { p ->
                    val first = p.listModels().firstOrNull() ?: return@let null
                    "${p.prefix}:$first"
                }
                ?: throw IllegalStateException("No configured providers for knowledge graph extraction")
            providerRegistry.chat(fallback, messages, options)
        }

        val chunks = flow.toList()
        return chunks.filter { it.text != null }.joinToString("") { it.text!! }
            .trim()
            .removeSurrounding("```json", "```")
            .removeSurrounding("```", "```")
            .trim()
    }

    /**
     * Parse the LLM response into lists of nodes and edges.
     * Returns null on parse failure (caller falls back to empty result).
     */
    private data class ParsedEdge(
        val type: String,
        val sourceLabel: String,
        val targetLabel: String,
        val weight: Float?,
        val properties: JsonObject?,
    )

    private fun parseResponse(response: String): Pair<List<KgNode>, List<KgEdge>>? {
        val root = try {
            json.parseToJsonElement(response).jsonObject
        } catch (_: Exception) {
            return null
        }

        val nodesJson = root["nodes"]?.jsonArray ?: return null
        val edgesJson = root["edges"]?.jsonArray ?: return null

        val nodes = nodesJson.mapNotNull { el -> parseNode(el) }
        val edges = edgesJson.mapNotNull { el -> parseEdge(el, nodes) }

        return Pair(nodes, edges)
    }

    private fun parseNode(el: JsonElement): KgNode? {
        val obj = el.jsonObject
        val label = obj["label"]?.jsonPrimitive?.contentOrNull?.trim() ?: return null
        if (label.isBlank()) return null

        val typeStr = obj["type"]?.jsonPrimitive?.contentOrNull ?: "unknown"
        val type = NodeType.from(typeStr)
        val properties = obj["properties"]?.jsonObject ?: JsonObject(emptyMap())

        val id = KgId.node(type, label)
        return KgNode(
            id = id,
            label = label,
            type = type,
            properties = properties,
        )
    }

    private fun parseEdge(el: JsonElement, nodes: List<KgNode>): KgEdge? {
        val obj = el.jsonObject
        val typeStr = obj["type"]?.jsonPrimitive?.contentOrNull ?: "relates_to"
        val edgeType = EdgeType.from(typeStr)
        if (edgeType == EdgeType.UNKNOWN && typeStr != "unknown") return null

        val sourceLabel = obj["source_label"]?.jsonPrimitive?.contentOrNull?.trim() ?: return null
        val targetLabel = obj["target_label"]?.jsonPrimitive?.contentOrNull?.trim() ?: return null

        if (sourceLabel.isBlank() || targetLabel.isBlank()) return null

        // Resolve source/target labels to node IDs
        val sourceNode = nodes.firstOrNull { it.label.equals(sourceLabel, ignoreCase = true) }
            ?: return null
        val targetNode = nodes.firstOrNull { it.label.equals(targetLabel, ignoreCase = true) }
            ?: return null

        val weight = obj["weight"]?.jsonPrimitive?.doubleOrNull?.toFloat()?.coerceIn(0f, 1f) ?: 0.5f
        val properties = obj["properties"]?.jsonObject ?: JsonObject(emptyMap())

        val id = KgId.edge(edgeType, sourceNode.id, targetNode.id)
        return KgEdge(
            id = id,
            type = edgeType,
            sourceId = sourceNode.id,
            targetId = targetNode.id,
            weight = weight,
            properties = properties,
        )
    }

    private fun formatNodeJson(node: KgNode): String {
        val props = if (node.properties.isEmpty()) "" else ",${node.properties.toString()}"
        return """{"id":"${escapeJson(node.id)}","label":"${escapeJson(node.label)}","type":"${node.type.name.lowercase()}"$props}"""
    }

    private fun formatEdgeJson(edge: KgEdge): String {
        val props = if (edge.properties.isEmpty()) "" else ",${edge.properties.toString()}"
        return """{"id":"${escapeJson(edge.id)}","type":"${edge.type.name.lowercase()}","source_id":"${escapeJson(edge.sourceId)}","target_id":"${escapeJson(edge.targetId)}","weight":${edge.weight}$props}"""
    }

    private fun escapeJson(s: String): String = s
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
}
