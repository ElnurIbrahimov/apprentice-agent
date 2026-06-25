package com.aura.agent

import com.aura.providers.ProviderMessage
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.util.UUID

/**
 * One conversation: a list of turns. Mirrors aura/core/conversation_manager.py
 * minus the persistence (Room-backed ConversationStore in module 3).
 */
@Serializable
data class Conversation(
    val id: String = UUID.randomUUID().toString(),
    val title: String = "New conversation",
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
    val systemPrompt: String? = null,
    val turns: MutableList<Turn> = mutableListOf(),
    val model: String? = null,
    val metadata: Map<String, String> = emptyMap(),
) {
    fun toMessages(): List<ProviderMessage> {
        val out = mutableListOf<ProviderMessage>()
        val sys = listOfNotNull(systemPrompt).filter { it.isNotBlank() }
        if (sys.isNotEmpty()) {
            out += ProviderMessage(role = ProviderMessage.Role.system, content = sys.joinToString("\n\n"))
        }
        for (turn in turns) {
            turn.user?.let { out += ProviderMessage(role = ProviderMessage.Role.user, content = it) }
            turn.assistant?.let { out += ProviderMessage(role = ProviderMessage.Role.assistant, content = it) }
            for (toolTurn in turn.toolTurns) {
                out += ProviderMessage(role = ProviderMessage.Role.tool, content = toolTurn.result, toolCallId = toolTurn.id)
            }
        }
        return out
    }

    fun touch() { /* data class val updatedAt can't be reassigned without copy; use copy() in callers */ }
    fun addUser(text: String) { turns += Turn(user = text); touch() }
    fun addAssistant(text: String) {
        if (turns.isEmpty() || turns.last().assistant != null || turns.last().user == null) {
            turns += Turn(assistant = text)
        } else {
            turns[turns.lastIndex] = turns.last().copy(assistant = text)
        }
        touch()
    }
    fun addToolCall(id: String, name: String, args: String) {
        if (turns.isEmpty()) turns += Turn()
        val last = turns.lastIndex
        turns[last] = turns[last].copy(toolTurns = turns[last].toolTurns + ToolTurn(id, name, args, ""))
    }
    fun setToolResult(id: String, result: String) {
        if (turns.isEmpty()) return
        val last = turns.lastIndex
        val toolTurns = turns[last].toolTurns.toMutableList()
        for (i in toolTurns.indices) {
            if (toolTurns[i].id == id) {
                toolTurns[i] = toolTurns[i].copy(result = result)
            }
        }
        turns[last] = turns[last].copy(toolTurns = toolTurns)
    }
}

@Serializable
data class Turn(
    val user: String? = null,
    val assistant: String? = null,
    val toolTurns: List<ToolTurn> = emptyList(),
)

@Serializable
data class ToolTurn(
    val id: String,
    val name: String,
    val args: String,
    val result: String,
)

object JsonCodec {
    val pretty: Json = Json { prettyPrint = true; ignoreUnknownKeys = true; encodeDefaults = true }
    val strict: Json = Json { ignoreUnknownKeys = true; encodeDefaults = true; isLenient = true }
}
