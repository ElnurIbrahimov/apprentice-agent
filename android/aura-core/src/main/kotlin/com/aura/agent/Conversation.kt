package com.aura.agent

import com.aura.providers.ProviderMessage
import kotlinx.serialization.Serializable
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

    /**
     * Append a user turn. Note: updatedAt is a data class val so we can't
     * reassign it in place; if you need to track recency, copy the conversation.
     * Keeping the mutator pattern here because the alternative (returning a
     * new Conversation per turn) balloons allocations on long sessions.
     */
    fun addUser(text: String) { turns += Turn(user = text) }
    fun addAssistant(text: String) {
        if (turns.isEmpty() || turns.last().assistant != null || turns.last().user == null) {
            turns += Turn(assistant = text)
        } else {
            turns[turns.lastIndex] = turns.last().copy(assistant = text)
        }
    }
    fun addToolCall(id: String, name: String, args: String) {
        if (turns.isEmpty()) turns += Turn()
        val last = turns.lastIndex
        turns[last] = turns.last().copy(toolTurns = turns[last].toolTurns + ToolTurn(id, name, args, ""))
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
        turns[last] = turns.last().copy(toolTurns = toolTurns)
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
