package com.aura.agent

import com.aura.providers.ProviderMessage
import com.aura.tools.Citation
import kotlinx.serialization.Serializable
import java.util.UUID

/**
 * One conversation: a list of turns. Mirrors aura/core/conversation_manager.py
 * minus the persistence (Room-backed ConversationStore in module 3).
 *
 * The turn list is immutable at the public API surface. Mutator methods return
 * a new [Conversation] so callers holding a snapshot (e.g. a Compose StateFlow
 * value) cannot race with concurrent updates. This also makes StateFlow equality
 * checks stable: a new turn produces a new [Conversation] reference, so
 * collectors recompose exactly once per meaningful change.
 */
@Serializable
data class Conversation(
    val id: String = UUID.randomUUID().toString(),
    val title: String = "New conversation",
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
    val systemPrompt: String? = null,
    val turns: List<Turn> = emptyList(),
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

    /** Append a user turn. */
    fun addUser(text: String): Conversation = copy(
        turns = turns + Turn(user = text),
        updatedAt = System.currentTimeMillis(),
    )

    /** Append or fill in an assistant turn. */
    fun addAssistant(text: String): Conversation {
        if (turns.isEmpty() || turns.last().assistant != null || turns.last().user == null) {
            return copy(turns = turns + Turn(assistant = text), updatedAt = System.currentTimeMillis())
        }
        return replaceLastTurn(turns.last().copy(assistant = text))
    }

    /** Append a tool call to the current turn. */
    fun addToolCall(id: String, name: String, args: String): Conversation {
        if (turns.isEmpty()) return copy(turns = listOf(Turn(toolTurns = listOf(ToolTurn(id, name, args, "")))))
        val last = turns.last()
        return replaceLastTurn(last.copy(toolTurns = last.toolTurns + ToolTurn(id, name, args, "")))
    }

    /** Set the result for a tool call on the current turn. */
    fun setToolResult(id: String, result: String): Conversation {
        if (turns.isEmpty()) return this
        val last = turns.last()
        val updatedToolTurns = last.toolTurns.map {
            if (it.id == id) it.copy(result = result) else it
        }
        return replaceLastTurn(last.copy(toolTurns = updatedToolTurns))
    }

    /**
     * Attach citations to the current turn (e.g. from web/research tools).
     * Replaces any existing citations on the last turn.
     */
    fun setCitations(citations: List<Citation>): Conversation {
        if (turns.isEmpty()) return this
        return replaceLastTurn(turns.last().copy(citations = citations))
    }

    /** Replace the last turn, used by callers streaming partial assistant text. */
    fun replaceLastTurn(newLast: Turn): Conversation = copy(
        turns = turns.dropLast(1) + newLast,
        updatedAt = System.currentTimeMillis(),
    )
}

@Serializable
data class Turn(
    val user: String? = null,
    val assistant: String? = null,
    val toolTurns: List<ToolTurn> = emptyList(),
    val citations: List<Citation> = emptyList(),
)

@Serializable
data class ToolTurn(
    val id: String,
    val name: String,
    val args: String,
    val result: String,
)
