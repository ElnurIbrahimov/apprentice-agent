package com.aura.ui.viewmodel

import com.aura.agent.Conversation

internal data class PreparedConversationRetry(
    val conversation: Conversation,
    val userText: String,
)

/**
 * Rewind the conversation to its last nonblank user turn and clear only the
 * failed response artifacts. The user turn itself is retained, so retrying
 * cannot duplicate it in history or in the model prompt.
 */
internal fun prepareConversationForRetry(conversation: Conversation): PreparedConversationRetry? {
    val index = conversation.turns.indexOfLast { !it.user.isNullOrBlank() }
    if (index < 0) return null
    val userText = conversation.turns[index].user?.trim().orEmpty()
    if (userText.isEmpty()) return null

    val turns = conversation.turns.take(index + 1).toMutableList()
    turns[index] = turns[index].copy(
        assistant = null,
        toolTurns = emptyList(),
        citations = emptyList(),
        recall = null,
    )
    return PreparedConversationRetry(
        conversation = conversation.copy(
            turns = turns,
            summaryThroughTurn = conversation.summaryThroughTurn.coerceAtMost(turns.size),
            updatedAt = System.currentTimeMillis(),
        ),
        userText = userText,
    )
}
