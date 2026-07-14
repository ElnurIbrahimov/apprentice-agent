package com.aura.provenance

/**
 * Stable locator for knowledge learned from a conversation turn.
 * Empty values mean the record came from a manual/system source rather than chat.
 */
data class ConversationProvenance(
    val conversationId: String = "",
    val turnTimestamp: Long = 0L,
) {
    val isPresent: Boolean
        get() = conversationId.isNotBlank() && turnTimestamp > 0L
}
