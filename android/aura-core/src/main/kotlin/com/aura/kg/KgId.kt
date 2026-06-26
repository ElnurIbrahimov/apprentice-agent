package com.aura.kg

import java.security.MessageDigest

object KgId {
    fun node(type: NodeType, label: String): String =
        sha256("kg|node|${type.name.lowercase()}|${label.lowercase().trim()}")

    fun edge(type: EdgeType, sourceId: String, targetId: String): String =
        sha256("kg|edge|${type.name.lowercase()}|$sourceId|$targetId")

    private fun sha256(input: String): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(input.toByteArray())
        return bytes.joinToString("") { "%02x".format(it) }
    }
}
