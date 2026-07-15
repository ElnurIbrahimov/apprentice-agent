package com.aura.tools

/**
 * Tool categories used by the Tools browser screen. Stable strings —
 * never rename, or the user's grouped view will break. Add new
 * categories here as new tool families are introduced.
 *
 * Each tool in [com.aura.tools.ToolsModule] picks one of these via
 * its `category` field. Empty string = "Other" (catch-all for tools
 * that don't fit anywhere).
 */
object ToolCategories {
    const val MEMORY = "memory"
    const val WEB = "web"
    const val PRODUCTIVITY = "productivity"
    const val SYSTEM = "system"
    const val COMMUNICATION = "communication"
    const val MEDIA = "media"
    const val VISION = "vision"
    const val AUTOMATION = "automation"
    const val KNOWLEDGE = "knowledge"
    const val CREATIVE = "creative"
    const val SKILLS = "skills"
    const val DEVICE = "device"
    const val OTHER = "other"

    val ALL: List<String> = listOf(
        MEMORY, WEB, PRODUCTIVITY, SYSTEM, COMMUNICATION,
        MEDIA, VISION, AUTOMATION, KNOWLEDGE, CREATIVE, DEVICE, OTHER,
    )

    fun displayName(category: String): String = when (category) {
        MEMORY -> "Memory"
        WEB -> "Web & Search"
        PRODUCTIVITY -> "Productivity"
        SYSTEM -> "System"
        COMMUNICATION -> "Communication"
        MEDIA -> "Media"
        VISION -> "Vision"
        AUTOMATION -> "Automation"
        KNOWLEDGE -> "Knowledge"
        CREATIVE -> "Creative Studio"
        SKILLS -> "Skills"
        DEVICE -> "Device"
        else -> "Other"
    }

    fun icon(category: String): String = when (category) {
        MEMORY -> "🧠"
        WEB -> "🌐"
        PRODUCTIVITY -> "✅"
        SYSTEM -> "⚙️"
        COMMUNICATION -> "✉️"
        MEDIA -> "🎵"
        VISION -> "👁️"
        AUTOMATION -> "🪄"
        KNOWLEDGE -> "🔗"
        CREATIVE -> "✍️"
        SKILLS -> "📚"
        DEVICE -> "📱"
        else -> "🔧"
    }
}
