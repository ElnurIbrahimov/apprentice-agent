package com.aura.agent

/**
 * Defines a specialist agent persona that can be selected based on user intent.
 * Each specialist has its own identity (system prompt), preferred tool set,
 * and an optional suggested model override.
 */
data class Specialist(
    val name: String,
    val icon: String,
    val systemPrompt: String,
    val toolsAllowed: Set<String> = emptySet(),
    val suggestedModel: String? = null,
) {
    companion object {
        /** General-purpose assistant – the default fallback. */
        val General = Specialist(
            name = "general",
            icon = "\uD83E\uDD16", // 🤖
            systemPrompt = """
                You are Aura's general-purpose agent. You handle anything the
                user asks — casual conversation, questions, tool use, multi-step
                tasks. Be concise, direct, and helpful.
            """.trimIndent(),
        )

        /** Coding & debugging specialist. */
        val Coder = Specialist(
            name = "coder",
            icon = "\uD83D\uDCBB", // 💻
            systemPrompt = """
                You are Aura's coding specialist. You excel at writing, reviewing,
                and debugging Kotlin, Python, Gradle, and other programming
                languages. You can search the web for docs and solutions, then
                apply them. Be precise — explain the fix, not just the code.
            """.trimIndent(),
            toolsAllowed = setOf("brave_search", "tavily_search", "fetch_url"),
        )

        /** Deep-research & fact-finding specialist. */
        val Researcher = Specialist(
            name = "researcher",
            icon = "\uD83D\uDD0D", // 🔍
            systemPrompt = """
                You are Aura's research specialist. Your job is to find, synthesise,
                and cite information from the web. Use deep research for complex
                multi-source topics; use web search for quick facts. Always cite
                sources and present a balanced view when sources disagree.
            """.trimIndent(),
            toolsAllowed = setOf("deep_research", "brave_search", "tavily_search"),
        )

        /** Creative & image-generation specialist. */
        val Creative = Specialist(
            name = "creative",
            icon = "\uD83C\uDFA8", // 🎨
            systemPrompt = """
                You are Aura's creative specialist. You help users generate images,
                describe visual ideas, and give feedback on photos and artwork.
                Be playful, enthusiastic, and visually descriptive. When generating
                images, craft detailed, evocative prompts.
            """.trimIndent(),
            toolsAllowed = setOf("image_gen"),
        )

        /** Executive assistant — calendar, contacts, tasks, memory. */
        val Executive = Specialist(
            name = "executive",
            icon = "\uD83D\uDCC5",
            systemPrompt = """
                You are Aura's executive assistant. You manage the user's calendar,
                contacts, tasks, and personal memory. Be brisk and efficient —
                get in, make the change, confirm, get out. When reading data present
                it clearly; when writing confirm what you did.
            """.trimIndent(),
            toolsAllowed = setOf("calendar_read", "calendar_write", "contacts_search", "remember", "recall"),
        )
        val PhoneNative = Specialist(
            name = "phone_native",
            icon = "\uD83D\uDCF1", // 📱
            systemPrompt = """
                You are Aura's phone-native specialist. You know Android inside out —
                you can capture photos, browse the gallery, check the user's location,
                set reminders, launch apps, read and post notifications, and query
                device state (battery, network, volume, DND). Be quick and practical.
            """.trimIndent(),
            toolsAllowed = setOf(
                "photo_library", "location_now", "set_reminder",
                "launch_app", "notification_list", "post_notification",
                "battery_state", "network_state", "system_volume", "dnd_mode",
            ),
        )

        /** All predefined specialists, keyed by name. */
        val ALL: List<Specialist> = listOf(General, Coder, Researcher, Creative, Executive, PhoneNative)

        /** Lookup by name. */
        fun byName(name: String): Specialist? = ALL.find { it.name == name }

        /**
         * Apply user overrides to the built-in specialists. Returns
         * the specialist with its systemPrompt replaced if an
         * override exists in the map, otherwise the original.
         */
        fun applyOverrides(overrides: Map<String, String>): List<Specialist> =
            ALL.map { s ->
                val custom = overrides[s.name]
                if (custom.isNullOrBlank()) s
                else s.copy(systemPrompt = custom)
            }

        /**
         * Apply user-defined tool overrides. Returns specialists with
         * their toolsAllowed replaced if an override exists in the map.
         * Empty map or missing key = keep the default toolsAllowed.
         */
        fun applyToolOverrides(overrides: Map<String, Set<String>>): List<Specialist> =
            ALL.map { s ->
                val customTools = overrides[s.name]
                if (customTools == null || customTools.isEmpty()) s
                else s.copy(toolsAllowed = customTools)
            }
    }
}
