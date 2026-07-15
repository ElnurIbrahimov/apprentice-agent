package com.aura.agent

/**
 * Simple keyword-based router that selects a [Specialist] based on the user's
 * query and the set of tools they currently have configured.
 *
 * Returns `null` (→ General specialist) unless strong keyword matches are found.
 */
object SpecialistRouter {

    /**
     * Pick the most appropriate specialist for a user query.
     *
     * @param query The user's raw input text.
     * @return the best specialist or null if General should handle it.
     */
    fun pickSpecialist(userMessage: String): Specialist? {
        val lower = userMessage.lowercase()

        // Order matters: check the most specific categories first so they
        // take priority over more generic ones.

        // ---- Phone-native (Android device actions) ----
        if (matchesAnyKeyword(lower, setOf(
                // Camera / gallery
                "camera", "gallery", "take a photo", "take a picture",
                "take photo", "take picture", "snap a photo", "snap a pic",
                "my photos", "show photos", "show picture",
                // Location
                "location", "where am i", "my location", "current location", "gps",
                // Reminders / alarms
                "reminder", "remind me", "alarm", "timer",
                // App launch
                "launch", "open app", "open the app", "start app", "open ",
                // Notifications
                "notification", "notify me", "alert",
                // Device state
                "screenshot", "display",
                "battery", "volume", "wifi", "bluetooth", "airplane mode",
                "dnd", "do not disturb", "flashlight", "torch",
            ))
            && !isQueryAbout(lower,
                "code", "program", "debug", "kotlin", "python",
                "research", "history", "image", "art", "artwork",
                "design", "logo", "poster", "banner",
                "describe", "explain", "tell", "about",
            )
        ) {
            return Specialist.PhoneNative
        }

        // ---- Writer (fiction, worldbuilding, narrative simulation) ----
        if (matchesAnyKeyword(lower, setOf(
                "write", "writing", "story", "novel", "chapter", "scene",
                "dialogue", "dialog", "character", "worldbuild", "worldbuilding",
                "world bible", "plot", "outline", "fiction", "prose", "narrative",
                "magic system", "lore", "screenplay", "what-if", "what if",
                "simulation", "continuity", "retcon",
            ))
            && !isQueryAbout(
                lower,
                "code", "program", "debug", "kotlin", "python", "research",
                "image", "photo", "art", "artwork", "design", "logo", "poster", "banner",
            )
        ) {
            return Specialist.Writer
        }

        // ---- Creative (image generation, art, vision) ----
        if (matchesAnyKeyword(lower, setOf(
                "image", "draw", "photo of", "picture of", "create art",
                "generate", "make a", "paint", "sketch", "illustrate",
                "art", "artwork", "design", "poster", "meme",
                "vision", "what's in this", "what is this", "describe this",
                "cartoon", "comic", "logo", "banner",
            ))
            && !isQueryAbout(lower, "code", "program", "debug", "research")
        ) {
            return Specialist.Creative
        }

        // ---- Executive (calendar, contacts, tasks, memory) ----
        if (matchesAnyKeyword(lower, setOf(
                "meeting", "calendar", "event", "schedule", "appointment",
                "contact", "contacts", "phonebook", "address book",
                "task", "tasks", "todo", "to-do", "remind me to",
                "remember", "recall", "my name", "my address", "my phone",
                "deadline", "deadlines", "due date", "plan",
            ))
        ) {
            return Specialist.Executive
        }

        // ---- Coder (programming, debugging) ----
        if (matchesAnyKeyword(lower, setOf(
                "code", "debug", "kotlin", "python", "gradle",
                "program", "programming", "compile", "build",
                "algorithm", "function", "class", "method",
                "refactor", "review", "pull request", "pr ",
                "syntax", "error", "exception", "stack trace",
                "java", "javascript", "typescript", "rust", "go ",
                "android", "jetpack", "compose", "xml layout",
                "unit test", "integration test", "test case",
            ))
        ) {
            return Specialist.Coder
        }

        // ---- Researcher (deep research, web search, citations) ----
        if (matchesAnyKeyword(lower, setOf(
                "research", "deep research", "deep_research",
                "web search", "search for", "search the web",
                "find", "look up", "who is", "what is",
                "cite", "citation", "source", "references",
                "tell me about", "explain", "how does",
                "history of", "news about", "latest",
                "compare", "analysis", "summary of",
                "documentation", "docs", "tutorial",
                "when did", "where is", "define",
            ))
        ) {
            return Specialist.Researcher
        }

        // No strong match → General fallback
        return null
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /**
     * Check whether the lowercased query contains any of the given keywords.
     *
     * Uses word-boundary regex matching for short keywords (≤ 5 characters)
     * to avoid false positives (e.g. "art" should not match "article").
     * For longer keywords simple substring matching is used so that
     * e.g. "deadline" still matches "deadlines".
     *
     * Keywords ending with a space (e.g. "open ", "pr ") are trimmed
     * and always matched via word boundaries.
     */
    private fun matchesAnyKeyword(lower: String, keywords: Set<String>): Boolean {
        return keywords.any { kw ->
            val word = kw.trim()
            val useWordBoundary = kw.endsWith(" ") || word.length <= 5
            if (useWordBoundary) {
                val escaped = Regex.escape(word)
                Regex("\\b$escaped\\b", RegexOption.IGNORE_CASE).containsMatchIn(lower)
            } else {
                lower.contains(word)
            }
        }
    }

    /**
     * Check whether the already-lowercased query mentions any of the given
     * categories — used to avoid false-positive routing (e.g. "tell me about
     * the history of photography" should NOT route to PhoneNative).
     */
    private fun isQueryAbout(lower: String, vararg categories: String): Boolean {
        return categories.any { cat ->
            matchesAnyKeyword(lower, setOf(cat))
        }
    }
}
