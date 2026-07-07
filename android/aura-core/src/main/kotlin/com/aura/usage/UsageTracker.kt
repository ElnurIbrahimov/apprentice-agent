package com.aura.usage

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Lightweight per-session usage tracker. Counts approximate tokens
 * (chars / 4) for all LLM input + output and tool fetch results.
 * Displayed in the chat header as "12K tokens" and in Settings as
 * cumulative usage. Reset per session.
 */
@Singleton
class UsageTracker @Inject constructor() {
    private val _sessionTokens = MutableStateFlow(0)
    val sessionTokens: StateFlow<Int> = _sessionTokens.asStateFlow()

    private val _sessionBytes = MutableStateFlow(0L)
    val sessionBytes: StateFlow<Long> = _sessionBytes.asStateFlow()

    /** Record an LLM call — both input and output characters. */
    fun recordLlmCall(inputChars: Int, outputChars: Int) {
        val tokens = (inputChars + outputChars) / 4
        _sessionTokens.value += tokens
        _sessionBytes.value += (inputChars + outputChars).toLong()
    }

    /** Record a tool fetch result (web page, search results, etc). */
    fun recordToolFetch(resultChars: Int) {
        _sessionBytes.value += resultChars.toLong()
    }

    /** Reset per session. Called when the app process restarts. */
    fun reset() {
        _sessionTokens.value = 0
        _sessionBytes.value = 0
    }

    /** Human-readable summary, e.g. "12.3K tokens · 45.6 KB". */
    fun summary(): String {
        val tokens = _sessionTokens.value
        val bytes = _sessionBytes.value
        val tokenStr = if (tokens >= 1000) "${"%.1f".format(tokens / 1000.0)}K" else "$tokens"
        val byteStr = if (bytes >= 1_000_000) "${"%.1f".format(bytes / 1_000_000.0)} MB" else "${"%.1f".format(bytes / 1000.0)} KB"
        return "$tokenStr tokens · $byteStr"
    }
}