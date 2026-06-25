package com.aura

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Tiny singleton that holds the most recent text shared into the app via
 * the share-sheet. MainActivity writes here, the ChatScreen reads on
 * first composition and clears the slot.
 *
 * Single-slot ring buffer (no history) — sharing is a one-shot "do
 * something with this" gesture, not a transcript.
 */
@Singleton
class IncomingShareStore @Inject constructor() {
    private val _pending = MutableStateFlow<String?>(null)
    val pending: StateFlow<String?> = _pending.asStateFlow()

    fun set(text: String) {
        _pending.value = text
    }

    /** Atomically read + clear, so a stale share doesn't get re-applied. */
    fun consume(): String? {
        val v = _pending.value
        _pending.value = null
        return v
    }
}
