package com.aura

import android.net.Uri
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Tiny singleton that holds the most recent text or image shared into
 * the app via the share-sheet. MainActivity writes here, the ChatScreen
 * reads on first composition and clears the slot.
 *
 * Single-slot ring buffer (no history) — sharing is a one-shot "do
 * something with this" gesture, not a transcript.
 *
 * Images are passed as a [Uri] rather than pre-decoded base64 so the
 * chat screen can decode the Bitmap at display resolution and route
 * it through [ChatViewModel.onImageCaptured] for vision analysis,
 * instead of dumping a wall of base64 text into the chat input.
 */
@Singleton
class IncomingShareStore @Inject constructor() {
    private val _pending = MutableStateFlow<String?>(null)
    val pending: StateFlow<String?> = _pending.asStateFlow()

    private val _pendingImageUri = MutableStateFlow<Uri?>(null)
    val pendingImageUri: StateFlow<Uri?> = _pendingImageUri.asStateFlow()

    fun set(text: String) {
        _pending.value = text
    }

    fun setImageUri(uri: Uri) {
        _pendingImageUri.value = uri
    }

    /** Atomically read + clear, so a stale share doesn't get re-applied. */
    fun consume(): String? {
        val v = _pending.value
        _pending.value = null
        return v
    }

    /** Atomically read + clear the image URI. */
    fun consumeImageUri(): Uri? {
        val v = _pendingImageUri.value
        _pendingImageUri.value = null
        return v
    }
}