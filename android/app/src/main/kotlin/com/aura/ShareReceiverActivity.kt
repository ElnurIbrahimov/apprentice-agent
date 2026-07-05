package com.aura

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.core.content.IntentCompat
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class ShareReceiverActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val payload = extractPayload(intent)
        if (payload.text.isNullOrBlank() && payload.imageUri == null) {
            finish()
            return
        }
        val main = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            payload.text?.let { putExtra(EXTRA_SHARED_TEXT, it) }
            payload.imageUri?.let { putExtra(EXTRA_SHARED_IMAGE_URI, it) }
        }
        startActivity(main)
        finish()
    }

    private data class SharePayload(val text: String?, val imageUri: Uri?)

    private fun extractPayload(intent: Intent?): SharePayload {
        if (intent == null) return SharePayload(null, null)
        if (intent.action != Intent.ACTION_SEND) return SharePayload(null, null)

        // Image share: pass the URI directly so ChatScreen can decode
        // the Bitmap at display resolution and route it through
        // onImageCaptured() for vision analysis — no base64 wall.
        val imageUri: Uri? = IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java)
        if (imageUri != null && contentResolver.getType(imageUri)?.startsWith("image/") == true) {
            return SharePayload(text = null, imageUri = imageUri)
        }

        // Text share
        val text = intent.getStringExtra(Intent.EXTRA_TEXT) ?: ""
        val subject = intent.getStringExtra(Intent.EXTRA_SUBJECT) ?: ""
        val combined = if (subject.isNotBlank() && text.isNotBlank()) "$subject\n\n$text"
        else text.ifBlank { subject }
        return SharePayload(
            text = combined.ifBlank { null },
            imageUri = null,
        )
    }

    companion object {
        const val EXTRA_SHARED_TEXT = "com.aura.SHARED_TEXT"
        const val EXTRA_SHARED_IMAGE_URI = "com.aura.SHARED_IMAGE_URI"
    }
}
