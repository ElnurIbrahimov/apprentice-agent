package com.aura

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Base64
import androidx.activity.ComponentActivity
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class ShareReceiverActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val payload = extractPayload(intent)
        if (payload.isBlank()) { finish(); return }
        val main = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            putExtra(EXTRA_SHARED_TEXT, payload)
        }
        startActivity(main)
        finish()
    }

    private fun extractPayload(intent: Intent?): String {
        if (intent == null) return ""
        return when (intent.action) {
            Intent.ACTION_SEND -> {
                val imageUri: Uri? = intent.getParcelableExtra(Intent.EXTRA_STREAM)
                if (imageUri != null) {
                    val base64 = try {
                        contentResolver.openInputStream(imageUri)?.use {
                            Base64.encodeToString(it.readBytes(), Base64.NO_WRAP)
                        }
                    } catch (_: Exception) { null }
                    if (!base64.isNullOrBlank()) {
                        val mimeType = contentResolver.getType(imageUri) ?: "image/*"
                        return "[Shared image: $mimeType]\n$base64"
                    }
                }
                val text = intent.getStringExtra(Intent.EXTRA_TEXT) ?: ""
                val subject = intent.getStringExtra(Intent.EXTRA_SUBJECT) ?: ""
                if (subject.isNotBlank() && text.isNotBlank()) "$subject\n\n$text"
                else text.ifBlank { subject }
            }
            else -> ""
        }
    }

    companion object {
        const val EXTRA_SHARED_TEXT = "com.aura.SHARED_TEXT"
    }
}
