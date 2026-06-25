package com.aura

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import dagger.hilt.android.AndroidEntryPoint

/**
 * Receives ACTION_SEND intents from other apps (text, URLs) and routes
 * the shared text into the Aura chat. Listed in AndroidManifest.xml with
 * an intent-filter for android.intent.action.SEND + mimeType text/plain
 * so Aura appears in the system share sheet.
 *
 * This Activity is intentionally tiny: it does not render UI itself. It
 * extracts the shared text, launches MainActivity with the text as an
 * extra, then finishes. The translucent theme ensures no visual flash.
 */
@AndroidEntryPoint
class ShareReceiverActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val sharedText = extractSharedText(intent)
        val main = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            putExtra(EXTRA_SHARED_TEXT, sharedText)
        }
        startActivity(main)
        finish()
    }

    private fun extractSharedText(intent: Intent?): String {
        return when (intent?.action) {
            Intent.ACTION_SEND -> {
                val text = intent.getStringExtra(Intent.EXTRA_TEXT) ?: ""
                val subject = intent.getStringExtra(Intent.EXTRA_SUBJECT) ?: ""
                if (subject.isNotBlank() && text.isNotBlank()) "$subject\n\n$text" else text.ifBlank { subject }
            }
            Intent.ACTION_VIEW -> {
                intent.data?.toString() ?: ""
            }
            else -> ""
        }
    }

    companion object {
        const val EXTRA_SHARED_TEXT = "com.aura.SHARED_TEXT"
    }
}
