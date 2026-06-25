package com.aura

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraTheme
import dagger.hilt.android.AndroidEntryPoint

/**
 * Receives ACTION_SEND intents from other apps (text, URLs) and routes
 * the shared text into the Aura chat. Listed in AndroidManifest.xml with
 * an intent-filter for android.intent.action.SEND + mimeType text/plain
 * so Aura appears in the system share sheet.
 */
@AndroidEntryPoint
class ShareReceiverActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val sharedText = extractSharedText(intent)
        // Launch MainActivity with the shared text as an extra
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
