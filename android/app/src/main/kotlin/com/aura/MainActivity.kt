package com.aura

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import com.aura.ui.nav.NavGraph
import com.aura.ui.theme.AuraTheme
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject lateinit var incomingShareStore: IncomingShareStore

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.auto(Color.Transparent.value.toInt(), Color.Transparent.value.toInt()),
            navigationBarStyle = SystemBarStyle.auto(Color.Transparent.value.toInt(), Color.Transparent.value.toInt()),
        )
        super.onCreate(savedInstanceState)
        handleSharedText(intent)
        setContent {
            AuraRoot()
        }
    }

    /**
     * Handles the case where Aura is already in the background and the user
     * shares new text. Without this, [intent] would still be the original
     * launching intent (with no EXTRA_SHARED_TEXT) and the share would
     * silently disappear. We update the stored intent and route the new
     * shared text into the IncomingShareStore for ChatScreen to pick up.
     */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleSharedText(intent)
    }

    private fun handleSharedText(intent: Intent?) {
        val sharedText = intent?.getStringExtra(ShareReceiverActivity.EXTRA_SHARED_TEXT) ?: return
        if (sharedText.isNotBlank()) {
            incomingShareStore.set(sharedText)
        }
    }
}

@Composable
fun AuraRoot() {
    AuraTheme {
        Surface(
            modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
            color = MaterialTheme.colorScheme.background,
        ) {
            NavGraph()
        }
    }
}
