package com.aura

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

        // If we were launched from a share intent, deposit the text in the
        // IncomingShareStore. ChatScreen consumes it on first composition.
        val sharedText = intent?.getStringExtra(ShareReceiverActivity.EXTRA_SHARED_TEXT)
        if (!sharedText.isNullOrBlank()) {
            incomingShareStore.set(sharedText)
        }

        setContent {
            AuraRoot()
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
