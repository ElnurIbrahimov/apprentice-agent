package com.aura.ui.voice

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.voice.SpeechToText

@Composable
fun VoiceOverlay(
    viewModel: VoiceViewModel = hiltViewModel(),
    onTranscript: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    val state by viewModel.sttState.collectAsState()
    val partial = when (val s = state) {
        is SpeechToText.State.PartialResult -> s.text
        is SpeechToText.State.FinalResult -> s.text
        else -> ""
    }

    LaunchedEffect(Unit) {
        viewModel.start()
    }

    LaunchedEffect(state) {
        when (val s = state) {
            is SpeechToText.State.FinalResult -> {
                if (s.text.isNotBlank()) onTranscript(s.text)
                viewModel.reset()
                onDismiss()
            }
            is SpeechToText.State.Error -> {
                viewModel.reset()
                onDismiss()
            }
            else -> Unit
        }
    }

    Surface(
        color = MaterialTheme.colorScheme.background,
        modifier = Modifier.fillMaxSize(),
    ) {
        Column(
            modifier = Modifier.fillMaxSize().padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            val transition = rememberInfiniteTransition(label = "mic-pulse")
            val scale by transition.animateFloat(
                initialValue = 1.0f,
                targetValue = 1.3f,
                animationSpec = infiniteRepeatable(tween(800), RepeatMode.Reverse),
                label = "pulse-scale",
            )
            Box(
                modifier = Modifier
                    .size(120.dp)
                    .scale(scale)
                    .background(MaterialTheme.colorScheme.error, CircleShape),
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    imageVector = Icons.Filled.Mic,
                    contentDescription = "Listening",
                    tint = MaterialTheme.colorScheme.onError,
                    modifier = Modifier.size(56.dp),
                )
            }
            Spacer(modifier = Modifier.height(32.dp))
            Text(
                text = statusText(state),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = partial.ifBlank { "Speak now…" },
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = if (partial.isBlank()) 0.5f else 0.9f),
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            )
            Spacer(modifier = Modifier.height(48.dp))
            IconButton(
                onClick = { viewModel.stop() },
                modifier = Modifier
                    .size(72.dp)
                    .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape),
            ) {
                Icon(
                    imageVector = Icons.Filled.Stop,
                    contentDescription = "Stop",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(32.dp),
                )
            }
        }
    }
}

private fun statusText(state: SpeechToText.State): String = when (state) {
    SpeechToText.State.Idle -> "Starting…"
    SpeechToText.State.Ready -> "Listening…"
    SpeechToText.State.Listening -> "Listening…"
    is SpeechToText.State.PartialResult -> "Heard you…"
    is SpeechToText.State.FinalResult -> "Got it"
    is SpeechToText.State.Error -> "Couldn't catch that"
}
