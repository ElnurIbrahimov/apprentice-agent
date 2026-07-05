package com.aura.ui.screens

import android.content.Intent
import android.provider.CalendarContract
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.proactive.ProactiveEventBus
import com.aura.ui.util.toSummary
import com.aura.ui.viewmodel.ProactiveHistoryViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class HistoryCardModel(
    val icon: String,
    val title: String,
    val body: String,
    val tapHint: String?,
    val timestamp: Long,
    val onClick: (() -> Unit)?,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProactiveHistoryScreen(
    viewModel: ProactiveHistoryViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Proactive history") })
        },
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(state.events.reversed()) { event ->
                val model = event.toCardModel(context)
                HistoryCard(
                    model = model,
                    modifier = if (model.onClick != null) Modifier.clickable(onClick = model.onClick) else Modifier,
                )
            }
        }
    }
}

private fun ProactiveEventBus.Event.toCardModel(context: android.content.Context): HistoryCardModel = when (this) {
    is ProactiveEventBus.Event.MorningBriefReady -> HistoryCardModel(
        icon = "\u2600\uFE0F",
        title = "Morning brief",
        body = body,
        tapHint = "Tap to chat",
        timestamp = timestamp,
        onClick = {
            context.startActivity(
                Intent(context, com.aura.MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_REORDER_TO_FRONT
                    putExtra("openChat", true)
                }
            )
        },
    )
    is ProactiveEventBus.Event.MorningBriefStructured -> HistoryCardModel(
        icon = "\u2600\uFE0F",
        title = "Morning brief",
        body = this.context.toSummary(),
        tapHint = "Tap to chat",
        timestamp = timestamp,
        onClick = {
            context.startActivity(
                Intent(context, com.aura.MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_REORDER_TO_FRONT
                    putExtra("openChat", true)
                }
            )
        },
    )
    is ProactiveEventBus.Event.CalendarEventSoon -> {
        val minutes = minutesUntil
        val label = if (minutes < 60) "in $minutes min" else "in ${minutes / 60}h ${minutes % 60}m"
        HistoryCardModel(
            icon = "📅",
            title = "Upcoming: $title",
            body = label,
            tapHint = "Open calendar",
            timestamp = timestamp,
            onClick = {
                val intent = Intent(Intent.ACTION_VIEW).apply {
                    data = CalendarContract.CONTENT_URI.buildUpon()
                        .appendPath("time")
                        .appendEncodedPath(System.currentTimeMillis().toString())
                        .build()
                }
                context.startActivity(intent)
            },
        )
    }
    is ProactiveEventBus.Event.LocationArrived -> HistoryCardModel(
        icon = "📍",
        title = "Arrived at $placeName",
        body = "Recalled ${recalledMemories.size} memories",
        tapHint = null,
        timestamp = timestamp,
        onClick = null,
    )
    is ProactiveEventBus.Event.MemoryDecayWarning -> HistoryCardModel(
        icon = "💭",
        title = "Memory fading",
        body = preview,
        tapHint = "Open memory",
        timestamp = timestamp,
        onClick = {
            context.startActivity(
                Intent(context, com.aura.MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_REORDER_TO_FRONT
                    putExtra("openMemory", true)
                }
            )
        },
    )
}

@Composable
private fun HistoryCard(
    model: HistoryCardModel,
    modifier: Modifier = Modifier,
) {
    val fmt = SimpleDateFormat("MMM d, HH:mm", Locale.US)
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(12.dp),
        modifier = modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Text(
                text = model.icon,
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.size(28.dp),
            )
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = model.title,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = model.body,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.85f),
                )
                Spacer(modifier = Modifier.height(2.dp))
                model.tapHint?.let { hint ->
                    Text(
                        text = hint,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = fmt.format(Date(model.timestamp)),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                )
            }
        }
    }
}
