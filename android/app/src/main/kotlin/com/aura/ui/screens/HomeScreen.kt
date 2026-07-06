package com.aura.ui.screens

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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.proactive.ProactiveEventBus
import com.aura.ui.util.toSummary
import com.aura.ui.viewmodel.HomeViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HomeScreen(
    viewModel: HomeViewModel = hiltViewModel(),
    onOpenChat: () -> Unit = {},
    onOpenProactive: () -> Unit = {},
    onOpenMemory: () -> Unit = {},
    onOpenTasks: () -> Unit = {},
    onOpenCalendar: () -> Unit = {},
) {
    val state by viewModel.state.collectAsState()
    val greeting = when (state.hour) {
        in 5..11 -> "Good morning"
        in 12..17 -> "Good afternoon"
        in 18..21 -> "Good evening"
        else -> "Working late"
    }
    val nameSuffix = state.userName?.let { ", $it" } ?: ""
    val dateStr = SimpleDateFormat("EEE, MMM d", Locale.US).format(Date())

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        // Proactive event card (if present) + "see all" link with
        // unread count. The card renders the latest event; the link
        // surfaces the total count and navigates to the history
        // screen.
        if (state.proactiveEvent != null || state.proactiveUnreadCount > 0) {
            state.proactiveEvent?.let { event ->
                val onTap = when (event) {
                    is ProactiveEventBus.Event.MorningBriefReady,
                    is ProactiveEventBus.Event.MorningBriefStructured -> onOpenChat
                    is ProactiveEventBus.Event.CalendarEventSoon -> onOpenCalendar
                    is ProactiveEventBus.Event.MemoryDecayWarning -> onOpenMemory
                    is ProactiveEventBus.Event.LocationArrived -> { {} }
                }
                ProactiveEventCard(
                    event = event,
                    onDismiss = { viewModel.dismissProactiveEvent() },
                    onTap = onTap,
                )
            }
            // "📬 N today · see all →" affordance — visible whenever
            // there is at least 1 unread event, even if the latest
            // event card has auto-dismissed.
            if (state.proactiveUnreadCount > 0) {
                ProactiveUnreadLink(
                    count = state.proactiveUnreadCount,
                    onClick = {
                        viewModel.onProactiveHistoryOpened()
                        onOpenProactive()
                    },
                )
            }
        }

        // Greeting
        Text(
            text = "$greeting$nameSuffix",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Text(
            text = dateStr,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )

        // Pending tasks
        if (state.pendingTasks.isNotEmpty()) {
            BriefCard(title = "📋 Open tasks", lines = state.pendingTasks)
        }

        // Recent memories
        if (state.recentMemories.isNotEmpty()) {
            BriefCard(
                title = "🧠 What I remember",
                lines = state.recentMemories.map { "· ${it.content}" },
            )
        }

        // Today's calendar
        if (state.today.isNotEmpty()) {
            BriefCard(title = "📅 Today", lines = state.today)
        }

        // Empty state
        if (state.pendingTasks.isEmpty() && state.recentMemories.isEmpty() && state.today.isEmpty() && !state.loading) {
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Text(
                        text = "Nothing scheduled, nothing remembered yet.",
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Chat with Aura to start building memory. Tell it things like \"my name is Elnur\" or \"I prefer dark mode\".",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(8.dp))

        // CTA
        Button(
            onClick = onOpenChat,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Open chat")
        }
        if (state.pendingTasks.isNotEmpty()) {
            OutlinedButton(
                onClick = onOpenTasks,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Open ${state.pendingTasks.size} task${if (state.pendingTasks.size == 1) "" else "s"}")
            }
        }
        if (state.recentMemories.isNotEmpty()) {
            OutlinedButton(
                onClick = onOpenMemory,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Go to memories")
            }
        }
    }
}

@Composable
private fun ProactiveEventCard(
    event: ProactiveEventBus.Event,
    onDismiss: () -> Unit,
    onTap: () -> Unit = {},
) {
    // Auto-dismiss after 30 seconds
    LaunchedEffect(event) {
        kotlinx.coroutines.delay(30_000L)
        onDismiss()
    }

    val (icon, title, body) = when (event) {
        is ProactiveEventBus.Event.MorningBriefReady -> {
            Triple("☀️", "Morning brief", event.body)
        }
        // The structured event fires alongside MorningBriefReady.
        // Render the deterministic summary from BriefContext so the
        // Home card shows what actually changed today instead of a
        // generic placeholder.
        is ProactiveEventBus.Event.MorningBriefStructured -> {
            Triple("\u2600\uFE0F", "Morning brief", event.context.toSummary())
        }
        is ProactiveEventBus.Event.CalendarEventSoon -> {
            val minutes = event.minutesUntil
            val label = if (minutes < 60) "in $minutes min" else "in ${minutes / 60}h ${minutes % 60}m"
            Triple("📅", "Upcoming event: ${event.title}", label)
        }
        is ProactiveEventBus.Event.LocationArrived -> {
            Triple("📍", "Arrived at ${event.placeName}", "I'm at ${event.placeName}")
        }
        is ProactiveEventBus.Event.MemoryDecayWarning -> {
            Triple("💭", "Memory fading", event.preview)
        }
    }

    Surface(
        color = MaterialTheme.colorScheme.tertiaryContainer,
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onTap() },
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Text(
                text = icon,
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.size(28.dp),
            )
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = body,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer.copy(alpha = 0.85f),
                )
            }
            IconButton(
                onClick = onDismiss,
                modifier = Modifier.size(24.dp),
            ) {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "Dismiss",
                    tint = MaterialTheme.colorScheme.onTertiaryContainer,
                    modifier = Modifier.size(18.dp),
                )
            }
        }
    }
}

@Composable
private fun BriefCard(title: String, lines: List<String>) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(8.dp))
            for (line in lines.take(5)) {
                Text(
                    text = line,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.85f),
                )
                Spacer(modifier = Modifier.height(4.dp))
            }
        }
    }
}

/**
 * "📬 N today · see all →" affordance for the proactive history
 * screen. Rendered on Home whenever the user has at least one
 * unread proactive event — independent of whether the latest-event
 * card is currently displayed (it auto-dismisses after 30s).
 *
 * Tapping the row marks the events as seen, clears the badge, and
 * navigates to the history screen via the provided callback.
 */
@Composable
private fun ProactiveUnreadLink(count: Int, onClick: () -> Unit) {
    val label = if (count == 1) "📬 1 today" else "📬 $count today"
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
        )
        Spacer(modifier = Modifier.weight(1f))
        TextButton(onClick = onClick) {
            Text("see all →")
        }
    }
}
