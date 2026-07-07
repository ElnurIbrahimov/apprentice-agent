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
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material3.Button
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
    val dateStr = SimpleDateFormat("EEEE, MMMM d", Locale.US).format(Date())

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        // ── Hero header: greeting + date ──────────────────────────────────
        Column(modifier = Modifier.padding(top = 8.dp)) {
            Text(
                text = "$greeting$nameSuffix",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = dateStr,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
            )
        }

        // ── Proactive event card ──────────────────────────────────────────
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

        // ── Quick action: jump into chat ──────────────────────────────────
        Surface(
            color = MaterialTheme.colorScheme.primary,
            shape = RoundedCornerShape(20.dp),
            modifier = Modifier
                .fillMaxWidth()
                .clickable { onOpenChat() },
        ) {
            Row(
                modifier = Modifier.padding(20.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Surface(
                    color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(14.dp),
                ) {
                    Icon(
                        imageVector = Icons.Filled.Send,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onPrimary,
                        modifier = Modifier
                            .size(40.dp)
                            .padding(8.dp),
                    )
                }
                Spacer(modifier = Modifier.width(14.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Talk to Aura",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Text(
                        text = "Ask anything, set a reminder, or check in",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.75f),
                    )
                }
            }
        }

        // ── Quick action grid: memory / tasks / calendar ──────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            QuickActionCard(
                icon = Icons.Filled.Psychology,
                label = "Memory",
                count = state.recentMemories.size,
                onClick = onOpenMemory,
                modifier = Modifier.weight(1f),
            )
            QuickActionCard(
                icon = Icons.Filled.TaskAlt,
                label = "Tasks",
                count = state.pendingTasks.size,
                onClick = onOpenTasks,
                modifier = Modifier.weight(1f),
            )
            QuickActionCard(
                icon = Icons.Filled.CalendarMonth,
                label = "Calendar",
                count = state.today.size,
                onClick = onOpenCalendar,
                modifier = Modifier.weight(1f),
            )
        }

        // ── Detail cards ──────────────────────────────────────────────────
        if (state.pendingTasks.isNotEmpty()) {
            BriefCard(
                title = "Open tasks",
                lines = state.pendingTasks,
                onClick = onOpenTasks,
            )
        }

        if (state.recentMemories.isNotEmpty()) {
            BriefCard(
                title = "What I remember",
                lines = state.recentMemories.map { "· ${it.content}" },
                onClick = onOpenMemory,
            )
        }

        if (state.today.isNotEmpty()) {
            BriefCard(
                title = "Today",
                lines = state.today,
                onClick = onOpenCalendar,
            )
        }

        // ── Empty state ───────────────────────────────────────────────────
        if (state.pendingTasks.isEmpty() && state.recentMemories.isEmpty() && state.today.isEmpty() && !state.loading) {
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                shape = RoundedCornerShape(16.dp),
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

        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable
private fun QuickActionCard(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    count: Int,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(16.dp),
        modifier = modifier.clickable { onClick() },
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp, horizontal = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Surface(
                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                shape = RoundedCornerShape(12.dp),
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier
                        .size(36.dp)
                        .padding(6.dp),
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (count > 0) {
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "$count",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                )
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
    LaunchedEffect(event) {
        kotlinx.coroutines.delay(30_000L)
        onDismiss()
    }

    val (icon, title, body) = when (event) {
        is ProactiveEventBus.Event.MorningBriefReady -> {
            Triple("☀️", "Morning brief", event.body)
        }
        is ProactiveEventBus.Event.MorningBriefStructured -> {
            Triple("☀️", "Morning brief", event.context.toSummary())
        }
        is ProactiveEventBus.Event.CalendarEventSoon -> {
            val minutes = event.minutesUntil
            val label = if (minutes < 60) "in $minutes min" else "in ${minutes / 60}h ${minutes % 60}m"
            Triple("📅", "Upcoming: ${event.title}", label)
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
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onTap() },
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Text(
                text = icon,
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.size(32.dp),
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
                modifier = Modifier.size(28.dp),
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
private fun BriefCard(title: String, lines: List<String>, onClick: () -> Unit = {}) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
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

@Composable
private fun ProactiveUnreadLink(count: Int, onClick: () -> Unit) {
    val label = if (count == 1) "📬 1 today" else "📬 $count today"
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
            .padding(horizontal = 4.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            imageVector = Icons.Filled.NotificationsActive,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(18.dp),
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
        )
        Spacer(modifier = Modifier.weight(1f))
        Text(
            text = "see all →",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.primary,
        )
    }
}