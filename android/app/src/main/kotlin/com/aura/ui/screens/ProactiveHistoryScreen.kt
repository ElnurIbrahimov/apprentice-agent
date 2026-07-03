package com.aura.ui.screens

import androidx.compose.foundation.layout.*; import androidx.compose.foundation.lazy.LazyColumn; import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape; import androidx.compose.material3.*; import androidx.compose.runtime.*; import androidx.compose.ui.Alignment; import androidx.compose.ui.Modifier; import androidx.compose.ui.text.font.FontWeight; import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel; import com.aura.proactive.ProactiveEventBus; import com.aura.ui.viewmodel.ProactiveHistoryViewModel
import java.text.SimpleDateFormat; import java.util.*

@Composable
fun ProactiveHistoryScreen(viewModel: ProactiveHistoryViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    Column(Modifier.fillMaxSize().padding(16.dp)) {
        Text("Proactive Events", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
        Text("${state.events.size} events (last 100)", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f))
        Spacer(Modifier.height(12.dp)); HorizontalDivider(); Spacer(Modifier.height(8.dp))
        if (state.events.isEmpty()) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Column(horizontalAlignment = Alignment.CenterHorizontally) { Text("🔔", style = MaterialTheme.typography.displayLarge); Spacer(Modifier.height(8.dp)); Text("No events yet", style = MaterialTheme.typography.titleMedium); Text("Proactive events fire automatically — calendar alerts, morning briefs, location arrivals.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)) } }
        else LazyColumn(contentPadding = PaddingValues(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) { items(state.events) { event -> EventRow(event) } }
    }
}

@Composable
private fun EventRow(event: ProactiveEventBus.Event) {
    val fmt = SimpleDateFormat("MMM d, HH:mm", Locale.US)
    val (icon, title, body) = when (event) {
        is ProactiveEventBus.Event.MorningBriefReady -> Triple("☀️", "Morning brief", event.body)
        is ProactiveEventBus.Event.MorningBriefStructured -> {
            val ctx = event.context
            val lines = mutableListOf<String>()
            if (ctx.decayedMemories.isNotEmpty()) lines += "💭 ${ctx.decayedMemories.size} fading"
            if (ctx.newMemories.isNotEmpty()) lines += "🧠 ${ctx.newMemories.size} new"
            if (ctx.newKgNodes.isNotEmpty()) lines += "🕸️ ${ctx.newKgNodes.size} facts"
            if (ctx.tasksDueToday.isNotEmpty()) lines += "📋 ${ctx.tasksDueToday.size} tasks"
            if (ctx.calendarToday.isNotEmpty()) lines += "📅 ${ctx.calendarToday.size} events"
            Triple("☀️", "Morning brief", lines.joinToString(" · "))
        }
        is ProactiveEventBus.Event.CalendarEventSoon -> Triple("📅", "Upcoming: ${event.title}", "in ${event.minutesUntil}m")
        is ProactiveEventBus.Event.LocationArrived -> Triple("📍", "Arrived: ${event.placeName}", "")
        is ProactiveEventBus.Event.MemoryDecayWarning -> Triple("💭", "Memory fading", event.preview)
    }
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.Top) {
            Text(icon, style = MaterialTheme.typography.titleLarge, modifier = Modifier.size(28.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) { Text(title, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold); if (body.isNotBlank()) { Spacer(Modifier.height(2.dp)); Text(body, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)) } }
            Text(fmt.format(Date(event.timestamp)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f))
        }
    }
}
