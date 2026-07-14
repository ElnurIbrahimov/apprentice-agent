package com.aura.ui.screens.home

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aura.proactive.ProactiveEventBus
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens
import com.aura.ui.util.toSummary
import com.aura.ui.viewmodel.HomeUiState

sealed interface HomePriority {
    data class Proactive(val event: ProactiveEventBus.Event) : HomePriority
    data class Calendar(val label: String) : HomePriority
    data class Task(val title: String) : HomePriority
    data class Reminder(val label: String) : HomePriority
    data class Memory(val summary: String) : HomePriority
    data object None : HomePriority
}

fun selectHomePriority(state: HomeUiState): HomePriority = when {
    state.proactiveEvent != null -> HomePriority.Proactive(state.proactiveEvent)
    state.today.isNotEmpty() -> HomePriority.Calendar(state.today.first())
    state.pendingTasks.isNotEmpty() -> HomePriority.Task(state.pendingTasks.first())
    state.upcomingReminders.isNotEmpty() -> HomePriority.Reminder(state.upcomingReminders.first())
    state.recentMemories.isNotEmpty() -> HomePriority.Memory(summarizeMemory(state.recentMemories.first().content))
    else -> HomePriority.None
}

internal fun summarizeMemory(content: String): String {
    val sentence = content.substringBefore('.').ifBlank { content }.trim()
    return if (sentence.length > 72) sentence.take(72).trimEnd() + "…" else sentence
}

private data class PriorityPresentation(
    val icon: ImageVector,
    val title: String,
    val message: String,
    val action: String,
)

@Composable
fun HomeBriefCard(
    priority: HomePriority,
    onAction: () -> Unit,
    onDismiss: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    val colors = AuraThemeTokens.colors
    val presentation = priority.presentation()
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onAction),
        color = colors.surface1,
        shape = MaterialTheme.shapes.large,
        tonalElevation = 0.dp,
    ) {
        Column(
            modifier = Modifier.padding(AuraSpacing.md),
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "Current priority",
                    style = MaterialTheme.typography.labelLarge,
                    color = colors.textSecondary,
                )
                Spacer(Modifier.weight(1f))
                if (onDismiss != null && priority is HomePriority.Proactive) {
                    IconButton(onClick = onDismiss, modifier = Modifier.size(AuraDimensions.minimumTouchTarget)) {
                        Icon(Icons.Default.Close, contentDescription = "Dismiss", tint = colors.textSecondary)
                    }
                }
            }
            Row(
                verticalAlignment = Alignment.Top,
                horizontalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
            ) {
                Surface(
                    color = colors.actionPrimary.copy(alpha = 0.14f),
                    shape = MaterialTheme.shapes.medium,
                ) {
                    Icon(
                        imageVector = presentation.icon,
                        contentDescription = null,
                        tint = colors.actionPrimary,
                        modifier = Modifier.padding(AuraSpacing.sm).size(24.dp),
                    )
                }
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(AuraSpacing.xs),
                ) {
                    Text(
                        text = presentation.title,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = colors.textPrimary,
                    )
                    Text(
                        text = presentation.message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = colors.textSecondary,
                        maxLines = 3,
                    )
                    Text(
                        text = presentation.action,
                        style = MaterialTheme.typography.labelLarge,
                        color = colors.actionPrimary,
                    )
                }
            }
        }
    }
}

private fun HomePriority.presentation(): PriorityPresentation = when (this) {
    is HomePriority.Proactive -> when (val item = event) {
        is ProactiveEventBus.Event.MorningBriefReady -> PriorityPresentation(
            Icons.Filled.WbSunny, item.title, item.body, "Discuss brief",
        )
        is ProactiveEventBus.Event.MorningBriefStructured -> PriorityPresentation(
            Icons.Filled.WbSunny, "Morning brief", item.context.toSummary(), "Discuss brief",
        )
        is ProactiveEventBus.Event.CalendarEventSoon -> PriorityPresentation(
            Icons.Filled.CalendarMonth,
            item.title,
            if (item.minutesUntil < 60) "Starts in ${item.minutesUntil} minutes" else "Starts in ${item.minutesUntil / 60} hours",
            "Open calendar",
        )
        is ProactiveEventBus.Event.LocationArrived -> PriorityPresentation(
            Icons.Filled.LocationOn, item.placeName, "Ask Aura what it remembers about this place", "Ask Aura",
        )
        is ProactiveEventBus.Event.MemoryDecayWarning -> PriorityPresentation(
            Icons.Filled.Psychology, "Memory needs attention", item.preview, "Review memory",
        )
    }
    is HomePriority.Calendar -> PriorityPresentation(
        Icons.Filled.CalendarMonth, label, "Your next calendar item", "Open calendar",
    )
    is HomePriority.Task -> PriorityPresentation(
        Icons.Filled.TaskAlt, title, "Your next open task", "Open tasks",
    )
    is HomePriority.Reminder -> PriorityPresentation(
        Icons.Filled.NotificationsActive, label, "Your next scheduled reminder", "Open reminders",
    )
    is HomePriority.Memory -> PriorityPresentation(
        Icons.Filled.Psychology, summary, "A recent detail Aura remembers", "Open memory",
    )
    HomePriority.None -> PriorityPresentation(
        Icons.Filled.TaskAlt,
        "Nothing needs attention yet",
        "Create a task or tell Aura what matters today.",
        "Create a task",
    )
}
