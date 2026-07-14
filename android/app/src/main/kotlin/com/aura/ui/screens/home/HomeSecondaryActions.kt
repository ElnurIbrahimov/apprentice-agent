package com.aura.ui.screens.home

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens

private data class HomeDestination(
    val label: String,
    val metadata: String,
    val icon: ImageVector,
    val onClick: () -> Unit,
)

@Composable
fun HomeSecondaryActions(
    memoryCount: Int,
    tasksCount: Int,
    calendarCount: Int,
    handsCount: Int,
    toolsCount: Int,
    proactiveCount: Int,
    onOpenMemory: () -> Unit,
    onOpenTasks: () -> Unit,
    onOpenCalendar: () -> Unit,
    onOpenHands: () -> Unit,
    onOpenTools: () -> Unit,
    onOpenProactive: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val destinations = listOf(
        HomeDestination(
            "Memory",
            if (memoryCount == 0) "Add memory" else "$memoryCount saved",
            Icons.Filled.Psychology,
            onOpenMemory,
        ),
        HomeDestination(
            "Tasks",
            if (tasksCount == 0) "Create task" else "$tasksCount open",
            Icons.Filled.TaskAlt,
            onOpenTasks,
        ),
        HomeDestination(
            "Calendar",
            if (calendarCount == 0) "Check calendar" else "$calendarCount today",
            Icons.Filled.CalendarMonth,
            onOpenCalendar,
        ),
        HomeDestination(
            "Hands",
            if (handsCount == 0) "Create hand" else "$handsCount active",
            Icons.Filled.Build,
            onOpenHands,
        ),
        HomeDestination(
            "Tools",
            if (toolsCount == 0) "Browse tools" else "$toolsCount available",
            Icons.Filled.Lightbulb,
            onOpenTools,
        ),
        HomeDestination(
            "Proactive",
            if (proactiveCount == 0) "View activity" else "$proactiveCount updates",
            Icons.Filled.NotificationsActive,
            onOpenProactive,
        ),
    )

    Column(
        modifier = modifier.testTag("home-secondary"),
        verticalArrangement = Arrangement.spacedBy(AuraSpacing.xs),
    ) {
        Text(
            text = "Open",
            style = MaterialTheme.typography.labelLarge,
            color = AuraThemeTokens.colors.textSecondary,
        )
        LazyRow(
            modifier = Modifier.testTag("home-destinations"),
            horizontalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
        ) {
            items(destinations, key = { it.label }) { destination ->
                HomeDestinationCard(destination)
            }
        }
    }
}

@Composable
private fun HomeDestinationCard(destination: HomeDestination) {
    val colors = AuraThemeTokens.colors
    Surface(
        modifier = Modifier
            .width(148.dp)
            .defaultMinSize(minHeight = 82.dp)
            .clickable(onClick = destination.onClick),
        color = colors.surface1,
        shape = MaterialTheme.shapes.medium,
        border = androidx.compose.foundation.BorderStroke(1.dp, colors.borderSubtle),
    ) {
        Column(
            modifier = Modifier.padding(AuraSpacing.sm),
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.xs),
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(AuraSpacing.xs),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    imageVector = destination.icon,
                    contentDescription = null,
                    tint = colors.actionPrimary,
                    modifier = Modifier.size(20.dp),
                )
                Text(
                    text = destination.label,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = colors.textPrimary,
                )
            }
            Text(
                text = destination.metadata,
                style = MaterialTheme.typography.bodySmall,
                color = colors.textSecondary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}
