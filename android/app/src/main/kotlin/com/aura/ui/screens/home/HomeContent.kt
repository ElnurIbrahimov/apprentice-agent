package com.aura.ui.screens.home

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aura.proactive.ProactiveEventBus
import com.aura.ui.components.AuraErrorState
import com.aura.ui.components.AuraInlineStatus
import com.aura.ui.components.AuraSkeleton
import com.aura.ui.components.InlineStatusTone
import com.aura.ui.components.ResponsiveContainer
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens
import com.aura.ui.util.toSummary
import com.aura.ui.viewmodel.HomeLoadState
import com.aura.ui.viewmodel.HomeUiState

@Composable
fun HomeContent(
    state: HomeUiState,
    greeting: String,
    dateLabel: String,
    onAskAura: (String) -> Unit = {},
    onRetry: () -> Unit = {},
    onDismissProactive: () -> Unit = {},
    onOpenChatWithBrief: (String) -> Unit = {},
    onOpenMemory: () -> Unit = {},
    onOpenTasks: () -> Unit = {},
    onOpenCalendar: () -> Unit = {},
    onOpenReminders: () -> Unit = {},
    onOpenHands: () -> Unit = {},
    onOpenTools: () -> Unit = {},
    onOpenCreative: () -> Unit = {},
    onOpenProactive: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    val colors = AuraThemeTokens.colors
    ResponsiveContainer(modifier = modifier.fillMaxSize()) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(vertical = AuraSpacing.md),
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.md),
        ) {
            item(key = "header") {
                Column(verticalArrangement = Arrangement.spacedBy(AuraSpacing.xxs)) {
                    Text(
                        text = greeting,
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = colors.textPrimary,
                    )
                    Text(
                        text = dateLabel,
                        style = MaterialTheme.typography.labelLarge,
                        color = colors.textSecondary,
                    )
                }
            }

            when (val loadState = state.loadState) {
                HomeLoadState.Loading -> item(key = "loading") {
                    Column(
                        modifier = Modifier.testTag("home-loading"),
                        verticalArrangement = Arrangement.spacedBy(AuraSpacing.md),
                    ) {
                        AuraSkeleton(height = 176.dp)
                        AuraSkeleton(height = 142.dp)
                        AuraSkeleton(height = 88.dp)
                    }
                }

                is HomeLoadState.Error -> {
                    if (!loadState.hasPartialContent) {
                        item(key = "full-error") {
                            AuraErrorState(
                                title = "Home unavailable",
                                message = loadState.message,
                                onRetry = onRetry,
                                retryLabel = "Retry",
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    } else {
                        item(key = "partial-error") {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(AuraSpacing.xs),
                            ) {
                                AuraInlineStatus(
                                    text = loadState.message,
                                    tone = InlineStatusTone.Warning,
                                    modifier = Modifier.weight(1f),
                                )
                                TextButton(onClick = onRetry) { Text("Retry") }
                            }
                        }
                        homeResolvedItems(
                            state = state,
                            isEmpty = false,
                            onAskAura = onAskAura,
                            onDismissProactive = onDismissProactive,
                            onOpenChatWithBrief = onOpenChatWithBrief,
                            onOpenMemory = onOpenMemory,
                            onOpenTasks = onOpenTasks,
                            onOpenCalendar = onOpenCalendar,
                            onOpenReminders = onOpenReminders,
                            onOpenHands = onOpenHands,
                            onOpenTools = onOpenTools,
                            onOpenCreative = onOpenCreative,
                            onOpenProactive = onOpenProactive,
                        )
                    }
                }

                HomeLoadState.Empty -> homeResolvedItems(
                    state = state,
                    isEmpty = true,
                    onAskAura = onAskAura,
                    onDismissProactive = onDismissProactive,
                    onOpenChatWithBrief = onOpenChatWithBrief,
                    onOpenMemory = onOpenMemory,
                    onOpenTasks = onOpenTasks,
                    onOpenCalendar = onOpenCalendar,
                    onOpenReminders = onOpenReminders,
                    onOpenHands = onOpenHands,
                    onOpenTools = onOpenTools,
                    onOpenCreative = onOpenCreative,
                    onOpenProactive = onOpenProactive,
                )

                HomeLoadState.Content -> homeResolvedItems(
                    state = state,
                    isEmpty = false,
                    onAskAura = onAskAura,
                    onDismissProactive = onDismissProactive,
                    onOpenChatWithBrief = onOpenChatWithBrief,
                    onOpenMemory = onOpenMemory,
                    onOpenTasks = onOpenTasks,
                    onOpenCalendar = onOpenCalendar,
                    onOpenReminders = onOpenReminders,
                    onOpenHands = onOpenHands,
                    onOpenTools = onOpenTools,
                    onOpenCreative = onOpenCreative,
                    onOpenProactive = onOpenProactive,
                )
            }
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.homeResolvedItems(
    state: HomeUiState,
    isEmpty: Boolean,
    onAskAura: (String) -> Unit,
    onDismissProactive: () -> Unit,
    onOpenChatWithBrief: (String) -> Unit,
    onOpenMemory: () -> Unit,
    onOpenTasks: () -> Unit,
    onOpenCalendar: () -> Unit,
    onOpenReminders: () -> Unit,
    onOpenHands: () -> Unit = {},
    onOpenTools: () -> Unit = {},
    onOpenCreative: () -> Unit = {},
    onOpenProactive: () -> Unit = {},
) {
    val priority = selectHomePriority(state)
    item(key = "primary") {
        Column(modifier = Modifier.testTag(if (isEmpty) "home-empty" else "home-content")) {
            HomePrimaryAction(onAskAura = onAskAura)
        }
    }
    item(key = "priority") {
        HomeBriefCard(
            priority = priority,
            onAction = {
                when (priority) {
                    is HomePriority.Proactive -> when (val event = priority.event) {
                        is ProactiveEventBus.Event.MorningBriefReady -> onOpenChatWithBrief(event.body)
                        is ProactiveEventBus.Event.MorningBriefStructured ->
                            onOpenChatWithBrief(event.context.toSummary())
                        is ProactiveEventBus.Event.CalendarEventSoon -> onOpenCalendar()
                        is ProactiveEventBus.Event.MemoryDecayWarning -> onOpenMemory()
                    }
                    is HomePriority.Calendar -> onOpenCalendar()
                    is HomePriority.Task -> onOpenTasks()
                    is HomePriority.Reminder -> onOpenReminders()
                    is HomePriority.Memory -> onOpenMemory()
                    HomePriority.None -> onOpenTasks()
                }
            },
            onDismiss = onDismissProactive.takeIf { priority is HomePriority.Proactive },
        )
    }
    item(key = "secondary") {
        HomeSecondaryActions(
            memoryCount = state.recentMemories.size,
            tasksCount = state.pendingTasks.size,
            calendarCount = state.today.size,
            handsCount = state.handsCount,
            toolsCount = state.toolsCount,
            proactiveCount = maxOf(state.proactiveCount, state.proactiveUnreadCount),
            onOpenMemory = onOpenMemory,
            onOpenTasks = onOpenTasks,
            onOpenCalendar = onOpenCalendar,
            onOpenHands = onOpenHands,
            onOpenTools = onOpenTools,
            onOpenCreative = onOpenCreative,
            onOpenProactive = onOpenProactive,
        )
    }
    if (isEmpty) {
        item(key = "fresh-start") {
            HomeFreshStart(
                onAskAura = onAskAura,
                onOpenMemory = onOpenMemory,
                onOpenTasks = onOpenTasks,
            )
        }
    }
    if (!isEmpty) {
        item(key = "at-a-glance") {
            HomeAtAGlance(state)
        }
    }
}

@Composable
private fun HomeFreshStart(
    onAskAura: (String) -> Unit,
    onOpenMemory: () -> Unit,
    onOpenTasks: () -> Unit,
) {
    val colors = AuraThemeTokens.colors
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = colors.surface0,
        shape = MaterialTheme.shapes.large,
        border = androidx.compose.foundation.BorderStroke(1.dp, colors.borderSubtle),
    ) {
        Column(
            modifier = Modifier.padding(AuraSpacing.md),
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.xs),
        ) {
            Text(
                text = "Build useful context",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = colors.textPrimary,
            )
            FreshStartAction(
                icon = Icons.Filled.AutoAwesome,
                title = "Decide what matters today",
                message = "Turn an uncertain day into a concrete plan.",
                onClick = { onAskAura("Help me decide what matters most today") },
            )
            FreshStartAction(
                icon = Icons.Filled.TaskAlt,
                title = "Turn a goal into a task",
                message = "Give the day one concrete next step.",
                onClick = onOpenTasks,
            )
            FreshStartAction(
                icon = Icons.Filled.Psychology,
                title = "Capture something to remember",
                message = "Add context Aura can use in future conversations.",
                onClick = onOpenMemory,
            )
        }
    }
}

@Composable
private fun FreshStartAction(
    icon: ImageVector,
    title: String,
    message: String,
    onClick: () -> Unit,
) {
    val colors = AuraThemeTokens.colors
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .defaultMinSize(minHeight = 64.dp)
            .clickable(onClick = onClick)
            .padding(vertical = AuraSpacing.xs),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = colors.actionPrimary,
            modifier = Modifier.size(22.dp),
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = colors.textPrimary,
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = colors.textSecondary,
                maxLines = 2,
            )
        }
    }
}

@Composable
private fun HomeAtAGlance(state: HomeUiState) {
    val colors = AuraThemeTokens.colors
    val lines = buildList {
        state.pendingTasks.drop(1).take(2).forEach { add("Task · $it") }
        state.upcomingReminders.drop(if (state.pendingTasks.isEmpty()) 1 else 0).take(1)
            .forEach { add("Reminder · $it") }
        state.today.drop(if (state.pendingTasks.isEmpty() && state.upcomingReminders.isEmpty()) 1 else 0).take(1)
            .forEach { add("Calendar · $it") }
    }.take(3)
    if (lines.isEmpty()) return

    Column(
        modifier = Modifier.fillMaxWidth().padding(bottom = AuraSpacing.sm),
        verticalArrangement = Arrangement.spacedBy(AuraSpacing.xs),
    ) {
        Text(
            text = "At a glance",
            style = MaterialTheme.typography.labelLarge,
            color = colors.textSecondary,
        )
        lines.forEach { line ->
            Text(
                text = line,
                style = MaterialTheme.typography.bodyMedium,
                color = colors.textPrimary,
            )
        }
    }
}
