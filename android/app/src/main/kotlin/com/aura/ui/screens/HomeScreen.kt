package com.aura.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.proactive.ProactiveEventBus
import com.aura.ui.util.toSummary
import com.aura.ui.viewmodel.HomeViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Example prompts shown on Home when the user has no context yet.
 * These are the first impressions of Aura — they should feel
 * inviting, capable, and personal. Tapping a chip starts a chat
 * with that prompt pre-filled.
 */
private val examplePrompts = listOf(
    ExamplePrompt("Plan my day", "Help me plan my day based on what I have going on", Icons.Filled.WbSunny),
    ExamplePrompt("Write a message", "Draft a message to a friend I haven't talked to in a while", Icons.Filled.AutoAwesome),
    ExamplePrompt("Set a reminder", "Remind me to take a break in 30 minutes", Icons.Filled.TaskAlt),
    ExamplePrompt("Brainstorm", "Help me brainstorm names for my new project", Icons.Filled.Lightbulb),
    ExamplePrompt("Summarize", "Read my last conversation and give me the key points", Icons.Filled.Psychology),
    ExamplePrompt("Today's plan", "What's on my calendar today?", Icons.Filled.CalendarMonth),
)

private data class ExamplePrompt(
    val label: String,
    val prompt: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector,
)

@Composable
fun HomeScreen(
    viewModel: HomeViewModel = hiltViewModel(),
    onOpenChat: (prefillDraft: String) -> Unit = {},
    /**
     * Like [onOpenChat] but the chat screen receives the brief as
     * `morningBriefSummary` — the chat auto-sends the brief text
     * as a user message so the user lands mid-conversation, not on
     * a blank draft. Used by the morning-brief proactive event card.
     */
    onOpenChatWithBrief: (briefText: String) -> Unit = {},
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

    // Sub-status line. This is where Aura gets its voice — instead of
    // a generic "what's up", it tells the user what it knows about right
    // now. Empty state is the only time it has to introduce itself.
    //
    // If the BriefContext has anything to say (decayed memories, new
    // facts, tasks due today, calendar events), use the structured
    // summary. Otherwise fall back to the count-based greeting.
    val briefSummary = state.briefContext.toSummary()
    val subStatus = when {
        briefSummary.isNotBlank() -> briefSummary
        state.pendingTasks.isNotEmpty() && state.today.isNotEmpty() ->
            "You have ${state.pendingTasks.size} open task${if (state.pendingTasks.size == 1) "" else "s"} and ${state.today.size} on the calendar today."
        state.pendingTasks.isNotEmpty() ->
            "You have ${state.pendingTasks.size} open task${if (state.pendingTasks.size == 1) "" else "s"} to clear."
        state.today.isNotEmpty() ->
            "Your day is starting to take shape — ${state.today.size} on the calendar."
        state.recentMemories.isNotEmpty() ->
            "I've been paying attention. I remember ${state.recentMemories.size} thing${if (state.recentMemories.size == 1) "" else "s"} about you."
        else ->
            "I'm fresh. What do you want to do first?"
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        // ── Hero: greeting + persona line + date ─────────────────────────
        Column(
            modifier = Modifier.padding(top = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(
                text = "$greeting$nameSuffix",
                style = MaterialTheme.typography.displayMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Text(
                text = subStatus,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = dateStr,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.4f),
            )
        }

        // ── Proactive event card (if any) ────────────────────────────────
        AnimatedVisibility(
            visible = state.proactiveEvent != null,
            enter = slideInVertically { -it } + fadeIn(),
            exit = fadeOut(),
        ) {
            state.proactiveEvent?.let { event ->
                val onTap = when (event) {
                    is ProactiveEventBus.Event.MorningBriefReady -> {
                        // Send the brief body as a user message so the user
                        // lands on a chat that's already mid-discussion of
                        // the brief, not on a blank draft.
                        { onOpenChatWithBrief(event.body) }
                    }
                    is ProactiveEventBus.Event.MorningBriefStructured -> {
                        // The structured event doesn't carry a freeform
                        // body — render the summary lines as the user
                        // message so the user can discuss the brief.
                        { onOpenChatWithBrief(event.context.toSummary()) }
                    }
                    is ProactiveEventBus.Event.CalendarEventSoon -> {
                        { onOpenCalendar() }
                    }
                    is ProactiveEventBus.Event.MemoryDecayWarning -> {
                        { onOpenMemory() }
                    }
                    is ProactiveEventBus.Event.LocationArrived -> {
                        // Open a chat asking "Where am I?" so the agent
                        // can describe the place + recalled memories.
                        { onOpenChat("Where am I? What do you remember about this place?") }
                    }
                }
                ProactiveEventCard(
                    event = event,
                    onDismiss = { viewModel.dismissProactiveEvent() },
                    onTap = onTap,
                )
            }
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

        // ── Quick action: jump into chat (or show examples) ─────────────
        if (state.recentMemories.isEmpty() && state.today.isEmpty() && state.pendingTasks.isEmpty()) {
            // Empty state — show example prompts instead of just data
            ExamplePromptsGrid(onPromptClick = { onOpenChat(it) })
        } else {
            Surface(
                color = MaterialTheme.colorScheme.primary,
                shape = RoundedCornerShape(20.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onOpenChat("") },
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
                            text = "Ask anything, set a reminder, or just check in",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.75f),
                        )
                    }
                }
            }
        }

        // ── Quick action grid: memory / tasks / calendar ─────────────────
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

        // ── Detail cards ─────────────────────────────────────────────────
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
                // First sentence per memory, truncated to 60 chars + ellipsis.
                // Raw content can be hundreds of chars; the brief card is a
                // 2-line peek, not the full text.
                lines = state.recentMemories.map { "· ${summarizeMemory(it.content)}" },
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

        Spacer(modifier = Modifier.height(8.dp))
    }
}

@Composable
private fun ExamplePromptsGrid(onPromptClick: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = "Try one of these",
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
            modifier = Modifier.padding(start = 4.dp, bottom = 4.dp),
        )
        // 2-column grid of prompt chips. Each chip is a small action
        // card — feels like suggestion, not a list of options.
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            examplePrompts.chunked(2).forEach { row ->
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    row.forEach { prompt ->
                        ExamplePromptChip(
                            prompt = prompt,
                            onClick = { onPromptClick(prompt.prompt) },
                            modifier = Modifier.weight(1f),
                        )
                    }
                    if (row.size == 1) Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun ExamplePromptChip(
    prompt: ExamplePrompt,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        shape = RoundedCornerShape(16.dp),
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .clickable { onClick() },
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Icon(
                imageVector = prompt.icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(18.dp),
            )
            Text(
                text = prompt.label,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = prompt.prompt,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                maxLines = 2,
            )
        }
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
                shape = CircleShape,
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier
                        .size(36.dp)
                        .padding(8.dp),
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

/**
 * One-line summary of a memory: the first sentence, hard-capped at
 * 60 chars with an ellipsis. Used by the "What I remember" card
 * so a 200-character memory doesn't take up 4 lines in the brief.
 */
private fun summarizeMemory(content: String): String {
    val firstSentence = content.substringBefore('.').ifBlank { content }.trim()
    return if (firstSentence.length > 60) {
        firstSentence.take(60).trimEnd() + "…"
    } else {
        firstSentence
    }
}