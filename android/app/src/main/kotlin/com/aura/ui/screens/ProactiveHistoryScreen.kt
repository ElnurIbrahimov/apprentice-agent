package com.aura.ui.screens

import android.content.Intent
import android.provider.CalendarContract
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.WindowInsets
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.HistoryToggleOff
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
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
import androidx.compose.ui.graphics.vector.ImageVector
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

/**
 * Category tag used by [HistoryCard] to pick the right accent colours
 * for the icon container — no colour values live in the model itself.
 */
enum class HistoryCardKind { MorningBrief, CalendarEvent, MemoryDecay }

/**
 * Polished icon + data model for a single proactive-history card.
 * [icon] is a Material [ImageVector] for a consistent brand look.
 * [kind] drives accent colours inside the card composable.
 */
data class HistoryCardModel(
    val icon: ImageVector,
    val kind: HistoryCardKind,
    val title: String,
    val body: String,
    val badge: Pair<String, Int>? = null,           // (label, count) e.g. ("recalled", 5)
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
        contentWindowInsets = WindowInsets(0),
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Filled.HistoryToggleOff,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(22.dp),
                        )
                        Spacer(modifier = Modifier.width(10.dp))
                        Text("Proactive history")
                    }
                },
            )
        },
    ) { padding ->
        val status by viewModel.status.collectAsState()
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            val reversed = state.events.reversed()
            if (reversed.isEmpty()) {
                item { EmptyState() }
            }
            items(reversed) { event ->
                val model = event.toCardModel(context)
                HistoryCard(
                    model = model,
                    modifier = if (model.onClick != null) Modifier.clickable(onClick = model.onClick) else Modifier,
                )
            }
            item {
                DebugSection(
                    status = status,
                    onFireBrief = { viewModel.fireMorningBrief() },
                    onFireDecay = { viewModel.fireDecayPass() },
                    onFireCalendar = { viewModel.fireCalendarCheck() },
                    onClearStatus = { viewModel.clearStatus() },
                )
            }
        }
    }
}

// ── Empty state ────────────────────────────────────────────────────────────

@Composable
private fun EmptyState() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 64.dp, bottom = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            imageVector = Icons.Filled.HistoryToggleOff,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.25f),
            modifier = Modifier.size(56.dp),
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "No proactive events yet",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "Morning briefs, calendar reminders, and memory insights will appear here.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.38f),
            modifier = Modifier.padding(horizontal = 40.dp),
        )
    }
}

// ── Event → model mapping ──────────────────────────────────────────────────

private fun ProactiveEventBus.Event.toCardModel(context: android.content.Context): HistoryCardModel = when (this) {
    is ProactiveEventBus.Event.MorningBriefReady -> HistoryCardModel(
        icon = Icons.Filled.WbSunny,
        kind = HistoryCardKind.MorningBrief,
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
        icon = Icons.Filled.WbSunny,
        kind = HistoryCardKind.MorningBrief,
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
            icon = Icons.Filled.CalendarMonth,
            kind = HistoryCardKind.CalendarEvent,
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
    is ProactiveEventBus.Event.MemoryDecayWarning -> HistoryCardModel(
        icon = Icons.Filled.Psychology,
        kind = HistoryCardKind.MemoryDecay,
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

// ── History card composable ─────────────────────────────────────────────────

@Composable
private fun HistoryCard(
    model: HistoryCardModel,
    modifier: Modifier = Modifier,
) {
    val fmt = SimpleDateFormat("MMM d, HH:mm", Locale.US)

    // Resolve accent colours by event kind
    val (iconTint, iconBackground) = when (model.kind) {
        HistoryCardKind.MorningBrief, HistoryCardKind.MemoryDecay ->
            MaterialTheme.colorScheme.primary to MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
        HistoryCardKind.CalendarEvent ->
            MaterialTheme.colorScheme.secondary to MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f)
    }
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(14.dp),
        tonalElevation = 0.dp,
        modifier = modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(14.dp),
            verticalAlignment = Alignment.Top,
        ) {
            // ── Icon container (branded circle) ──────────────────────
            Surface(
                color = iconBackground,
                shape = CircleShape,
            ) {
                Icon(
                    imageVector = model.icon,
                    contentDescription = null,
                    tint = iconTint,
                    modifier = Modifier
                        .size(40.dp)
                        .padding(8.dp),
                )
            }
            Spacer(modifier = Modifier.width(14.dp))
            Column(modifier = Modifier.weight(1f)) {
                // ── Title row ────────────────────────────────────────
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = model.title,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(1f),
                    )
                    model.badge?.let { (label, count) ->
                        Surface(
                            color = iconBackground,
                            shape = RoundedCornerShape(6.dp),
                        ) {
                            Text(
                                text = "$count $label",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Medium,
                                color = iconTint,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(4.dp))
                // ── Body ─────────────────────────────────────────────
                Text(
                    text = model.body,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.85f),
                )
                // ── Footer row: tap hint + timestamp ─────────────────
                Spacer(modifier = Modifier.height(6.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    model.tapHint?.let { hint ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Filled.PlayArrow,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f),
                                modifier = Modifier.size(12.dp),
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = hint,
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Medium,
                                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.8f),
                            )
                        }
                    }
                    Spacer(modifier = Modifier.weight(1f))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Filled.Schedule,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.38f),
                            modifier = Modifier.size(12.dp),
                        )
                        Spacer(modifier = Modifier.width(3.dp))
                        Text(
                            text = fmt.format(Date(model.timestamp)),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.45f),
                        )
                    }
                }
            }
        }
    }
}

// ── Debug section with branded action rows ─────────────────────────────────

@Composable
private fun DebugSection(
    status: String?,
    onFireBrief: () -> Unit,
    onFireDecay: () -> Unit,
    onFireCalendar: () -> Unit,
    onClearStatus: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        HorizontalDivider(
            color = MaterialTheme.colorScheme.outlineVariant,
        )
        Spacer(modifier = Modifier.height(4.dp))

        // ── Section header ──────────────────────────────────────────────
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Filled.Refresh,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(18.dp),
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "Debug · fire now",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary,
            )
        }
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = "Run each loop on demand without waiting for its scheduled interval.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f),
        )
        Spacer(modifier = Modifier.height(4.dp))

        // ── Action: Morning brief ───────────────────────────────────────
        DebugActionRow(
            icon = Icons.Filled.WbSunny,
            iconTint = MaterialTheme.colorScheme.primary,
            iconBackground = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
            label = "Morning brief",
            description = "Generate and deliver the daily morning summary",
            onClick = onFireBrief,
        )
        // ── Action: Memory decay ────────────────────────────────────────
        DebugActionRow(
            icon = Icons.Filled.DeleteSweep,
            iconTint = MaterialTheme.colorScheme.primary,
            iconBackground = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
            label = "Memory decay pass",
            description = "Prune and consolidate aged memories",
            onClick = onFireDecay,
        )
        // ── Action: Calendar check ──────────────────────────────────────
        DebugActionRow(
            icon = Icons.Filled.Notifications,
            iconTint = MaterialTheme.colorScheme.secondary,
            iconBackground = MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f),
            label = "Calendar check",
            description = "Scan for upcoming calendar events",
            onClick = onFireCalendar,
        )

        // ── Status message ──────────────────────────────────────────────
        status?.let { msg ->
            Spacer(modifier = Modifier.height(4.dp))
            Surface(
                color = if (msg.contains("Error", ignoreCase = true))
                    MaterialTheme.colorScheme.errorContainer
                else
                    MaterialTheme.colorScheme.tertiaryContainer,
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        imageVector = if (msg.contains("Error", ignoreCase = true))
                            Icons.Filled.Clear
                        else
                            Icons.Filled.CheckCircle,
                        contentDescription = null,
                        tint = if (msg.contains("Error", ignoreCase = true))
                            MaterialTheme.colorScheme.error
                        else
                            MaterialTheme.colorScheme.tertiary,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = msg,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        text = "Dismiss",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.clickable { onClearStatus() },
                    )
                }
            }
        }
    }
}

/**
 * A single debug action row with a branded icon container, label,
 * description, and a compact "Run" button.
 */
@Composable
private fun DebugActionRow(
    icon: ImageVector,
    iconTint: androidx.compose.ui.graphics.Color,
    iconBackground: androidx.compose.ui.graphics.Color,
    label: String,
    description: String,
    onClick: () -> Unit,
) {
    Surface(
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(12.dp),
        tonalElevation = 0.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Branded icon circle
            Surface(
                color = iconBackground,
                shape = CircleShape,
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = iconTint,
                    modifier = Modifier
                        .size(36.dp)
                        .padding(7.dp),
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = label,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f),
                )
            }
            Button(
                onClick = onClick,
                colors = ButtonDefaults.buttonColors(
                    containerColor = iconBackground,
                    contentColor = iconTint,
                ),
                shape = RoundedCornerShape(8.dp),
                contentPadding = ButtonDefaults.TextButtonContentPadding,
                modifier = Modifier.height(34.dp),
            ) {
                Icon(
                    imageVector = Icons.Filled.PlayArrow,
                    contentDescription = null,
                    modifier = Modifier.size(14.dp),
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Run",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
    }
}
