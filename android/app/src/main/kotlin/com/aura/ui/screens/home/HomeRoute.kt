package com.aura.ui.screens.home

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.ui.viewmodel.HomeViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HomeRoute(
    viewModel: HomeViewModel = hiltViewModel(),
    onOpenChat: (prefillDraft: String) -> Unit = {},
    onOpenChatWithBrief: (briefText: String) -> Unit = {},
    onOpenProactive: () -> Unit = {},
    onOpenMemory: () -> Unit = {},
    onOpenTasks: () -> Unit = {},
    onOpenCalendar: () -> Unit = {},
    onOpenReminders: () -> Unit = {},
    onOpenTools: () -> Unit = {},
    onOpenHands: () -> Unit = {},
    onOpenCreative: () -> Unit = {},
) {
    val state by viewModel.state.collectAsState()
    val greeting = remember(state.hour, state.userName) {
        val salutation = when (state.hour) {
            in 5..11 -> "Good morning"
            in 12..17 -> "Good afternoon"
            in 18..21 -> "Good evening"
            else -> "Working late"
        }
        state.userName?.let { "$salutation, $it" } ?: salutation
    }
    val dateLabel = remember {
        SimpleDateFormat("EEEE, MMMM d", Locale.US).format(Date())
    }

    HomeContent(
        state = state,
        greeting = greeting,
        dateLabel = dateLabel,
        onAskAura = onOpenChat,
        onRetry = viewModel::refresh,
        onDismissProactive = viewModel::dismissProactiveEvent,
        onOpenChatWithBrief = onOpenChatWithBrief,
        onOpenMemory = onOpenMemory,
        onOpenTasks = onOpenTasks,
        onOpenCalendar = onOpenCalendar,
        onOpenReminders = onOpenReminders,
        onOpenHands = onOpenHands,
        onOpenTools = onOpenTools,
        onOpenCreative = onOpenCreative,
        onOpenProactive = {
            viewModel.onProactiveHistoryOpened()
            onOpenProactive()
        },
    )
}
