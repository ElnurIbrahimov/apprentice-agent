package com.aura.widget

import android.appwidget.AppWidgetManager
import android.content.ComponentName
import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import android.widget.RemoteViews
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.MainActivity
import com.aura.R
import com.aura.agent.Turn
import com.aura.ui.theme.AuraTheme
import com.aura.ui.viewmodel.ChatViewModel
import com.aura.ui.viewmodel.ModelSelectionState
import dagger.hilt.android.AndroidEntryPoint

internal fun buildQuickAskSystemPrompt(prefix: String): String = buildString {
    append(
        "This is a compact Aura session opened from the home-screen widget. " +
            "Answer directly and keep ordinary answers brief, but use tools, memory, " +
            "and deeper reasoning whenever the question needs them.",
    )
    if (prefix.isNotBlank()) {
        append("\n\nWidget instruction:\n")
        append(prefix.trim())
    }
}

internal fun latestQuickAskResponse(turns: List<Turn>): String =
    turns.asReversed().firstNotNullOfOrNull { it.assistant?.takeIf(String::isNotBlank) }.orEmpty()

/**
 * Compact transparent surface opened by [AskAuraWidget]. It deliberately uses
 * [ChatViewModel], the same agentic pipeline as the full chat: memory recall,
 * tools, specialists, KG/profile extraction, persistence, and provider failover.
 */
@AndroidEntryPoint
class QuickAskActivity : ComponentActivity() {
    private var appWidgetId: Int = AppWidgetManager.INVALID_APPWIDGET_ID

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        appWidgetId = intent?.extras?.getInt(
            AppWidgetManager.EXTRA_APPWIDGET_ID,
            AppWidgetManager.INVALID_APPWIDGET_ID,
        ) ?: AppWidgetManager.INVALID_APPWIDGET_ID

        window.setDimAmount(0.5f)
        window.addFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND)
        window.statusBarColor = android.graphics.Color.TRANSPARENT

        setContent {
            AuraTheme {
                val viewModel: ChatViewModel = hiltViewModel()
                QuickAskContent(
                    viewModel = viewModel,
                    appWidgetId = appWidgetId,
                    widgetPrefix = WidgetConfig.prefixFor(this, appWidgetId),
                    configuredWidgetModel = WidgetConfig.modelFor(this, appWidgetId, ""),
                    onCompleted = ::updateWidgetWithResponse,
                    onOpenFullChat = {
                        startActivity(Intent(this, MainActivity::class.java))
                        finish()
                    },
                    onDismiss = ::finish,
                )
            }
        }
    }

    private fun updateWidgetWithResponse(query: String, response: String) {
        val manager = AppWidgetManager.getInstance(this)
        val ids = if (appWidgetId != AppWidgetManager.INVALID_APPWIDGET_ID) {
            intArrayOf(appWidgetId)
        } else {
            manager.getAppWidgetIds(ComponentName(this, AskAuraWidget::class.java))
        }
        for (id in ids) {
            val views = RemoteViews(packageName, R.layout.widget_ask_aura)
            views.setTextViewText(
                R.id.widget_memory_text,
                "Q: ${query.take(100)}\nA: ${response.take(150)}",
            )
            manager.updateAppWidget(id, views)
        }
    }
}

@Composable
private fun QuickAskContent(
    viewModel: ChatViewModel,
    appWidgetId: Int,
    widgetPrefix: String,
    configuredWidgetModel: String,
    onCompleted: (String, String) -> Unit,
    onOpenFullChat: () -> Unit,
    onDismiss: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
    var query by remember { mutableStateOf("") }
    var submittedQuery by remember { mutableStateOf<String?>(null) }
    var lastPublishedResponse by remember { mutableStateOf("") }
    var initialized by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        if (!initialized) {
            viewModel.startIsolatedSession(
                systemPrompt = buildQuickAskSystemPrompt(widgetPrefix),
                model = configuredWidgetModel.takeIf(String::isNotBlank),
                title = if (appWidgetId == AppWidgetManager.INVALID_APPWIDGET_ID) {
                    "Quick Ask"
                } else {
                    "Quick Ask · Widget $appWidgetId"
                },
            )
            initialized = true
        }
    }

    val response = latestQuickAskResponse(state.conversation.turns)
    LaunchedEffect(state.streaming, response, submittedQuery) {
        val sent = submittedQuery
        if (!state.streaming && sent != null && response.isNotBlank() && response != lastPublishedResponse) {
            lastPublishedResponse = response
            onCompleted(sent, response)
        }
    }

    val sendEnabled = query.isNotBlank() &&
        !state.streaming &&
        state.modelSelection is ModelSelectionState.Ready
    val submit = {
        if (sendEnabled) {
            val cleaned = query.trim()
            submittedQuery = cleaned
            lastPublishedResponse = ""
            viewModel.onUserMessage(cleaned)
        }
    }

    Surface(
        modifier = Modifier.fillMaxWidth().padding(16.dp),
        shape = RoundedCornerShape(24.dp),
        tonalElevation = 8.dp,
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Ask Aura", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "Full memory + tools",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
                TextButton(onClick = onDismiss) { Text("Close") }
            }

            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                placeholder = { Text("Ask anything…") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { submit() }),
                enabled = !state.streaming,
            )

            when {
                state.streaming && response.isBlank() -> Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Text("Aura is thinking…", style = MaterialTheme.typography.bodySmall)
                }
                state.error != null -> Surface(
                    color = MaterialTheme.colorScheme.errorContainer,
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        state.error.orEmpty(),
                        modifier = Modifier.padding(12.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                response.isNotBlank() -> Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        response,
                        modifier = Modifier.padding(14.dp),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                state.modelSelection !is ModelSelectionState.Ready -> Text(
                    "Choose and verify a model in Aura Settings first.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
            ) {
                if (response.isNotBlank()) {
                    OutlinedButton(onClick = onOpenFullChat) { Text("Open chat") }
                }
                Button(onClick = submit, enabled = sendEnabled) {
                    Text(if (state.streaming) "Thinking" else "Send")
                }
            }
        }
    }
}
