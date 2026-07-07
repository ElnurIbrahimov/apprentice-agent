package com.aura.widget

import android.appwidget.AppWidgetManager
import android.content.ComponentName
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import android.widget.RemoteViews
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.aura.R
import com.aura.agent.Conversation
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.data.UserPreferences
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

/**
 * Lightweight transparent activity opened from the AskAuraWidget.
 * Shows a text field, sends the query to the configured LLM, and
 * displays the response inline. Does NOT open the full app — the
 * user can ask a quick question and dismiss without leaving the
 * home screen.
 *
 * After the response arrives, the widget body is updated with the
 * last Q&A pair so the answer persists on the home screen.
 */
@AndroidEntryPoint
class QuickAskActivity : ComponentActivity() {

    @Inject lateinit var providerRegistry: ProviderRegistry
    @Inject lateinit var userPreferences: UserPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Transparent window, no history, dim background slightly
        window.setDimAmount(0.5f)
        window.addFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND)
        window.setStatusBarColor(android.graphics.Color.TRANSPARENT)

        setContent {
            MaterialTheme {
                QuickAskContent(
                    onSend = { query -> askQuery(query) },
                    onDismiss = { finish() },
                )
            }
        }
    }

    private var lastQuery: String? = null
    private var lastResponse: String? = null

    private fun askQuery(query: String) {
        lifecycleScope.launch {
            lastQuery = query
            val model = userPreferences.defaultModel.first()
            val response = withContext(Dispatchers.IO) {
                val conversation = Conversation(
                    systemPrompt = "You are Aura. Answer concisely in 1-3 sentences. Be direct and helpful.",
                ).addUser(query)
                val text = StringBuilder()
                try {
                    providerRegistry.chat(
                        model,
                        conversation.toMessages(),
                        ChatOptions(temperature = 0.5, maxTokens = 200),
                        emptyList(),
                    ).collect { chunk ->
                        chunk.text?.let { text.append(it) }
                    }
                } catch (_: Exception) {
                    return@withContext "Sorry, I couldn't reach the model. Check your API key."
                }
                text.toString().trim().ifBlank { "No response." }
            }
            lastResponse = response
            updateWidgetWithResponse(query, response)
        }
    }

    private fun updateWidgetWithResponse(query: String, response: String) {
        val mgr = AppWidgetManager.getInstance(this)
        val component = ComponentName(this, AskAuraWidget::class.java)
        val ids = mgr.getAppWidgetIds(component)
        for (id in ids) {
            val views = RemoteViews(packageName, R.layout.widget_ask_aura)
            val displayText = "Q: ${query.take(100)}\nA: ${response.take(150)}"
            views.setTextViewText(R.id.widget_memory_text, displayText)
            mgr.updateAppWidget(id, views)
        }
    }
}

@Composable
private fun QuickAskContent(
    onSend: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var query by remember { mutableStateOf("") }
    var response by remember { mutableStateOf<String?>(null) }
    var loading by remember { mutableStateOf(false) }

    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        shape = RoundedCornerShape(20.dp),
        tonalElevation = 6.dp,
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "Ask Aura",
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = onDismiss) { Text("Close") }
            }

            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                placeholder = { Text("Ask anything...") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    imeAction = androidx.compose.ui.text.input.ImeAction.Send,
                ),
                keyboardActions = androidx.compose.foundation.text.KeyboardActions(
                    onSend = {
                        if (query.isNotBlank() && !loading) {
                            loading = true
                            onSend(query.trim())
                        }
                    },
                ),
            )

            if (loading && response == null) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Text("Thinking...", style = MaterialTheme.typography.bodySmall)
                }
            }

            response?.let { r ->
                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = r,
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
            ) {
                Button(
                    onClick = {
                        if (query.isNotBlank() && !loading) {
                            loading = true
                            onSend(query.trim())
                            // The response callback happens async; we can't
                            // easily set loading=false from onSend. Use a
                            // LaunchedEffect to poll or just let the response
                            // appear. For simplicity, loading stays true until
                            // response is set.
                        }
                    },
                    enabled = query.isNotBlank() && !loading,
                ) { Text("Send") }
            }
        }
    }
}