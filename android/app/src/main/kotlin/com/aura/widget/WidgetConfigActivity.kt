package com.aura.widget

import android.appwidget.AppWidgetManager
import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.aura.data.UserPreferences
import com.aura.providers.ProviderRegistry
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import javax.inject.Inject

/**
 * Configuration activity for the AskAuraWidget. Launched when the
 * user places the widget on the home screen. Lets them pick which
 * model the widget uses and set an optional prompt prefix.
 *
 * Config is stored in SharedPreferences keyed by widget ID so each
 * widget instance can have different settings.
 */
@AndroidEntryPoint
class WidgetConfigActivity : ComponentActivity() {

    @Inject lateinit var providerRegistry: ProviderRegistry
    @Inject lateinit var userPreferences: UserPreferences

    private var appWidgetId = AppWidgetManager.INVALID_APPWIDGET_ID

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setResult(RESULT_CANCELED)

        appWidgetId = intent?.extras?.getInt(
            AppWidgetManager.EXTRA_APPWIDGET_ID,
            AppWidgetManager.INVALID_APPWIDGET_ID,
        ) ?: AppWidgetManager.INVALID_APPWIDGET_ID

        if (appWidgetId == AppWidgetManager.INVALID_APPWIDGET_ID) {
            finish()
            return
        }

        val configuredModels = runBlocking {
            providerRegistry.configured().flatMap { p ->
                p.listModels().map { "${p.prefix}:$it" }
            }
        }
        val defaultModel = runBlocking { userPreferences.defaultModel.first() }
        val currentConfig = WidgetConfig.load(this, appWidgetId)
        val currentModel = currentConfig?.model ?: defaultModel
        val currentPrefix = currentConfig?.promptPrefix ?: ""

        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                WidgetConfigContent(
                    availableModels = configuredModels,
                    initialModel = currentModel,
                    initialPrefix = currentPrefix,
                    onSave = { model, prefix ->
                        WidgetConfig.save(this, appWidgetId, model, prefix)
                        val resultValue = android.content.Intent().apply {
                            putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, appWidgetId)
                        }
                        setResult(RESULT_OK, resultValue)
                        finish()
                    },
                    onCancel = { finish() },
                )
            }
        }
    }
}

data class WidgetConfigData(val model: String, val promptPrefix: String)

object WidgetConfig {
    private fun prefs(context: Context) =
        context.getSharedPreferences("widget_config", Context.MODE_PRIVATE)

    fun load(context: Context, widgetId: Int): WidgetConfigData? {
        val model = prefs(context).getString("model_$widgetId", null) ?: return null
        val prefix = prefs(context).getString("prefix_$widgetId", "") ?: ""
        return WidgetConfigData(model, prefix)
    }

    fun save(context: Context, widgetId: Int, model: String, prefix: String) {
        prefs(context).edit()
            .putString("model_$widgetId", model)
            .putString("prefix_$widgetId", prefix)
            .apply()
    }

    fun modelFor(context: Context, widgetId: Int, default: String): String =
        prefs(context).getString("model_$widgetId", default) ?: default

    fun prefixFor(context: Context, widgetId: Int): String =
        prefs(context).getString("prefix_$widgetId", "") ?: ""
}

@Composable
private fun WidgetConfigContent(
    availableModels: List<String>,
    initialModel: String,
    initialPrefix: String,
    onSave: (String, String) -> Unit,
    onCancel: () -> Unit,
) {
    var selectedModel by remember { mutableStateOf(initialModel) }
    var prefix by remember { mutableStateOf(initialPrefix) }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("Configure widget", style = MaterialTheme.typography.headlineSmall)
            Text("Pick a model and optional prompt prefix.", style = MaterialTheme.typography.bodySmall)

            Text("Model", style = MaterialTheme.typography.labelLarge)
            availableModels.forEach { model ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    androidx.compose.material3.RadioButton(
                        selected = model == selectedModel,
                        onClick = { selectedModel = model },
                    )
                    Text(
                        text = com.aura.ui.util.modelDisplayName(model),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))
            Text("Prompt prefix (optional)", style = MaterialTheme.typography.labelLarge)
            OutlinedTextField(
                value = prefix,
                onValueChange = { prefix = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("e.g. Be concise.") },
                singleLine = true,
            )

            Spacer(modifier = Modifier.height(16.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(onClick = onCancel) { Text("Cancel") }
                Spacer(modifier = Modifier.height(0.dp).padding(horizontal = 8.dp))
                Button(onClick = { onSave(selectedModel, prefix) }) { Text("Save") }
            }
        }
    }
}