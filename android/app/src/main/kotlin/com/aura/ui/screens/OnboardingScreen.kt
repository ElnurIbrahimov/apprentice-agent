package com.aura.ui.screens

import androidx.compose.animation.AnimatedContent
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.ui.settings.ProviderKeyField
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.FirstRunGate
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class OnboardingUiState(
    val ollamaKey: String = "",
    val anthropicKey: String = "",
    val configuredCount: Int = 0,
    val verifying: Boolean = false,
    val verifyResult: String? = null,
)

@HiltViewModel
class OnboardingViewModel @Inject constructor(
    private val firstRunGate: FirstRunGate,
    private val providerKeys: ProviderKeys,
    private val providerRegistry: ProviderRegistry,
) : ViewModel() {

    private val _state = MutableStateFlow(OnboardingUiState())
    val state: StateFlow<OnboardingUiState> = _state.asStateFlow()

    fun saveOllamaKey(key: String) {
        _state.value = _state.value.copy(ollamaKey = key)
        viewModelScope.launch { providerKeys.set("ollama", key) }
    }

    fun saveAnthropicKey(key: String) {
        _state.value = _state.value.copy(anthropicKey = key)
        viewModelScope.launch { providerKeys.set("anthropic", key) }
    }

    fun refreshConfigured() {
        viewModelScope.launch {
            _state.value = _state.value.copy(
                configuredCount = providerRegistry.configured().size
            )
        }
    }

    /**
     * Verify the saved API key by hitting the provider's models endpoint. If
     * the call succeeds, the key is good. If it fails, show the error.
     */
    fun verifyKey(prefix: String) {
        if (providerRegistry.configured().none { it.prefix == prefix }) {
            _state.value = _state.value.copy(verifyResult = "✗ No key saved for $prefix")
            return
        }
        _state.value = _state.value.copy(verifying = true, verifyResult = null)
        viewModelScope.launch {
            val provider = providerRegistry.all().firstOrNull { it.prefix == prefix }
            val result = if (provider == null) {
                "✗ Provider $prefix not found"
            } else {
                runCatching { provider.listModels() }
                    .map { "✓ Verified — ${it.size} models available" }
                    .getOrElse { "✗ Failed: ${it.message?.take(80)}" }
            }
            _state.value = _state.value.copy(verifying = false, verifyResult = result)
        }
    }

    fun finish() {
        viewModelScope.launch { firstRunGate.markComplete() }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OnboardingScreen(
    viewModel: OnboardingViewModel = hiltViewModel(),
    onComplete: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
    var page by remember { mutableIntStateOf(0) }

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = {},
            navigationIcon = {
                if (page > 0) {
                    IconButton(onClick = { page-- }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            },
            actions = {
                if (page < 2) {
                    TextButton(onClick = onComplete) {
                        Text("Skip", color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
                    }
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.background,
            ),
        )

        AnimatedContent(targetState = page, label = "onboarding-page") { currentPage ->
            when (currentPage) {
                0 -> PageWelcome(
                    onNext = {
                        page = 1
                        viewModel.refreshConfigured()
                    }
                )
                1 -> PageKeys(
                    ollamaKey = state.ollamaKey,
                    anthropicKey = state.anthropicKey,
                    configuredCount = state.configuredCount,
                    verifying = state.verifying,
                    verifyResult = state.verifyResult,
                    onOllamaKeyChange = viewModel::saveOllamaKey,
                    onAnthropicKeyChange = viewModel::saveAnthropicKey,
                    onRefreshConfigured = viewModel::refreshConfigured,
                    onVerifyOllama = { viewModel.verifyKey("ollama") },
                    onVerifyAnthropic = { viewModel.verifyKey("anthropic") },
                    onNext = { page = 2 },
                )
                2 -> PageDone(
                    configuredCount = state.configuredCount,
                    onFinish = {
                        viewModel.finish()
                        onComplete()
                    },
                )
            }
        }
    }
}

@Composable
private fun PageWelcome(onNext: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "👋",
            style = MaterialTheme.typography.displayLarge,
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            text = "Welcome to Aura",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "A personal AI assistant that runs on your phone. It remembers things, manages tasks, delegates work to specialists, and can see your calendar, photos, and notifications.\n\nTo think, it needs an LLM. You can use Ollama Cloud, Anthropic, OpenAI, or DeepSeek.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(48.dp))
        Button(
            onClick = onNext,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Set up a provider")
            Spacer(modifier = Modifier.width(8.dp))
            Icon(Icons.AutoMirrored.Filled.ArrowForward, contentDescription = null)
        }
    }
}

@Composable
private fun PageKeys(
    ollamaKey: String,
    anthropicKey: String,
    configuredCount: Int,
    verifying: Boolean,
    verifyResult: String?,
    onOllamaKeyChange: (String) -> Unit,
    onAnthropicKeyChange: (String) -> Unit,
    onRefreshConfigured: () -> Unit,
    onVerifyOllama: () -> Unit,
    onVerifyAnthropic: () -> Unit,
    onNext: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
    ) {
        Text(
            text = "Connect a provider",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Start with Ollama Cloud (cheapest, no quota management). You can add more later in Settings.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(modifier = Modifier.height(24.dp))

        ProviderKeyField(
            label = "Ollama Cloud",
            value = ollamaKey,
            onValueChange = { onOllamaKeyChange(it); onRefreshConfigured() },
            helperText = "Get a key at ollama.com/settings/keys — starts at \$5/mo",
        )

        Spacer(modifier = Modifier.height(16.dp))

        ProviderKeyField(
            label = "Anthropic (optional)",
            value = anthropicKey,
            onValueChange = { onAnthropicKeyChange(it); onRefreshConfigured() },
            helperText = "Get a key at console.anthropic.com/settings/keys",
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Verify buttons — let the user prove the key works
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onVerifyOllama, enabled = ollamaKey.isNotBlank() && !verifying) {
                Text(if (verifying) "Checking…" else "Test Ollama key")
            }
            OutlinedButton(onClick = onVerifyAnthropic, enabled = anthropicKey.isNotBlank() && !verifying) {
                Text(if (verifying) "Checking…" else "Test Anthropic key")
            }
        }

        verifyResult?.let { msg ->
            Spacer(Modifier.height(8.dp))
            val color = if (msg.startsWith("✓")) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
            Text(msg, color = color, style = MaterialTheme.typography.bodySmall)
        }

        Spacer(modifier = Modifier.height(24.dp))

        if (configuredCount > 0) {
            Surface(
                color = MaterialTheme.colorScheme.primaryContainer,
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "$configuredCount provider${if (configuredCount > 1) "s" else ""} configured",
                    modifier = Modifier.padding(12.dp),
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            OutlinedButton(onClick = onNext) {
                Text("I'll do this later")
            }
            Button(onClick = onNext) {
                Text(if (configuredCount > 0) "Continue" else "Skip for now")
            }
        }
    }
}

@Composable
private fun PageDone(
    configuredCount: Int,
    onFinish: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "✨",
            style = MaterialTheme.typography.displayLarge,
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            text = "You're all set",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = if (configuredCount > 0)
                "$configuredCount provider${if (configuredCount > 1) "s" else ""} configured. Aura is ready to think."
            else
                "No providers configured yet — you can add them anytime in Settings.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "Tip: say \"my name is ___\" or \"I prefer dark mode\" — Aura will remember it.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(48.dp))
        Button(
            onClick = onFinish,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Start chatting")
        }
    }
}
