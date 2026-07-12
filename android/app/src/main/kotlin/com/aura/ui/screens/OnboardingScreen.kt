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
import androidx.compose.material.icons.filled.Hub
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.TaskAlt
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
import com.aura.providers.MoaPresetRepository
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import androidx.compose.ui.draw.scale
import androidx.compose.foundation.layout.Box
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.RepeatMode

data class OnboardingUiState(
    val ollamaKey: String = "",
    val anthropicKey: String = "",
    val configuredCount: Int = 0,
    val verifying: Boolean = false,
    val verifyResult: String? = null,
    /**
     * True when the default MoA preset's reference models + aggregator
     * are all from providers the user has configured. Drives whether
     * the PageDone "MoA — Mixture of Agents" card is shown. The card
     * used to render unconditionally, which lied to anyone with
     * fewer than the required providers configured.
     */
    val moaAvailable: Boolean = false,
)

@HiltViewModel
class OnboardingViewModel @Inject constructor(
    private val firstRunGate: FirstRunGate,
    private val providerKeys: ProviderKeys,
    private val providerRegistry: ProviderRegistry,
    private val moaPresetRepository: MoaPresetRepository,
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
            val configured = providerRegistry.configured()
            _state.value = _state.value.copy(
                configuredCount = configured.size,
                moaAvailable = computeMoaAvailable(configured.map { it.prefix }),
            )
        }
    }

    /**
     * MoA is available when the default preset's reference models and
     * aggregator are all from providers the user has configured.
     * Default preset (per MoaPresetRepository): 2× ollama references +
     * 1× deepseek aggregator. So users need at least ollama + deepseek
     * configured for the default MoA to work.
     */
    private fun computeMoaAvailable(configuredPrefixes: List<String>): Boolean {
        val preset = moaPresetRepository.loadPresets().values.firstOrNull { it.enabled }
            ?: return false
        val requiredPrefixes = preset.referenceModels.map { it.providerPrefix }.toSet() +
            preset.aggregator.providerPrefix
        return configuredPrefixes.toSet().containsAll(requiredPrefixes)
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

    fun finish(onComplete: () -> Unit) {
        viewModelScope.launch {
            firstRunGate.markComplete()
            onComplete()
        }
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
                    TextButton(onClick = { viewModel.finish(onComplete) }) {
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
                    moaAvailable = state.moaAvailable,
                    onFinish = { viewModel.finish(onComplete) },
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
        // Animated brand mark — primary teal circle with a star
        // inside. The whole thing pulses gently to feel alive.
        val transition = androidx.compose.animation.core.rememberInfiniteTransition(label = "brand")
        val scale by transition.animateFloat(
            initialValue = 1f,
            targetValue = 1.05f,
            animationSpec = androidx.compose.animation.core.infiniteRepeatable(
                animation = androidx.compose.animation.core.tween(2000),
                repeatMode = androidx.compose.animation.core.RepeatMode.Reverse,
            ),
            label = "brand-scale",
        )
        Surface(
            color = MaterialTheme.colorScheme.primary,
            shape = androidx.compose.foundation.shape.CircleShape,
            shadowElevation = 12.dp,
            modifier = Modifier
                .size(112.dp)
                .scale(scale),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(
                    text = "✦",
                    style = MaterialTheme.typography.displayLarge,
                    color = MaterialTheme.colorScheme.onPrimary,
                )
            }
        }
        Spacer(modifier = Modifier.height(32.dp))
        Text(
            text = "Aura",
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Your personal AI assistant",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(32.dp))
        // Feature pills — three short promises
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            FeaturePill(icon = Icons.Filled.Memory, text = "Remembers what matters to you")
            FeaturePill(icon = Icons.Filled.TaskAlt, text = "Manages tasks and calendar")
            FeaturePill(icon = Icons.Filled.Hub, text = "Connects to the best models")
            FeaturePill(icon = Icons.Filled.Lock, text = "Keys stay on your device")
        }
        Spacer(modifier = Modifier.height(40.dp))
        Button(
            onClick = onNext,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = RoundedCornerShape(16.dp),
        ) {
            Text(
                text = "Get started",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
        }
    }
}

@Composable
private fun FeaturePill(icon: androidx.compose.ui.graphics.vector.ImageVector, text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = androidx.compose.foundation.shape.CircleShape,
            modifier = Modifier.size(36.dp),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(imageVector = icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
            }
        }
        Text(
            text = text,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.85f),
        )
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
    moaAvailable: Boolean,
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
        Spacer(modifier = Modifier.height(16.dp))
        // MoA card. Only shown when the user has configured providers
        // for the default MoA preset (ollama + deepseek per
        // MoaPresetRepository). Showing it when MoA isn't actually
        // usable was a lie — the user would tap "🚀 Deep" and get a
        // 401 from the unconfigured aggregator.
        if (moaAvailable) {
            Surface(
                color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = "🧠 MoA — Mixture of Agents",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Aura can use 3 AI models together for harder problems. Tap \"🚀 Deep\" in chat to enable it — the models collaborate to give you better answers.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.8f),
                        textAlign = TextAlign.Center,
                    )
                }
            }
        } else {
            // Honest substitute: tell the user what MoA needs so they
            // can add the missing providers in Settings later.
            Text(
                text = "Tip: add more providers in Settings to unlock Mixture of Agents — it runs 3 models together for harder questions.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
                textAlign = TextAlign.Center,
            )
        }
        Spacer(modifier = Modifier.height(48.dp))
        Button(
            onClick = onFinish,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Start chatting")
        }
    }
}
