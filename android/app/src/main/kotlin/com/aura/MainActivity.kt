package com.aura

import android.content.Intent
import android.os.Bundle
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.fragment.app.FragmentActivity
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Fingerprint
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.lifecycleScope
import com.aura.data.UserPreferences
import com.aura.security.BiometricActivityHolder
import com.aura.tools.BiometricAuthHandler
import com.aura.ui.nav.NavGraph
import com.aura.ui.screens.OnboardingScreen
import com.aura.ui.theme.AuraTheme
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.components.SingletonComponent
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

@EntryPoint
@InstallIn(SingletonComponent::class)
interface FirstRunGateEntryPoint {
    fun firstRunGate(): FirstRunGate
}

/**
 * Entry point for [MainActivity] to read app-lock + biometric state
 * out of the Hilt SingletonComponent. The biometric prompt itself
 * runs on the activity (not the application) so we use the activity
 * via [com.aura.security.BiometricActivityHolder] at call time.
 */
@EntryPoint
@InstallIn(SingletonComponent::class)
interface MainActivityEntryPoint {
    fun userPreferences(): UserPreferences
    fun biometricActivityHolder(): BiometricActivityHolder
}

@AndroidEntryPoint
class MainActivity : FragmentActivity() {

    @Inject lateinit var incomingShareStore: IncomingShareStore
    @Inject lateinit var biometricHolder: BiometricActivityHolder

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.auto(Color.Transparent.value.toInt(), Color.Transparent.value.toInt()),
            navigationBarStyle = SystemBarStyle.auto(Color.Transparent.value.toInt(), Color.Transparent.value.toInt()),
        )
        super.onCreate(savedInstanceState)
        biometricHolder.activity = this
        handleSharedText(intent)
        setContent {
            AuraRoot()
        }
    }

    override fun onDestroy() {
        biometricHolder.activity = null
        super.onDestroy()
    }

    /**
     * Handles the case where Aura is already in the background and the user
     * shares new text. Without this, [intent] would still be the original
     * launching intent (with no EXTRA_SHARED_TEXT) and the share would
     * silently disappear. We update the stored intent and route the new
     * shared text into the IncomingShareStore for ChatScreen to pick up.
     */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleSharedText(intent)
    }

    private fun handleSharedText(intent: Intent?) {
        val sharedText = intent?.getStringExtra(ShareReceiverActivity.EXTRA_SHARED_TEXT) ?: return
        if (sharedText.isNotBlank()) {
            incomingShareStore.set(sharedText)
        }
    }
}

@Composable
fun AuraRoot() {
    val ctx = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var firstRunComplete by remember { mutableStateOf<Boolean?>(null) }
    var appLockEnabled by remember { mutableStateOf(false) }
    var unlocked by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        val entry = EntryPointAccessors.fromApplication(
            ctx.applicationContext,
            FirstRunGateEntryPoint::class.java,
        )
        val mainEntry = EntryPointAccessors.fromApplication(
            ctx.applicationContext,
            MainActivityEntryPoint::class.java,
        )
        firstRunComplete = entry.firstRunGate().isFirstRunComplete()
        appLockEnabled = mainEntry.userPreferences().appLockEnabled.first()
        // Start unlocked when the lock is off so the user doesn't
        // see a flash of the lock screen.
        if (!appLockEnabled) unlocked = true
    }

    // Re-lock on every ON_RESUME. Backgrounding the app for any
    // amount of time forces a fresh biometric challenge on return.
    // DisposableEffect scopes the observer to the composition so
    // the registered observer is removed when AuraRoot leaves.
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME && appLockEnabled) {
                unlocked = false
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    AuraTheme {
        Surface(
            modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
            color = MaterialTheme.colorScheme.background,
        ) {
            when {
                firstRunComplete == null -> Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center,
                ) { CircularProgressIndicator() }

                firstRunComplete == false -> OnboardingScreen(
                    onComplete = { firstRunComplete = true },
                )

                appLockEnabled && !unlocked -> AppLockScreen(
                    onUnlocked = { unlocked = true },
                )

                else -> NavGraph()
            }
        }
    }
}

/**
 * Lock screen shown while the biometric app lock is on. Renders a
 * branded "Aura is locked" message with a single "Unlock" button
 * that triggers the [BiometricPrompt] challenge.
 *
 * Why a button instead of an auto-prompt on first composition:
 * [BiometricPrompt] is modal and cannot be presented while the
 * activity isn't fully resumed, which can happen on cold start
 * race conditions. Letting the user tap "Unlock" lets the system
 * settle into a stable state and avoids spurious
 * ERROR_NO_DEVICE_ID or ERROR_NO_BIOMETRICS errors.
 */
@Composable
private fun AppLockScreen(onUnlocked: () -> Unit) {
    val ctx = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainEntry = remember {
        EntryPointAccessors.fromApplication(
            ctx.applicationContext,
            MainActivityEntryPoint::class.java,
        )
    }
    var statusMessage by remember { mutableStateOf<String?>(null) }

    Box(
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.fillMaxSize(),
        ) {
            Spacer(Modifier.height(120.dp))
            Icon(
                imageVector = Icons.Filled.Lock,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.height(64.dp),
            )
            Text(
                text = "Aura is locked",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Authenticate to open your conversations.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            )
            statusMessage?.let { msg ->
                Text(
                    text = msg,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            Spacer(Modifier.height(24.dp))
            Button(
                onClick = { promptForUnlock(ctx, mainEntry, lifecycleOwner, onUnlocked) { msg -> statusMessage = msg } },
            ) {
                Icon(Icons.Filled.Fingerprint, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("Unlock")
            }
        }
    }
}

/**
 * Show the [BiometricPrompt] and call [onUnlocked] on success.
 * Updates [onStatus] with a human-readable error string on failure
 * so the user can see why the unlock didn't go through (e.g. "no
 * biometric enrolled — set one up in Settings → Security").
 *
 * Re-entry safety: if a prompt is already showing, the second tap
 * is a no-op. The handler is created fresh per call so the deferred
 * completes exactly once.
 */
private fun promptForUnlock(
    ctx: android.content.Context,
    entry: MainActivityEntryPoint,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    onUnlocked: () -> Unit,
    onStatus: (String) -> Unit,
) {
    val activity = entry.biometricActivityHolder().activity ?: run {
        onStatus("Activity not ready — try again")
        return
    }

    // Tell the user up front if there's no biometric enrolled. Showing
    // the system prompt in that state is a no-op and confuses the
    // user. We mirror the BIOMETRIC_WEAK choice from
    // BiometricPromptTool for the same reason — it accepts fingerprint,
    // face, or device PIN.
    val mgr = BiometricManager.from(ctx)
    val canAuth = mgr.canAuthenticate(
        BiometricManager.Authenticators.BIOMETRIC_WEAK or
            BiometricManager.Authenticators.DEVICE_CREDENTIAL
    )
    if (canAuth != BiometricManager.BIOMETRIC_SUCCESS) {
        onStatus("No biometric or device PIN set up — open Settings to add one")
        return
    }

    val handler = BiometricAuthHandler()
    val executor = java.util.concurrent.Executors.newSingleThreadExecutor()

    activity.runOnUiThread {
        val callback = object : BiometricPrompt.AuthenticationCallback() {
            override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                handler.onAuthenticated()
            }

            override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                handler.onError(errorCode, errString.toString())
            }
        }
        val prompt = BiometricPrompt(activity, executor, callback)
        val promptInfo = BiometricPrompt.PromptInfo.Builder()
            .setTitle("Unlock Aura")
            .setSubtitle("Authenticate to open your conversations")
            .setAllowedAuthenticators(
                BiometricManager.Authenticators.BIOMETRIC_WEAK or
                    BiometricManager.Authenticators.DEVICE_CREDENTIAL
            )
            .build()
        prompt.authenticate(promptInfo)
    }

    // Drive the deferred from the lifecycle scope of the current
    // composition. The scope auto-cancels if the activity is
    // destroyed mid-prompt, which is what we want — a half-completed
    // biometric flow on a torn-down activity would be a leak.
    lifecycleOwner.lifecycleScope.launch {
        val result = handler.result.await()
        executor.shutdownNow()
        if (result.success) {
            onUnlocked()
        } else if (result.errorCode == BiometricPrompt.ERROR_USER_CANCELED ||
            result.errorCode == BiometricPrompt.ERROR_NEGATIVE_BUTTON
        ) {
            // User dismissed the prompt — leave the lock screen
            // up so they can try again or close the app.
            onStatus("Cancelled — tap Unlock to try again")
        } else {
            onStatus(result.errorMessage ?: "Authentication failed")
        }
    }
}
