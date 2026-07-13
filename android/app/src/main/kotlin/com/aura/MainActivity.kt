package com.aura

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.core.content.IntentCompat
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.lifecycleScope
import com.aura.data.UserPreferences
import com.aura.security.BiometricActivityHolder
import com.aura.tools.BiometricAuthHandler
import com.aura.ui.components.AuraAppLockContent
import com.aura.ui.components.AuraStartupState
import com.aura.ui.nav.NavGraph
import com.aura.ui.screens.OnboardingScreen
import com.aura.ui.theme.AuraTheme
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.AndroidEntryPoint
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

@EntryPoint
@InstallIn(SingletonComponent::class)
interface FirstRunGateEntryPoint {
    fun firstRunGate(): FirstRunGate
}

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

    private var openChatOnLaunch: Boolean = false
    private var openMemoryOnLaunch: Boolean = false
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.auto(Color.Transparent.value.toInt(), Color.Transparent.value.toInt()),
            navigationBarStyle = SystemBarStyle.auto(Color.Transparent.value.toInt(), Color.Transparent.value.toInt()),
        )
        super.onCreate(savedInstanceState)
        biometricHolder.activity = this
        handleSharedText(intent)
        handleDeepLink(intent)
        setContent {
            AuraRoot()
        }
    }

    override fun onDestroy() {
        biometricHolder.activity = null
        super.onDestroy()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleSharedText(intent)
        handleDeepLink(intent)
    }

    private fun handleSharedText(intent: Intent?) {
        val sharedText = intent?.getStringExtra(ShareReceiverActivity.EXTRA_SHARED_TEXT)
        if (!sharedText.isNullOrBlank()) {
            incomingShareStore.set(sharedText)
        }
        val sharedImageUri = intent?.let { IntentCompat.getParcelableExtra(it, ShareReceiverActivity.EXTRA_SHARED_IMAGE_URI, Uri::class.java) }
        if (sharedImageUri != null) {
            incomingShareStore.setImageUri(sharedImageUri)
        }
    }

    private fun handleDeepLink(intent: Intent) {
        openChatOnLaunch = intent.getBooleanExtra("openChat", false)
        openMemoryOnLaunch = intent.getBooleanExtra("openMemory", false)
    }

    fun consumeOpenChatFlag(): Boolean = openChatOnLaunch.also { openChatOnLaunch = false }
    fun consumeOpenMemoryFlag(): Boolean = openMemoryOnLaunch.also { openMemoryOnLaunch = false }
}

@Composable
fun AuraRoot() {
    val ctx = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainActivity = ctx as? MainActivity
    var firstRunComplete by remember { mutableStateOf<Boolean?>(null) }
    var unlocked by remember { mutableStateOf(false) }

    val mainEntry = remember {
        EntryPointAccessors.fromApplication(
            ctx.applicationContext,
            MainActivityEntryPoint::class.java,
        )
    }

    // Collect appLockEnabled as a reactive Flow so enabling it in
    // Settings takes effect immediately in the active session —
    // the user doesn't need to restart the app for the gate to engage.
    val appLockEnabled by mainEntry.userPreferences().appLockEnabled
        .collectAsState(initial = false)

    // Collect themeMode the same way so Settings theme changes
    // apply live without a restart.
    val themeMode by mainEntry.userPreferences().themeMode
        .collectAsState(initial = "system")

    LaunchedEffect(Unit) {
        val entry = EntryPointAccessors.fromApplication(
            ctx.applicationContext,
            FirstRunGateEntryPoint::class.java,
        )
        firstRunComplete = entry.firstRunGate().isFirstRunComplete()
        if (!appLockEnabled) unlocked = true
    }

    // When appLockEnabled transitions to true, lock immediately.
    LaunchedEffect(appLockEnabled) {
        if (appLockEnabled) {
            unlocked = false
        } else {
            unlocked = true
        }
    }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME && appLockEnabled) {
                unlocked = false
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    AuraTheme(themeMode = themeMode) {
        Surface(
            modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
            color = MaterialTheme.colorScheme.background,
        ) {
            when {
                firstRunComplete == null -> AuraStartupState()

                firstRunComplete == false -> OnboardingScreen(
                    onComplete = { firstRunComplete = true },
                )

                appLockEnabled && !unlocked -> AppLockScreen(
                    onUnlocked = { unlocked = true },
                )

                else -> NavGraph(
                    openChatOnLaunch = mainActivity?.consumeOpenChatFlag() ?: false,
                    openMemoryOnLaunch = mainActivity?.consumeOpenMemoryFlag() ?: false,
                )
            }
        }
    }
}

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

    AuraAppLockContent(
        statusMessage = statusMessage,
        onUnlock = {
            promptForUnlock(ctx, mainEntry, lifecycleOwner, onUnlocked) { message ->
                statusMessage = message
            }
        },
    )
}

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

    val mgr = BiometricManager.from(ctx)
    val canAuth = mgr.canAuthenticate(
        BiometricManager.Authenticators.BIOMETRIC_STRONG or
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
                BiometricManager.Authenticators.BIOMETRIC_STRONG or
                    BiometricManager.Authenticators.DEVICE_CREDENTIAL
            )
            .build()
        prompt.authenticate(promptInfo)
    }

    lifecycleOwner.lifecycleScope.launch {
        val result = handler.result.await()
        executor.shutdownNow()
        if (result.success) {
            onUnlocked()
        } else if (result.errorCode == BiometricPrompt.ERROR_USER_CANCELED ||
            result.errorCode == BiometricPrompt.ERROR_NEGATIVE_BUTTON
        ) {
            onStatus("Cancelled — tap Unlock to try again")
        } else {
            onStatus(result.errorMessage ?: "Authentication failed")
        }
    }
}
