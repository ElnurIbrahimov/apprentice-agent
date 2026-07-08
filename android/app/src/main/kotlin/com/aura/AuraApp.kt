package com.aura

import android.app.Application
import androidx.hilt.work.HiltWorkerFactory
import androidx.work.Configuration
import com.aura.proactive.ProactiveBootstrap
import com.aura.providers.ProviderKeys
import dagger.hilt.android.HiltAndroidApp
import kotlinx.coroutines.runBlocking
import javax.inject.Inject

@HiltAndroidApp
class AuraApp : Application(), Configuration.Provider {

    @Inject lateinit var workerFactory: HiltWorkerFactory
    @Inject lateinit var proactiveBootstrap: ProactiveBootstrap
    @Inject lateinit var providerKeys: ProviderKeys

    override fun onCreate() {
        super.onCreate()
        // Block on the initial DataStore load so the first chat
        // request sees a populated key cache. Without this the
        // Settings screen flickers "0 providers configured" on
        // first show, and the user's first chat request can hit
        // a 401 because the key hasn't loaded yet. DataStore is
        // fast on warm start (5-20ms) and bounded on cold start
        // (~50-100ms) — acceptable on app launch.
        runBlocking { providerKeys.awaitLoaded() }
        proactiveBootstrap.start()
    }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setWorkerFactory(workerFactory)
            .build()
}
