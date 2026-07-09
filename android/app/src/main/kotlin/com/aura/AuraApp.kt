package com.aura

import android.app.Application
import androidx.hilt.work.HiltWorkerFactory
import androidx.work.Configuration
import com.aura.proactive.ProactiveBootstrap
import com.aura.providers.ProviderKeys
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject

@HiltAndroidApp
class AuraApp : Application(), Configuration.Provider {

    @Inject lateinit var workerFactory: HiltWorkerFactory
    @Inject lateinit var proactiveBootstrap: ProactiveBootstrap
    @Inject lateinit var providerKeys: ProviderKeys

    override fun onCreate() {
        super.onCreate()
        // Do NOT block the main thread on the DataStore load. The
        // initial key load runs asynchronously in ProviderKeys.init
        // on Dispatchers.IO. UI consumers that need the keys gate on
        // providerKeys.loaded (e.g. ChatViewModel.refreshModels()
        // calls providerKeys.loaded.first { it }). ProactiveBootstrap
        // also reads gates asynchronously in its own IO scope.
        //
        // The previous runBlocking { providerKeys.awaitLoaded() } could
        // ANR on slow storage, corrupt DataStore, or large key files.
        proactiveBootstrap.start()
    }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setWorkerFactory(workerFactory)
            .build()
}