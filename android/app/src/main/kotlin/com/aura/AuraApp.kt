package com.aura

import android.app.Application
import android.os.Handler
import android.os.Looper
import androidx.hilt.work.HiltWorkerFactory
import androidx.work.Configuration
import com.aura.core.error.CrashHandler
import com.aura.core.error.CrashLogger
import com.aura.proactive.ProactiveBootstrap
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject
import javax.inject.Provider

@HiltAndroidApp
class AuraApp : Application(), Configuration.Provider {

    @Inject lateinit var workerFactory: HiltWorkerFactory
    @Inject lateinit var proactiveBootstrap: Provider<ProactiveBootstrap>
    @Inject lateinit var crashLogger: CrashLogger

    override fun onCreate() {
        super.onCreate()
        CrashHandler.install(crashLogger)
        // Do NOT block the main thread on the DataStore load. The
        // initial key load runs asynchronously in ProviderKeys.init
        // on Dispatchers.IO. UI consumers that need the keys gate on
        // providerKeys.loaded (e.g. ChatViewModel.refreshModels()
        // calls providerKeys.loaded.first { it }). ProactiveBootstrap
        // also reads gates asynchronously in its own IO scope.
        //
        // The previous runBlocking { providerKeys.awaitLoaded() } could
        // ANR on slow storage, corrupt DataStore, or large key files.
        // Scheduling and monitoring are maintenance work, not first-frame work.
        Handler(Looper.getMainLooper()).postDelayed({
            proactiveBootstrap.get().start()
        }, 750L)
    }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setWorkerFactory(workerFactory)
            .build()
}