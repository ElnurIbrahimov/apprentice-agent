package com.aura.proactive

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Bridges the proactive [ProactiveEventBus] into a state that the UI can
 * observe. For v1 we just hold the most recent event of each kind; the
 * UI shows a single "incoming" card. A future revision will accumulate a
 * feed.
 *
 * Hilt: Singleton (process-scoped, so the collector outlives any single
 * activity). The collector coroutine lives for the lifetime of the process.
 */
@Singleton
class ProactiveEvents @Inject constructor(
    private val bus: ProactiveEventBus,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val _latest = MutableStateFlow<ProactiveEventBus.Event?>(null)
    val latest: StateFlow<ProactiveEventBus.Event?> = _latest.asStateFlow()

    init {
        scope.launch {
            bus.events.collect { event ->
                _latest.value = event
            }
        }
    }

    fun dismiss() {
        _latest.value = null
    }
}
