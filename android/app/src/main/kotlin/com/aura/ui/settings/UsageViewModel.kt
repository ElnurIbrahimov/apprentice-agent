package com.aura.ui.settings

import androidx.lifecycle.ViewModel
import com.aura.usage.UsageSnapshot
import com.aura.usage.UsageTracker
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject

@HiltViewModel
class UsageViewModel @Inject constructor(
    private val tracker: UsageTracker,
) : ViewModel() {
    val usage: StateFlow<UsageSnapshot> = tracker.snapshot

    fun reset() = tracker.reset()
}
