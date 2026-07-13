package com.aura.ui.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.core.error.CrashLogEntry
import com.aura.core.error.CrashLogger
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class DiagnosticsUiState(
    val entries: List<CrashLogEntry> = emptyList(),
    val loading: Boolean = true,
    val exporting: Boolean = false,
    val exportFile: File? = null,
    val error: String? = null,
)

@HiltViewModel
class DiagnosticsViewModel @Inject constructor(
    private val crashLogger: CrashLogger,
    @ApplicationContext private val context: Context,
) : ViewModel() {

    private val _state = MutableStateFlow(DiagnosticsUiState())
    val state: StateFlow<DiagnosticsUiState> = _state.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            runCatching { crashLogger.entries() }
                .onSuccess { entries ->
                    _state.update { it.copy(entries = entries, loading = false, error = null) }
                }
                .onFailure { failure ->
                    _state.update {
                        it.copy(
                            loading = false,
                            error = failure.message ?: failure.javaClass.simpleName,
                        )
                    }
                }
        }
    }

    fun clearAll() {
        viewModelScope.launch {
            runCatching {
                crashLogger.clear()
                crashLogger.entries()
            }.onSuccess { entries ->
                _state.update { it.copy(entries = entries, error = null) }
            }.onFailure { failure ->
                _state.update { it.copy(error = failure.message ?: failure.javaClass.simpleName) }
            }
        }
    }

    fun prepareExport() {
        if (_state.value.exporting) return
        _state.update { it.copy(exporting = true) }
        viewModelScope.launch {
            runCatching { crashLogger.exportTo(context.cacheDir) }
                .onSuccess { file ->
                    _state.update { it.copy(exporting = false, exportFile = file, error = null) }
                }
                .onFailure { failure ->
                    _state.update {
                        it.copy(
                            exporting = false,
                            error = failure.message ?: failure.javaClass.simpleName,
                        )
                    }
                }
        }
    }

    fun consumeExport() {
        _state.update { it.copy(exportFile = null) }
    }

    fun clearError() {
        _state.update { it.copy(error = null) }
    }
}
