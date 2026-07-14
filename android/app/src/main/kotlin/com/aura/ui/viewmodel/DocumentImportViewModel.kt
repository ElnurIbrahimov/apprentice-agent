package com.aura.ui.viewmodel

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.documents.DocumentEntity
import com.aura.documents.DocumentRepository
import com.aura.documents.DocumentTextExtractor
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class DocumentImportUiState(
    val documents: List<DocumentEntity> = emptyList(),
    val importing: Boolean = false,
    val stage: String? = null,
    val message: String? = null,
    val error: String? = null,
)

@HiltViewModel
class DocumentImportViewModel @Inject constructor(
    private val extractor: DocumentTextExtractor,
    private val repository: DocumentRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(DocumentImportUiState())
    val state: StateFlow<DocumentImportUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            repository.observeAll().collect { documents ->
                _state.update { it.copy(documents = documents) }
            }
        }
    }

    fun import(uri: Uri) {
        if (_state.value.importing) return
        viewModelScope.launch {
            _state.update {
                it.copy(importing = true, stage = "Reading document…", message = null, error = null)
            }
            runCatching {
                val extracted = extractor.extract(uri)
                _state.update { it.copy(stage = "Creating searchable memory…") }
                repository.import(
                    id = extracted.id,
                    name = extracted.name,
                    mimeType = extracted.mimeType,
                    sourceUri = extracted.sourceUri,
                    text = extracted.text,
                )
            }.onSuccess { result ->
                _state.update {
                    it.copy(
                        importing = false,
                        stage = null,
                        message = "Imported ${result.document.name} · ${result.chunkCount} searchable chunks",
                    )
                }
            }.onFailure { failure ->
                _state.update {
                    it.copy(
                        importing = false,
                        stage = null,
                        error = failure.message ?: "Document import failed.",
                    )
                }
            }
        }
    }

    fun delete(id: String) {
        viewModelScope.launch {
            runCatching { repository.delete(id) }
                .onSuccess {
                    _state.update { it.copy(message = "Document and its memory chunks deleted.", error = null) }
                }
                .onFailure { failure ->
                    _state.update { it.copy(error = failure.message ?: "Could not delete the document.") }
                }
        }
    }

    fun clearNotice() {
        _state.update { it.copy(message = null, error = null) }
    }
}