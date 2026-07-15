package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.creative.CreativeEngine
import com.aura.creative.CreativeMode
import com.aura.creative.CreativeProject
import com.aura.creative.CreativeProjectStore
import com.aura.creative.WorldBible
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class CreativeStudioUiState(
    val projects: List<CreativeProject> = emptyList(),
    val selectedProject: CreativeProject? = null,
    val loading: Boolean = true,
    val generating: Boolean = false,
    val output: String = "",
    val error: String? = null,
    val message: String? = null,
    val createdProjectId: String? = null,
)

@HiltViewModel
class CreativeStudioViewModel @Inject constructor(
    private val store: CreativeProjectStore,
    private val engine: CreativeEngine,
) : ViewModel() {
    private val _state = MutableStateFlow(CreativeStudioUiState())
    val state: StateFlow<CreativeStudioUiState> = _state.asStateFlow()
    private var generationJob: Job? = null

    init {
        viewModelScope.launch {
            store.observeAll().collect { projects ->
                val selectedId = _state.value.selectedProject?.id
                _state.update { current ->
                    current.copy(
                        projects = projects,
                        selectedProject = selectedId?.let { id -> projects.find { it.id == id } }
                            ?: current.selectedProject,
                        loading = false,
                    )
                }
            }
        }
    }

    fun createProject(
        name: String,
        description: String,
        genre: String,
        tone: String,
        templateId: String,
    ) {
        viewModelScope.launch {
            runCatching { store.create(name, description, genre, tone, templateId) }
                .onSuccess { project ->
                    _state.update { it.copy(createdProjectId = project.id, selectedProject = project, error = null) }
                }
                .onFailure { error -> _state.update { it.copy(error = error.message ?: "Could not create project.") } }
        }
    }

    fun consumeCreatedProject() {
        _state.update { it.copy(createdProjectId = null) }
    }

    fun loadProject(id: String) {
        viewModelScope.launch {
            _state.update { it.copy(loading = true, error = null) }
            val project = store.get(id)
            _state.update {
                it.copy(
                    selectedProject = project,
                    loading = false,
                    error = if (project == null) "Creative project not found." else null,
                )
            }
        }
    }

    fun saveMetadata(
        name: String,
        description: String,
        genre: String,
        tone: String,
        templateId: String,
    ) {
        val id = _state.value.selectedProject?.id ?: return
        viewModelScope.launch {
            runCatching { store.updateProject(id, name, description, genre, tone, templateId) }
                .onSuccess { project ->
                    _state.update { it.copy(selectedProject = project, message = "Project details saved.", error = null) }
                }
                .onFailure { error -> _state.update { it.copy(error = error.message ?: "Could not save project.") } }
        }
    }

    fun saveWorld(world: WorldBible, message: String = "World bible saved.") {
        val id = _state.value.selectedProject?.id ?: return
        viewModelScope.launch {
            runCatching { store.updateWorld(id, world) }
                .onSuccess { project -> _state.update { it.copy(selectedProject = project, message = message, error = null) } }
                .onFailure { error -> _state.update { it.copy(error = error.message ?: "Could not save world bible.") } }
        }
    }

    fun generate(mode: CreativeMode, prompt: String, perspective: String = "") {
        val project = _state.value.selectedProject ?: return
        generationJob?.cancel()
        generationJob = viewModelScope.launch {
            _state.update { it.copy(generating = true, output = "", error = null) }
            runCatching {
                engine.generate(project.id, mode, prompt, perspective).collect { chunk ->
                    _state.update { it.copy(output = it.output + chunk) }
                }
            }.onFailure { error ->
                _state.update { it.copy(error = error.message ?: "Creative generation failed.") }
            }
            val refreshed = store.get(project.id)
            _state.update { it.copy(selectedProject = refreshed ?: it.selectedProject, generating = false) }
        }
    }

    fun cancelGeneration() {
        generationJob?.cancel()
        generationJob = null
        _state.update { it.copy(generating = false) }
    }

    fun canonizeSimulation(simulationId: String) {
        val id = _state.value.selectedProject?.id ?: return
        viewModelScope.launch {
            val project = store.canonizeSimulation(id, simulationId)
            _state.update { it.copy(selectedProject = project ?: it.selectedProject, message = "Simulation added to canon timeline.") }
        }
    }

    fun deleteProject(id: String) {
        viewModelScope.launch {
            store.delete(id)
            _state.update { it.copy(selectedProject = if (it.selectedProject?.id == id) null else it.selectedProject) }
        }
    }

    fun clearNotice() {
        _state.update { it.copy(error = null, message = null) }
    }
}