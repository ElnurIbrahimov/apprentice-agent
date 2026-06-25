package com.aura.providers

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File

/**
 * On-device provider backed by llama.cpp Android binding.
 * v1 stub: the binding ships in v1.5 once we settle on llama.cpp vs MLC-LLM.
 * v1 contract: if a model file is present, attempt to load; otherwise emit a clear error chunk.
 *
 * Source files when binding lands: src/main/cpp/llama_jni.cpp
 */
class LocalProvider(
    private val modelDir: File,
) : Provider {
    override val prefix = "local"
    override val displayName = "On-Device"

    override fun isConfigured(): Boolean = modelDir.listFiles()?.any { it.extension == "gguf" } == true

    override fun chat(model: String, messages: List<ProviderMessage>, options: ChatOptions, tools: List<ToolDefinition>): Flow<ProviderChunk> = flow {
        val modelFile = File(modelDir, "$model.gguf")
        if (!modelFile.exists()) {
            emit(ProviderChunk(error = ProviderError("model_not_found", "No GGUF model at ${modelFile.absolutePath}. Download a model from Settings → Models.", retryable = false)))
            return@flow
        }
        // v1 stub: emit a placeholder. Real llama.cpp JNI integration in v1.5.
        emit(ProviderChunk(error = ProviderError("not_implemented", "Local model inference ships in v1.5 once llama.cpp Android binding is wired. For now use a cloud provider.", retryable = false)))
    }

    override suspend fun listModels(): List<String> = listOf("gemma3-1b-q4", "gemma3-4b-q4", "gemma3-7b-q4")
    override suspend fun cancel() {}
}
