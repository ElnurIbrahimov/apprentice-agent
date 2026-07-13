package com.aura.testing

import com.aura.providers.InMemoryModelCatalogCache
import com.aura.providers.ModelCatalogCache
import com.aura.providers.ModelCatalogRepository
import com.aura.providers.Provider

import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderModule
import dagger.Module
import dagger.Provides
import dagger.hilt.components.SingletonComponent
import dagger.hilt.testing.TestInstallIn
import dagger.multibindings.IntoMap
import dagger.multibindings.StringKey
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import okhttp3.OkHttpClient

object FakeProviderController {
    var models: List<String> = listOf("model-a", "model-b")
    var validKey: String = "test-key-valid"
    var failure: Throwable? = null

    fun reset() {
        models = listOf("model-a", "model-b")
        validKey = "test-key-valid"
        failure = null
    }
}

@Module
@TestInstallIn(
    components = [SingletonComponent::class],
    replaces = [ProviderModule::class],
)
object FakeProviderModule {

    @Provides
    @Singleton
    fun httpClient(): OkHttpClient = OkHttpClient()

    @Provides
    @Singleton
    fun catalogCache(): ModelCatalogCache = InMemoryModelCatalogCache()

    @Provides
    @Singleton
    fun catalogRepository(
        registry: com.aura.providers.ProviderRegistry,
        cache: ModelCatalogCache,
    ): ModelCatalogRepository = ModelCatalogRepository(registry, cache)

    @Provides
    @IntoMap
    @StringKey("ollama")
    fun fakeProvider(keys: ProviderKeys): Provider = object : Provider {
        override val prefix = "ollama"
        override val displayName = "Ollama Cloud"
        override fun isConfigured(): Boolean = !keys.keyFor(prefix).isNullOrBlank()

        override fun chat(
            model: String,
            messages: List<ProviderMessage>,
            options: com.aura.providers.ChatOptions,
            tools: List<com.aura.providers.ToolDefinition>,
        ): Flow<ProviderChunk> = flowOf(ProviderChunk(text = "ok"))

        override suspend fun listModels(): List<String> {
            FakeProviderController.failure?.let { throw it }
            if (keys.keyFor(prefix) != FakeProviderController.validKey) {
                throw IllegalStateException("invalid test key")
            }
            return FakeProviderController.models
        }

        override suspend fun cancel() = Unit
    }
}
