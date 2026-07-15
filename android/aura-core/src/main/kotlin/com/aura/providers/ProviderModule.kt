package com.aura.providers

import com.aura.security.SecureDataStore
import com.aura.data.UserPreferences
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoMap
import dagger.multibindings.StringKey
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object ProviderModule {

    @Provides
    @Singleton
    fun provideModelCatalogCache(store: SecureDataStore): ModelCatalogCache =
        SecureModelCatalogCache(store)

    @Provides
    @Singleton
    fun provideModelCatalogRepository(
        registry: ProviderRegistry,
        cache: ModelCatalogCache,
    ): ModelCatalogRepository = ModelCatalogRepository(
        providerRegistry = registry,
        cache = cache,
    )

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    @Provides
    @IntoMap
    @StringKey("ollama")
    fun provideOllama(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "ollama",
        displayName = "Ollama Cloud",
        baseUrl = "https://ollama.com/v1",
        providerKeys = keys,
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("anthropic")
    fun provideAnthropic(client: OkHttpClient, keys: ProviderKeys): Provider = AnthropicProvider(
        providerKeys = keys,
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("openai")
    fun provideOpenAI(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "openai",
        displayName = "OpenAI",
        baseUrl = "https://api.openai.com/v1",
        providerKeys = keys,
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("deepseek")
    fun provideDeepSeek(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "deepseek",
        displayName = "DeepSeek",
        baseUrl = "https://api.deepseek.com/v1",
        providerKeys = keys,
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("gemini")
    fun provideGemini(client: OkHttpClient, keys: ProviderKeys): Provider = GeminiProvider(
        providerKeys = keys,
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("groq")
    fun provideGroq(client: OkHttpClient, keys: ProviderKeys): Provider = GroqProvider(
        providerKeys = keys,
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("openrouter")
    fun provideOpenRouter(client: OkHttpClient, keys: ProviderKeys): Provider = OpenRouterProvider(
        providerKeys = keys,
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("moa")
    fun provideMoa(
        registry: dagger.Lazy<ProviderRegistry>,
        presetRepo: MoaPresetRepository,
        userPreferences: UserPreferences,
    ): Provider = MoaProvider(
        registry = registry,
        presets = presetRepo.loadPresets(),
        userPreferences = userPreferences,
    )

    // -------- New chat LLM providers (all OpenAI-compat) --------

    @Provides
    @IntoMap
    @StringKey("mistral")
    fun provideMistral(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "mistral", displayName = "Mistral",
        baseUrl = "https://api.mistral.ai/v1", providerKeys = keys, httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("xai")
    fun provideXai(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "xai", displayName = "xAI Grok",
        baseUrl = "https://api.x.ai/v1", providerKeys = keys, httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("together")
    fun provideTogether(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "together", displayName = "Together AI",
        baseUrl = "https://api.together.xyz/v1", providerKeys = keys, httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("cerebras")
    fun provideCerebras(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "cerebras", displayName = "Cerebras",
        baseUrl = "https://api.cerebras.ai/v1", providerKeys = keys, httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("nvidia")
    fun provideNvidia(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "nvidia", displayName = "NVIDIA NIM",
        baseUrl = "https://integrate.api.nvidia.com/v1", providerKeys = keys, httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("llama")
    fun provideLlama(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "llama", displayName = "Meta Llama",
        baseUrl = "https://api.llama.com/compat/v1", providerKeys = keys, httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("agnes")
    fun provideAgnes(client: OkHttpClient, keys: ProviderKeys): Provider = OllamaCloudProvider(
        prefix = "agnes", displayName = "Agnes AI",
        baseUrl = "https://apihub.agnes-ai.com/v1", providerKeys = keys, httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("chatgpt")
    fun provideChatGptSubscription(client: OkHttpClient, keys: ProviderKeys): Provider =
        ChatGptSubscriptionProvider(providerKeys = keys, httpClient = client)

    @Provides
    @IntoMap
    @StringKey("custom")
    fun provideCustomEndpoint(
        client: OkHttpClient,
        customState: CustomEndpointState,
    ): Provider = CustomOpenAiCompatProvider(state = customState, httpClient = client)
}
