package com.aura.capabilities.di

import com.aura.capabilities.CapabilityProvider
import com.aura.capabilities.elevenlabs.ElevenLabsTtsProvider
import com.aura.capabilities.exa.ExaSearchProvider
import com.aura.capabilities.jina.JinaReaderProvider
import com.aura.capabilities.kling.KlingVideoProvider
import com.aura.capabilities.stability.StabilityImageProvider
import com.aura.capabilities.worldlabs.WorldLabs3DProvider
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoMap
import dagger.multibindings.StringKey

/**
 * Hilt multibindings for non-chat [CapabilityProvider]s. Adding a new
 * capability backend means (1) implementing the [CapabilityProvider] subinterface
 * for its [com.aura.capabilities.CapabilityKind], (2) adding a `@Binds` entry
 * here, (3) adding the prefix to [com.aura.providers.ProviderKeys.PREFIXES].
 */
@Module
@InstallIn(SingletonComponent::class)
abstract class CapabilityModule {

    @Binds
    @IntoMap
    @StringKey("exa")
    abstract fun bindExa(impl: ExaSearchProvider): CapabilityProvider

    @Binds
    @IntoMap
    @StringKey("jina")
    abstract fun bindJina(impl: JinaReaderProvider): CapabilityProvider

    @Binds
    @IntoMap
    @StringKey("elevenlabs")
    abstract fun bindElevenLabs(impl: ElevenLabsTtsProvider): CapabilityProvider

    @Binds
    @IntoMap
    @StringKey("stability")
    abstract fun bindStability(impl: StabilityImageProvider): CapabilityProvider

    @Binds
    @IntoMap
    @StringKey("kling")
    abstract fun bindKling(impl: KlingVideoProvider): CapabilityProvider

    @Binds
    @IntoMap
    @StringKey("worldlabs")
    abstract fun bindWorldLabs(impl: WorldLabs3DProvider): CapabilityProvider
}
