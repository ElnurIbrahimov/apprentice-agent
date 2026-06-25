package com.aura.ui.screens

import com.aura.IncomingShareStore
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

/**
 * Hilt EntryPoint for getting application-scoped dependencies from a
 * Composable. ChatScreen uses this to access IncomingShareStore without
 * having to inject ChatViewModel with it (which would require a
 * different scope or @Singleton annotation).
 */
@EntryPoint
@InstallIn(SingletonComponent::class)
interface HiltEntryPoint {
    fun incomingShareStore(): IncomingShareStore
}
