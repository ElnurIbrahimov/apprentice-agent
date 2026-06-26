package com.aura.tools

import kotlinx.serialization.Serializable

/**
 * A source citation returned by a research/web tool.
 */
@Serializable
data class Citation(
    val index: Int,
    val title: String,
    val url: String,
)
