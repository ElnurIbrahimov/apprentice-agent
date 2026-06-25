package com.aura.providers

import kotlinx.serialization.Serializable

@Serializable
data class ChatOptions(
    val temperature: Double = 0.7,
    val topP: Double = 1.0,
    val maxTokens: Int? = null,
    val stop: List<String> = emptyList(),
    val seed: Int? = null,
    val responseFormat: ResponseFormat = ResponseFormat.TEXT,
)

@Serializable
enum class ResponseFormat { TEXT, JSON }
