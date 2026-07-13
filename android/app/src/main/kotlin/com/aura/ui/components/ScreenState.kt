package com.aura.ui.components

sealed interface ScreenState<out T> {
    data object Loading : ScreenState<Nothing>
    data class Empty(
        val title: kotlin.String,
        val message: kotlin.String,
    ) : ScreenState<Nothing>
    data class Content<T>(val value: T) : ScreenState<T>
    data class Error(
        val title: kotlin.String,
        val message: kotlin.String,
        val retryable: Boolean = true,
    ) : ScreenState<Nothing>
}

inline fun <T, R> ScreenState<T>.mapContent(transform: (T) -> R): ScreenState<R> = when (this) {
    ScreenState.Loading -> ScreenState.Loading
    is ScreenState.Empty -> this
    is ScreenState.Error -> this
    is ScreenState.Content -> ScreenState.Content(transform(value))
}
