package com.aura.testing

object TestData {
    const val PROVIDER_PREFIX: kotlin.String = "test"
    const val VALID_API_KEY: kotlin.String = "test-key-valid"
    const val INVALID_API_KEY: kotlin.String = "test-key-invalid"
    const val PRIMARY_MODEL_ID: kotlin.String = "test:primary-model"
    const val SECONDARY_MODEL_ID: kotlin.String = "test:secondary-model"

    val MODEL_IDS: List<kotlin.String> = listOf(PRIMARY_MODEL_ID, SECONDARY_MODEL_ID)
}
