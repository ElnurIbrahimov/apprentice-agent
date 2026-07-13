package com.aura.hands

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import kotlinx.serialization.Serializable

@Serializable
data class HandCondition(
    val variable: String,
    val operator: String,
    val value: String = "",
) {
    fun matches(variables: Map<String, String>): Boolean {
        val actual = variables[variable]
        return when (operator) {
            "equals" -> actual?.equals(value, ignoreCase = true) == true
            "not_equals" -> actual?.equals(value, ignoreCase = true) != true
            "contains" -> actual?.contains(value, ignoreCase = true) == true
            "not_contains" -> actual?.contains(value, ignoreCase = true) != true
            "greater_than" -> compareNumbers(actual, value) { left, right -> left > right }
            "less_than" -> compareNumbers(actual, value) { left, right -> left < right }
            "is_set", "not_empty" -> !actual.isNullOrBlank()
            "is_empty", "empty" -> actual.isNullOrEmpty()
            else -> false
        }
    }

    private fun compareNumbers(
        actual: String?,
        expected: String,
        predicate: (Double, Double) -> Boolean,
    ): Boolean {
        val left = actual?.toDoubleOrNull() ?: return false
        val right = expected.toDoubleOrNull() ?: return false
        return predicate(left, right)
    }

    fun failureDescription(): String = when (operator) {
        "equals" -> "$variable must equal '$value'"
        "not_equals" -> "$variable must not equal '$value'"
        "contains" -> "$variable must contain '$value'"
        "not_contains" -> "$variable must not contain '$value'"
        "greater_than" -> "$variable must be greater than '$value'"
        "less_than" -> "$variable must be less than '$value'"
        "is_set", "not_empty" -> "$variable must be set"
        "is_empty", "empty" -> "$variable must be empty"
        else -> "$variable uses unsupported condition '$operator'"
    }
}

enum class HandScheduleType(val value: String) {
    NONE("none"),
    DAILY("daily"),
    WEEKDAYS("weekdays"),
    WEEKLY("weekly");

    companion object {
        fun from(value: String): HandScheduleType = entries.firstOrNull { it.value == value } ?: NONE
    }
}

enum class HandRunTrigger(val value: String) {
    MANUAL("manual"),
    AGENT("agent"),
    PHRASE("phrase"),
    SCHEDULE("schedule"),
}

enum class HandRunStatus(val value: String) {
    RUNNING("running"),
    SUCCESS("success"),
    FAILED("failed"),
    SKIPPED("skipped"),
    NEEDS_PERMISSION("needs_permission"),
    NEEDS_APPROVAL("needs_approval"),
}

@Entity(
    tableName = "hand_runs",
    indices = [
        Index(value = ["handId"]),
        Index(value = ["startedAt"]),
    ],
)
data class HandRun(
    @PrimaryKey val id: String,
    val handId: String,
    /** Snapshot survives a later hand rename or deletion. */
    val handName: String,
    val trigger: String,
    val status: String = HandRunStatus.RUNNING.value,
    val startedAt: Long = System.currentTimeMillis(),
    val finishedAt: Long? = null,
    val output: String = "",
    val failedStep: Int? = null,
    /** Resolved inputs with obvious secret-shaped values redacted. */
    val variablesJson: String = "{}",
)
