package com.aura.hands

import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolResult
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** CRUD, deterministic input resolution, execution, and run-history ownership. */
@Singleton
class HandRepository @Inject constructor(
    private val dao: HandDao,
) {
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun getAll(): List<Hand> = dao.getAll()
    suspend fun getByName(name: String): Hand? = dao.getByName(name)
    suspend fun getById(id: String): Hand? = dao.getById(id)
    suspend fun getEnabled(): List<Hand> = dao.getEnabled()
    fun observeRecentRuns(limit: Int = 100): Flow<List<HandRun>> = dao.observeRecentRuns(limit)
    fun observeRunsForHand(handId: String, limit: Int = 50): Flow<List<HandRun>> =
        dao.observeRunsForHand(handId, limit)

    suspend fun insert(hand: Hand) = dao.insert(hand)
    suspend fun update(hand: Hand) = dao.update(hand)
    suspend fun deleteByName(name: String) = dao.deleteByName(name)
    suspend fun deleteById(id: String) = dao.deleteById(id)
    suspend fun deleteRunHistory() = dao.deleteRunHistory()

    /** Execute a hand and persist one terminal history row for every attempt. */
    suspend fun run(
        hand: Hand,
        executor: ToolExecutor,
        ctx: ToolContext,
        variables: Map<String, String> = emptyMap(),
        trigger: String = HandRunTrigger.MANUAL.value,
    ): ToolResult {
        val startedAt = System.currentTimeMillis()
        var runRecord = HandRun(
            id = UUID.randomUUID().toString(),
            handId = hand.id,
            handName = hand.name,
            trigger = trigger,
            startedAt = startedAt,
            variablesJson = redactedVariablesJson(variables),
        )
        runCatching { dao.insertRun(runRecord) }

        suspend fun finish(
            result: ToolResult,
            status: HandRunStatus,
            output: String,
            failedStep: Int? = null,
        ): ToolResult {
            runCatching {
                dao.updateRun(
                    runRecord.copy(
                        status = status.value,
                        finishedAt = System.currentTimeMillis(),
                        output = output.take(MAX_HISTORY_OUTPUT_CHARS),
                        failedStep = failedStep,
                    ),
                )
            }
            return result
        }

        suspend fun invalidConfiguration(section: String, error: Exception): ToolResult {
            val detail = error.message?.take(160).orEmpty()
            val message = buildString {
                append("Hand '")
                append(hand.name)
                append("' has invalid ")
                append(section)
                append(" configuration")
                if (detail.isNotBlank()) append(": $detail")
            }
            return finish(
                ToolResult.Error(message, "invalid_hand_configuration"),
                HandRunStatus.FAILED,
                message,
            )
        }

        if (!hand.enabled) {
            val message = "Hand '${hand.name}' is disabled"
            return finish(
                ToolResult.Error(message, "hand_disabled"),
                HandRunStatus.SKIPPED,
                message,
            )
        }

        val defaultVariables = try {
            decodeVariables(hand.variables)
        } catch (error: Exception) {
            return invalidConfiguration("variables", error)
        }
        val resolvedVariables = defaultVariables + variables
        runRecord = runRecord.copy(variablesJson = redactedVariablesJson(resolvedVariables))
        runCatching { dao.updateRun(runRecord) }

        val conditions = try {
            decodeConditions(hand.conditions)
        } catch (error: Exception) {
            return invalidConfiguration("conditions", error)
        }
        val failedCondition = conditions.firstOrNull { !it.matches(resolvedVariables) }
        if (failedCondition != null) {
            val message = "Skipped hand '${hand.name}': ${failedCondition.failureDescription()}"
            return finish(ToolResult.Ok(message), HandRunStatus.SKIPPED, message)
        }

        val steps = try {
            decodeSteps(hand.steps)
        } catch (error: Exception) {
            return invalidConfiguration("steps", error)
        }
        if (steps.isEmpty()) {
            val message = "No steps defined for hand '${hand.name}'"
            return finish(ToolResult.Ok(message), HandRunStatus.SUCCESS, message)
        }

        val outputs = mutableListOf<String>()
        for ((index, step) in steps.withIndex()) {
            val substitution = substitute(step.args, resolvedVariables)
            if (substitution.missingVariables.isNotEmpty()) {
                val message = "Step ${index + 1} (${step.tool}) is missing variables: " +
                    substitution.missingVariables.sorted().joinToString()
                return finish(
                    ToolResult.Error(message, "missing_hand_variable"),
                    HandRunStatus.FAILED,
                    message,
                    index + 1,
                )
            }

            val args = substitution.args.toJsonString()
            val result = try {
                executor.execute(step.tool, args, ctx)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                val message = "Step ${index + 1} (${step.tool}) crashed: " +
                    (error.message ?: error::class.java.simpleName)
                return finish(
                    ToolResult.Error(message, "hand_step_exception"),
                    HandRunStatus.FAILED,
                    message,
                    index + 1,
                )
            }
            when (result) {
                is ToolResult.Ok -> outputs += "Step ${index + 1} (${step.tool}): ${result.output}"
                is ToolResult.Error -> {
                    val message = "Step ${index + 1} (${step.tool}) failed: ${result.message}"
                    return finish(
                        ToolResult.Error(message, result.code),
                        HandRunStatus.FAILED,
                        message,
                        index + 1,
                    )
                }
                is ToolResult.NeedsPermission -> return finish(
                    result,
                    HandRunStatus.NEEDS_PERMISSION,
                    result.rationale,
                    index + 1,
                )
                is ToolResult.NeedsApproval -> return finish(
                    result,
                    HandRunStatus.NEEDS_APPROVAL,
                    result.rationale,
                    index + 1,
                )
            }
        }

        val output = "Hand '${hand.name}' completed.\n${outputs.joinToString("\n")}"
        return finish(ToolResult.Ok(output), HandRunStatus.SUCCESS, output)
    }

    /** Parse the editor/runtime step format, including the legacy stringified args shape. */
    fun parseSteps(stepsJson: String): List<HandStep> =
        runCatching { decodeSteps(stepsJson) }.getOrDefault(emptyList())

    fun stepsToJson(steps: List<HandStep>): String = JsonArray(steps.map { it.toJsonObject() }).toString()

    fun parseVariables(raw: String): Map<String, String> =
        runCatching { decodeVariables(raw) }.getOrDefault(emptyMap())

    fun variablesToJson(variables: Map<String, String>): String =
        JsonObject(variables.mapValues { JsonPrimitive(it.value) }).toString()

    fun parseConditions(raw: String): List<HandCondition> =
        runCatching { decodeConditions(raw) }.getOrDefault(emptyList())

    fun conditionsToJson(conditions: List<HandCondition>): String = json.encodeToString(conditions)

    private fun decodeSteps(stepsJson: String): List<HandStep> {
        val array = json.parseToJsonElement(stepsJson).jsonArray
        return array.mapNotNull { element ->
            val obj = runCatching { element.jsonObject }.getOrNull() ?: return@mapNotNull null
            val tool = obj["tool"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
            val rawArgs = obj["args"]
            val args = when (rawArgs) {
                is JsonObject -> rawArgs.mapValues { (_, value) ->
                    value.jsonPrimitive.contentOrNull ?: value.toString()
                }
                is JsonPrimitive -> {
                    val parsed = json.parseToJsonElement(rawArgs.content).jsonObject
                    parsed.mapValues { (_, value) ->
                        value.jsonPrimitive.contentOrNull ?: value.toString()
                    }
                }
                null -> emptyMap()
                else -> throw IllegalArgumentException("step args must be an object")
            }
            HandStep(tool = tool, args = args)
        }
    }

    private fun decodeVariables(raw: String): Map<String, String> =
        json.parseToJsonElement(raw).jsonObject.mapValues { (_, value) ->
            value.jsonPrimitive.contentOrNull ?: value.toString()
        }

    private fun decodeConditions(raw: String): List<HandCondition> =
        json.decodeFromString(raw)

    private data class Substitution(
        val args: Map<String, String>,
        val missingVariables: Set<String>,
    )

    private fun substitute(args: Map<String, String>, variables: Map<String, String>): Substitution {
        val missing = linkedSetOf<String>()
        val resolved = args.mapValues { (_, value) ->
            TEMPLATE_PATTERN.replace(value) { match ->
                val name = match.groupValues[1]
                variables[name] ?: run {
                    missing += name
                    match.value
                }
            }
        }
        return Substitution(resolved, missing)
    }

    private fun redactedVariablesJson(variables: Map<String, String>): String = buildJsonObject {
        variables.forEach { (name, value) ->
            val stored = if (SECRET_NAME_PATTERN.containsMatchIn(name)) "[redacted]" else value
            put(name, JsonPrimitive(stored))
        }
    }.toString()

    companion object {
        private val TEMPLATE_PATTERN = Regex("""\{\{\s*([A-Za-z][A-Za-z0-9_.-]*)\s*\}\}""")
        private val SECRET_NAME_PATTERN = Regex("token|secret|password|api.?key", RegexOption.IGNORE_CASE)
        private const val MAX_HISTORY_OUTPUT_CHARS = 8_000
    }
}
