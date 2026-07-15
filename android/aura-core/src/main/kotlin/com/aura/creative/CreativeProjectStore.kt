package com.aura.creative

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class CreativeProjectStore @Inject constructor(
    private val dao: CreativeProjectDao,
) {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    fun observeAll(): Flow<List<CreativeProject>> = dao.observeAll().map { rows -> rows.map(::toDomain) }

    suspend fun create(
        name: String,
        description: String = "",
        genre: String = "",
        tone: String = "",
        templateId: String = "",
    ): CreativeProject {
        require(name.isNotBlank()) { "Project name is required." }
        val now = System.currentTimeMillis()
        val row = CreativeProjectEntity(
            id = UUID.randomUUID().toString(),
            name = name.trim().take(120),
            description = description.trim().take(1_000),
            genre = genre.trim().take(80),
            tone = tone.trim().take(120),
            worldJson = encodeWorld(WorldBible()),
            templateId = templateId,
            createdAt = now,
            updatedAt = now,
        )
        dao.upsert(row)
        return toDomain(row)
    }

    suspend fun get(id: String): CreativeProject? = dao.getById(id)?.let(::toDomain)

    suspend fun updateProject(
        id: String,
        name: String,
        description: String,
        genre: String,
        tone: String,
        templateId: String,
    ): CreativeProject? {
        val current = dao.getById(id) ?: return null
        val updated = current.copy(
            name = name.trim().take(120),
            description = description.trim().take(1_000),
            genre = genre.trim().take(80),
            tone = tone.trim().take(120),
            templateId = templateId,
            updatedAt = System.currentTimeMillis(),
        )
        dao.upsert(updated)
        return toDomain(updated)
    }

    suspend fun updateWorld(id: String, world: WorldBible): CreativeProject? {
        val current = dao.getById(id) ?: return null
        val updated = current.copy(
            worldJson = encodeWorld(world),
            updatedAt = System.currentTimeMillis(),
        )
        dao.upsert(updated)
        return toDomain(updated)
    }

    suspend fun recordSimulation(id: String, simulation: SimulationRecord): CreativeProject? {
        val project = get(id) ?: return null
        return updateWorld(id, project.world.copy(simulations = listOf(simulation) + project.world.simulations))
    }

    suspend fun canonizeSimulation(id: String, simulationId: String): CreativeProject? {
        val project = get(id) ?: return null
        val simulation = project.world.simulations.find { it.id == simulationId } ?: return project
        val world = project.world.copy(
            simulations = project.world.simulations.map {
                if (it.id == simulationId) it.copy(canonized = true) else it
            },
            timeline = listOf(
                WorldEvent(
                    title = simulation.premise.take(120),
                    description = simulation.outcome,
                    participants = listOfNotNull(simulation.perspective.takeIf(String::isNotBlank)),
                ),
            ) + project.world.timeline,
        )
        return updateWorld(id, world)
    }

    suspend fun incrementTurn(id: String) {
        val current = dao.getById(id) ?: return
        dao.upsert(
            current.copy(
                turnCount = current.turnCount + 1,
                lastSessionEnded = System.currentTimeMillis(),
                updatedAt = System.currentTimeMillis(),
            ),
        )
    }

    suspend fun delete(id: String) = dao.delete(id)

    fun encodeWorld(world: WorldBible): String = json.encodeToString(world)

    fun decodeWorld(value: String): WorldBible = runCatching {
        json.decodeFromString<WorldBible>(value)
    }.getOrDefault(WorldBible())

    private fun toDomain(row: CreativeProjectEntity) = CreativeProject(
        id = row.id,
        name = row.name,
        description = row.description,
        genre = row.genre,
        tone = row.tone,
        world = decodeWorld(row.worldJson),
        templateId = row.templateId,
        turnCount = row.turnCount,
        createdAt = row.createdAt,
        updatedAt = row.updatedAt,
    )
}