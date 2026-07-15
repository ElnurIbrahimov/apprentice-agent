package com.aura.creative

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CreativeProjectStoreTest {
    private val dao = mockk<CreativeProjectDao>(relaxed = true)
    private val store = CreativeProjectStore(dao)

    @Test
    fun `create persists a project with a valid empty world bible`() = runTest {
        val captured = slot<CreativeProjectEntity>()
        coEvery { dao.upsert(capture(captured)) } returns Unit

        val project = store.create(
            name = "The Glass City",
            description = "A city that remembers every resident",
            genre = "speculative fiction",
            tone = "luminous and unsettling",
            templateId = "novel",
        )

        assertEquals("The Glass City", project.name)
        assertEquals("speculative fiction", captured.captured.genre)
        assertEquals("novel", captured.captured.templateId)
        assertEquals(WorldBible(), store.decodeWorld(captured.captured.worldJson))
    }

    @Test
    fun `updateWorld preserves project identity and stores structured components`() = runTest {
        val original = CreativeProjectEntity(
            id = "project-1",
            name = "World",
            worldJson = store.encodeWorld(WorldBible()),
            createdAt = 10L,
            updatedAt = 10L,
        )
        coEvery { dao.getById("project-1") } returns original
        val captured = slot<CreativeProjectEntity>()
        coEvery { dao.upsert(capture(captured)) } returns Unit
        val world = WorldBible(
            characters = listOf(WorldCharacter(id = "c1", name = "Mara", role = "cartographer")),
            rules = listOf(WorldRule(id = "r1", name = "Maps rewrite memory", description = "Every map changes a remembered route")),
        )

        val updated = store.updateWorld("project-1", world)

        assertEquals(10L, captured.captured.createdAt)
        assertTrue(captured.captured.updatedAt >= 10L)
        assertEquals("Mara", updated!!.world.characters.single().name)
        assertEquals("Maps rewrite memory", store.decodeWorld(captured.captured.worldJson).rules.single().name)
    }

    @Test
    fun `observeAll decodes worlds and keeps dao ordering`() = runTest {
        every { dao.observeAll() } returns flowOf(
            listOf(
                CreativeProjectEntity("new", "Newest", worldJson = store.encodeWorld(WorldBible(overview = "new")), updatedAt = 2L),
                CreativeProjectEntity("old", "Older", worldJson = store.encodeWorld(WorldBible(overview = "old")), updatedAt = 1L),
            ),
        )

        val projects = store.observeAll().first()

        assertEquals(listOf("new", "old"), projects.map { it.id })
        assertEquals("new", projects.first().world.overview)
    }

    @Test
    fun `delete removes the project`() = runTest {
        store.delete("project-1")
        coVerify { dao.delete("project-1") }
    }
}