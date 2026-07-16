package com.aura.pipeline

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ProductionPipelineTest {

    @Test
    fun all_contains_six_pipelines() {
        assertEquals(6, ProductionPipeline.all.size)
    }

    @Test
    fun byId_returns_correct_pipeline() {
        val novel = ProductionPipeline.byId("novel")
        assertNotNull(novel)
        assertEquals("Novel", novel!!.name)
    }

    @Test
    fun byId_returns_null_for_unknown() {
        assertNull(ProductionPipeline.byId("nonexistent"))
    }

    @Test
    fun novel_pipeline_has_seven_stages() {
        val novel = ProductionPipeline.novel
        assertEquals(7, novel.stages.size)
    }

    @Test
    fun trailer_pipeline_has_eight_stages() {
        val trailer = ProductionPipeline.trailer
        assertEquals(8, trailer.stages.size)
    }

    @Test
    fun stages_have_valid_dependencies() {
        for (pipeline in ProductionPipeline.all) {
            val stageIds = pipeline.stages.map { it.id }.toSet()
            for (stage in pipeline.stages) {
                for (dep in stage.dependsOn) {
                    assertTrue(
                        "Stage ${stage.id} in ${pipeline.name} depends on $dep which doesn't exist",
                        dep in stageIds,
                    )
                }
            }
        }
    }

    @Test
    fun first_stage_has_no_dependencies() {
        for (pipeline in ProductionPipeline.all) {
            val firstStage = pipeline.stages.first()
            assertTrue(
                "First stage ${firstStage.id} in ${pipeline.name} should have no dependencies",
                firstStage.dependsOn.isEmpty(),
            )
        }
    }

    @Test
    fun last_stage_depends_on_something() {
        for (pipeline in ProductionPipeline.all) {
            val lastStage = pipeline.stages.last()
            assertTrue(
                "Last stage ${lastStage.id} in ${pipeline.name} should have dependencies",
                lastStage.dependsOn.isNotEmpty(),
            )
        }
    }

    @Test
    fun all_stages_have_executor() {
        for (pipeline in ProductionPipeline.all) {
            for (stage in pipeline.stages) {
                assertTrue(
                    "Stage ${stage.id} in ${pipeline.name} has empty executor",
                    stage.executor.isNotBlank(),
                )
            }
        }
    }

    @Test
    fun default_roles_are_populated() {
        for (pipeline in ProductionPipeline.all) {
            assertTrue(
                "Pipeline ${pipeline.name} should have default roles",
                pipeline.defaultRoles.isNotEmpty(),
            )
        }
    }
}