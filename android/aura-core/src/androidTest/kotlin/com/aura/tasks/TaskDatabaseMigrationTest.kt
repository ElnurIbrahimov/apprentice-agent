package com.aura.tasks

import androidx.room.testing.MigrationTestHelper
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Rule
import org.junit.Test

class TaskDatabaseMigrationTest {

    @get:Rule
    val helper = MigrationTestHelper(
        InstrumentationRegistry.getInstrumentation(),
        TaskDatabase::class.java.canonicalName,
        FrameworkSQLiteOpenHelperFactory(),
    )

    @Test
    fun migrate1To2_preservesTasks_andCreatesReminders() {
        val db = helper.createDatabase("test-tasks.db", 1)
        // Insert a task using the v1 schema (no reminders table exists yet).
        db.execSQL(
            """
            INSERT INTO tasks (id, title, description, createdAt, dueAt, completedAt, status, priority, tags)
            VALUES ('task-1', 'Buy milk', '2% organic', 1700000000, NULL, NULL, 'pending', 1, 'groceries')
            """.trimIndent(),
        )
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-tasks.db",
            2,
            true,
            TasksModule.MIGRATION_1_2,
        )

        // Verify the task survived the migration.
        val cursor = migrated.query("SELECT * FROM tasks WHERE id = 'task-1'")
        cursor.use {
            assert(it.moveToFirst()) { "Task row should survive migration" }
            assert(it.getString(it.getColumnIndexOrThrow("title")) == "Buy milk")
            assert(it.getString(it.getColumnIndexOrThrow("status")) == "pending")
        }

        // Verify the reminders table exists with the correct schema.
        val reminderCursor = migrated.query("SELECT name FROM sqlite_master WHERE type='table' AND name='reminders'")
        reminderCursor.use {
            assert(it.moveToFirst()) { "reminders table should exist after migration" }
        }

        migrated.close()
    }
}