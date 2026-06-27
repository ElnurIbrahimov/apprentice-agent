package com.aura.memory

import android.content.Context
import androidx.room.testing.MigrationTestHelper
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import com.aura.data.RoomConfig
import org.junit.Rule
import org.junit.Test

class MemoryDatabaseMigrationTest {

    @get:Rule
    val helper = MigrationTestHelper(
        InstrumentationRegistry.getInstrumentation(),
        MemoryDatabase::class.java.canonicalName,
        FrameworkSQLiteOpenHelperFactory(),
    )

    @Test
    fun migrate1To2() {
        val db = helper.createDatabase("test-aura-memory.db", 1)
        db.close()

        helper.runMigrationsAndValidate(
            "test-aura-memory.db",
            2,
            true,
            MemoryModule.MIGRATION_1_2,
        )
    }
}
