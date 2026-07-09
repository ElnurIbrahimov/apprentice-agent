package com.aura.tasks

import android.content.Context
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.aura.data.RoomConfig
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object TasksModule {

    /**
     * Migration 1→2: adds the `reminders` table for [ReminderEntity].
     * The `tasks` table is unchanged from v1. Existing task rows survive
     * untouched because the columns and constraints are identical.
     */
    val MIGRATION_1_2 = object : Migration(1, 2) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT NOT NULL PRIMARY KEY,
                    message TEXT NOT NULL,
                    triggerAt INTEGER NOT NULL,
                    createdAt INTEGER NOT NULL,
                    taskId TEXT NOT NULL
                )
                """.trimIndent(),
            )
        }
    }

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): TaskDatabase =
        RoomConfig.builder(
            context,
            TaskDatabase::class.java,
            "aura-tasks.db",
            migrations = arrayOf(MIGRATION_1_2),
        ).build()

    @Provides
    fun provideTaskDao(db: TaskDatabase): TaskDao = db.taskDao()

    @Provides
    fun provideReminderDao(db: TaskDatabase): ReminderDao = db.reminderDao()
}
