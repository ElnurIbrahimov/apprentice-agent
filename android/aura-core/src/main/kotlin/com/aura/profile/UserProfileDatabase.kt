package com.aura.profile

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(entities = [UserProfileEntity::class], version = 1, exportSchema = false)
abstract class UserProfileDatabase : RoomDatabase() {
    abstract fun userProfileDao(): UserProfileDao
}
