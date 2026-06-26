package com.aura.hands

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(entities = [Hand::class], version = 1, exportSchema = false)
abstract class HandDatabase : RoomDatabase() {
    abstract fun handDao(): HandDao
}
