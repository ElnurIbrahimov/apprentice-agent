package com.aura.profile

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "user_profile")
data class UserProfileEntity(
    @PrimaryKey val id: Int = 1,
    val name: String? = null,
    val traitsJson: String = "[]",
    val preferencesJson: String = "{}",
    val factsJson: String = "[]",
    val lastUpdated: Long = 0L,
)
