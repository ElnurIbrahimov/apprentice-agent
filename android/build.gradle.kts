plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.ksp) apply false
    alias(libs.plugins.hilt) apply false
}

// Global resolution: prevent KSP / Hilt pulling duplicate `org.jetbrains.annotations` classes
// from multiple versions (com.intellij:annotations:12 vs org.jetbrains:annotations:23).
// Force the newer one everywhere and exclude the transitive IntelliJ annotations jar.
allprojects {
    configurations.all {
        resolutionStrategy.eachDependency {
            if (requested.group == "org.jetbrains" && requested.name == "annotations") {
                useVersion("23.0.0")
                because("Force single annotations version to avoid KSP/Hilt duplicate-class errors.")
            }
        }
        // Exclude the older com.intellij:annotations jar that Hilt's KSP processor pulls in
        // transitively. The newer org.jetbrains:annotations covers everything we need.
        exclude(group = "com.intellij", module = "annotations")
    }
}
