# Aura proguard rules
-keep class com.aura.** { *; }
-keepclassmembers class * {
    @kotlinx.serialization.Serializable <fields>;
}
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.SerializationKt
-keep,includedescriptorclasses class com.aura.**$$serializer { *; }
-keepclassmembers class com.aura.** {
    *** Companion;
}
-keepclasseswithmembers class com.aura.** {
    kotlinx.serialization.KSerializer serializer(...);
}
-keep class kotlinx.coroutines.** { *; }
-dontwarn org.bouncycastle.**
-dontwarn org.conscrypt.**
-dontwarn org.openjsse.**
# Android desugaring rewrites Java 9 string-concatenation call sites; R8 must
# not require the JVM-only bootstrap implementation in the final APK.
-dontwarn java.lang.invoke.StringConcatFactory
# Annotation-processing APIs arrive through transitive compile-time tooling;
# they are not Android runtime dependencies.
-dontwarn javax.lang.model.**
