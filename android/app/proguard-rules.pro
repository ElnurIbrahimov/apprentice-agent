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
