-keep class com.aura.core.** { *; }
# Android desugaring rewrites Java 9 string-concatenation call sites; R8 must
# not require the JVM-only bootstrap implementation in the final APK.
-dontwarn java.lang.invoke.StringConcatFactory
