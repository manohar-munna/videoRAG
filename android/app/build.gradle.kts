plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.cctv.videorag"
    compileSdk = 34
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "com.cctv.videorag"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -O3"
                abiFilters += listOf("arm64-v8a", "x86_64")
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        debug {
            // Build the NATIVE code optimised even in debug APKs.
            //
            // AGP passes -DCMAKE_BUILD_TYPE=Debug for the debug variant, which compiles
            // ggml/llama/mtmd at -O0. Inference is almost entirely native math, so this
            // is not a small penalty: encoding one 299-token keyframe took >6 min in the
            // debug APK versus 39 s for the same frame, same model, same phone via a
            // Release-built binary. Kotlin stays debuggable; only the C/C++ is optimised.
            externalNativeBuild {
                cmake {
                    arguments += listOf("-DCMAKE_BUILD_TYPE=Release")
                }
            }
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // Android UI & Core Lifecycle
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // CameraX Ingestion
    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // ONNX Runtime Android (NPU Acceleration via NNAPI)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")

    // Local Storage (SQLite for metadata)
    implementation("androidx.sqlite:sqlite-ktx:2.4.0")

    // Coroutines for non-blocking UI
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
