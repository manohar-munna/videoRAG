cmake_minimum_required(VERSION 3.22.1)

project("videorag_native" C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# llama.cpp
#
# Built via its own CMake rather than a hand-listed source set. The previous
# revision of this file enumerated 14 source files from a vendored May-2024
# snapshot, which froze the engine at a build that predates Qwen2-VL: it had no
# LLM_ARCH_QWEN2VL, no M-RoPE, and no qwen2vl_merger projector, so every
# llama_load_model_from_file() on our GGUF returned nullptr and the app silently
# fell back to string templates. add_subdirectory() lets upstream own its own
# build graph so this cannot drift again.
# ---------------------------------------------------------------------------

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)  # we want no CLI executables
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)  # only the CLI tools need it
# Upstream's hook for embedders: adds tools/mtmd alone, without the tools/ tree.
# Fires only while LLAMA_BUILD_TOOLS is OFF, so it must not be combined with it.
set(LLAMA_BUILD_MTMD     ON  CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)  # no usable libomp in the NDK
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)  # statically absorb into our .so

# ggml emits no -march when cross-compiling unless this is set, which leaves
# every ARM feature probe running against a bare armv8-a baseline. Measured on a
# Snapdragon 8 Gen 2: without it, HAVE_DOTPROD / HAVE_MATMUL_INT8 /
# HAVE_FP16_VECTOR_ARITHMETIC all fail even on clang 18.
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "armv8.2-a+fp16+dotprod+i8mm" CACHE STRING "" FORCE)
elseif(${ANDROID_ABI} STREQUAL "x86_64")
    # x86_64 is the emulator target. Without these ggml builds a scalar kernel, and
    # cross-compiling means GGML_NATIVE cannot probe the host: measured 24 s to encode
    # a keyframe on an emulator versus 13 s on a physical arm64 phone, i.e. the
    # emulator was slower than the device it stands in for. Every x86_64 host that can
    # run the Android emulator under WHPX/KVM supports AVX2.
    set(GGML_AVX  ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA  ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Our JNI bridge
# ---------------------------------------------------------------------------

add_library(llama_jni SHARED native-lib.cpp)

target_compile_options(llama_jni PRIVATE -O3 -DNDEBUG)

target_include_directories(llama_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)
find_library(android-lib android)

target_link_libraries(llama_jni
    mtmd            # multimodal: clip + chat-template-aware tokenisation
    llama
    ggml
    ${log-lib}
    ${jnigraphics-lib}
    ${android-lib}
)
