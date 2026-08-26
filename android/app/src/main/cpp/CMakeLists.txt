cmake_minimum_required(VERSION 3.22.1)

project("videorag_native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GGML_VULKAN ON)
add_compile_definitions(GGML_USE_VULKAN)

add_compile_options(
    -O3
    -ffast-math
    -pthread
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE
)

if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    add_compile_options(
        -march=armv8-a+fp16+dotprod
    )
endif()

add_library(
    llama_jni
    SHARED
    native-lib.cpp
)

find_library(
    log-lib
    log
)

find_library(
    jnigraphics-lib
    jnigraphics
)

target_link_libraries(
    llama_jni
    ${log-lib}
    ${jnigraphics-lib}
)
