cmake_minimum_required(VERSION 3.22.1)

project("videorag_native" C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
    -O3
    -ffast-math
    -pthread
    -DNDEBUG
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE
    -DGGML_USE_K_QUANTS
)

if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    add_compile_options(
        -march=armv8-a+fp16+dotprod
    )
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/llama_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/llama_engine/common
    ${CMAKE_CURRENT_SOURCE_DIR}/llama_engine/examples/llava
)

set(LLAMA_SRCS
    llama_engine/ggml.c
    llama_engine/ggml-alloc.c
    llama_engine/ggml-backend.c
    llama_engine/ggml-quants.c
    llama_engine/llama.cpp
    llama_engine/unicode.cpp
    llama_engine/unicode-data.cpp
    llama_engine/sgemm.cpp
    llama_engine/common/sampling.cpp
    llama_engine/common/common.cpp
    llama_engine/common/grammar-parser.cpp
    llama_engine/common/json-schema-to-grammar.cpp
    llama_engine/common/ngram-cache.cpp
    llama_engine/examples/llava/clip.cpp
    llama_engine/examples/llava/llava.cpp
    native-lib.cpp
)

add_library(
    llama_jni
    SHARED
    ${LLAMA_SRCS}
)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)
find_library(android-lib android)

target_link_libraries(
    llama_jni
    ${log-lib}
    ${jnigraphics-lib}
    ${android-lib}
)
