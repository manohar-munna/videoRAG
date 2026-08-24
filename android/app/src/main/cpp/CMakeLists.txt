cmake_minimum_required(VERSION 3.22.1)

project("videorag_native")

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
