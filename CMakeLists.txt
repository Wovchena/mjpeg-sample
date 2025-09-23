cmake_minimum_required(VERSION 3.14)
project(mjpeg_random_flat CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable warnings
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(mjpeg_random main.cpp)

# --- Fetch nothings/stb (header-only) ---
include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
# Use the robust pattern that works in all 3.14+ CMake
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

# Tell the compiler where to find stb_image_write.h
target_include_directories(mjpeg_random PRIVATE ${stb_SOURCE_DIR})

# Silence MSVC secure CRT warnings in stb (optional)
if(MSVC)
    target_compile_definitions(mjpeg_random PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
