cmake_minimum_required(VERSION 3.16)
project(DigitalHumanSDK VERSION 0.1.0 LANGUAGES CXX)

############################## 全局配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认构建类型为 Release（仅在未指定时设置，避免覆盖 preset/CLI 配置）
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 定义路径宏（给后续 target 使用）
add_compile_definitions(
    ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

############################## OpenMP
find_package(OpenMP REQUIRED)

############################## 第三方依赖
# 注意：不再使用全局 include_directories / link_directories / add_compile_options，
# 改为 target 级配置，避免污染所有 target（包括示例和测试）。
# 也不再无条件添加 -O3 / -march=native / -DNDEBUG，改由 CMAKE_BUILD_TYPE 控制，
# Debug 构建可以正常启用 -O0 -g，二进制可移植到其他 CPU。
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)
message(STATUS ">> ncnn Found")
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswresample
    libswscale
)

message(STATUS ">> FFMPEG Found: ${FFMPEG_VERSION}")
message(STATUS ">> FFMPEG Include: ${FFMPEG_INCLUDE_DIRS}")

# 查找 PortAudio (音频播放)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
message(STATUS ">> PortAudio Found: ${PORTAUDIO_VERSION}")
message(STATUS ">> PortAudio Include: ${PORTAUDIO_INCLUDE_DIRS}")

# 查找 dlib
find_package(dlib REQUIRED)
message(STATUS ">> dlib Found: ${dlib_VERSION}")
# dlib::dlib target 会自动处理 include path

message(STATUS "OpenCV library status:")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

############################## 子目录
add_subdirectory(src)
add_subdirectory(examples)
