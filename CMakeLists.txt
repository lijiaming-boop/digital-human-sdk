cmake_minimum_required(VERSION 3.16)
project(DigitalHumanSDK VERSION 0.1.0 LANGUAGES CXX)

############################## 全局配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认开启优化
add_compile_options(-O3)

# 启用本机指令集优化（如 AVX2 / SSE）
add_compile_options(-march=native)
add_definitions(-DNDEBUG)

# 常用警告
add_compile_options(-Wall -Wextra)

############################## OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

############################## 输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

############################## 全局头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

############################## 第三方依赖
# ~/.local/lib 中有缺失的 .so 开发符号链接（libjpeg.so, libsqlite3.so 等）
link_directories(BEFORE "$ENV{HOME}/.local/lib")

# 查找opencv库（必须在 add_subdirectory 之前，否则子目录无法获取头文件路径）
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

# 查找 dlib
find_package(dlib REQUIRED)
message(STATUS ">> dlib Found: ${dlib_VERSION}")
# dlib::dlib target 会自动处理 include path


# 将OpenCV的头文件添加到全局包含路径
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FFMPEG_INCLUDE_DIRS})
include_directories(${dlib_INCLUDE_DIRS})
# 打印信息
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

############################## 子目录
add_subdirectory(src)
add_subdirectory(examples)

include_directories(${CMAKE_SOURCE_DIR}/include)