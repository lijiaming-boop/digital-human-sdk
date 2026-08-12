cmake_minimum_required(VERSION 3.16)
project(DigitalHumanSDK VERSION 0.2.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

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

############################## 构建选项
# P1 模块化：可选依赖按需启用，不启用时不要求对应库存在。
option(BUILD_EXAMPLES "Build example programs and tests" ON)
option(DIGITAL_HUMAN_ENABLE_HTTP
    "Build HTTP text-generation and TTS clients when libcurl is available" ON)
option(DIGITAL_HUMAN_ENABLE_MEDIA
    "Build H.264/AAC encoding and stream publishing (requires FFmpeg)" ON)
option(DIGITAL_HUMAN_ENABLE_AUDIO_IO
    "Build local audio playback (requires PortAudio)" ON)

############################## 必需依赖
# OpenMP — 运行时核心推理与图像处理并行
find_package(OpenMP REQUIRED)

# OpenCV — 运行时核心、头像、会话（cv::Mat 出现在公共 API）
find_package(OpenCV REQUIRED)

# ncnn — 模型推理（Wav2Lip / SCRFD）
option(DIGITAL_HUMAN_REQUIRE_NCNN_VULKAN
    "Require an ncnn build with Vulkan support for GPU inference"
    ON)
if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET
   AND EXISTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/ncnn/ncnnConfig.cmake")
    unset(ncnn_DIR CACHE)
    find_package(ncnn CONFIG REQUIRED
        PATHS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/ncnn"
        NO_DEFAULT_PATH)
else()
    find_package(ncnn CONFIG REQUIRED)
endif()
if(DIGITAL_HUMAN_REQUIRE_NCNN_VULKAN AND NOT NCNN_VULKAN)
    message(FATAL_ERROR
        "DigitalHumanSDK requires Vulkan-enabled ncnn. Install ncnn[vulkan] "
        "or configure with -DDIGITAL_HUMAN_REQUIRE_NCNN_VULKAN=OFF for CPU-only builds.")
endif()
if(NCNN_VULKAN)
    find_package(Vulkan REQUIRED)
endif()
message(STATUS ">> ncnn Found")

# PkgConfig — 用于 FFmpeg / libcurl / PortAudio
find_package(PkgConfig REQUIRED)

############################## 可选依赖
# FFmpeg — 音频加载 + 媒体编码发布
# 音频加载 (audio_loader) 是运行时核心的一部分，因此即使不启用 MEDIA，
# 运行时仍需要 FFmpeg。MEDIA 启用时额外用于 stream_publisher。
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswresample
    libswscale
)
message(STATUS ">> FFMPEG Found: ${FFMPEG_VERSION}")

# libcurl — HTTP 文本生成 / TTS 适配
if(DIGITAL_HUMAN_ENABLE_HTTP)
    pkg_check_modules(CURL QUIET IMPORTED_TARGET libcurl)
    if(CURL_FOUND)
        message(STATUS ">> libcurl Found: ${CURL_VERSION}")
    else()
        message(WARNING
            "libcurl development files were not found; HTTP clients will not "
            "be built. Dialog interfaces and tests remain available.")
    endif()
endif()

# PortAudio — 本机音频播放
# 当前 render_thread / audio_sync_scheduler 直接持有 AudioPlayer 对象，
# 尚未通过接口抽象解耦，因此 AUDIO_IO 实际为必选。保留选项用于未来解耦后
# 的最小构建；关闭时构建会在编译阶段报错，给出明确提示。
if(DIGITAL_HUMAN_ENABLE_AUDIO_IO)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
    message(STATUS ">> PortAudio Found: ${PORTAUDIO_VERSION}")
else()
    message(FATAL_ERROR
        "DIGITAL_HUMAN_ENABLE_AUDIO_IO=OFF is not yet supported: "
        "render_thread and audio_sync_scheduler directly depend on AudioPlayer. "
        "Future refactoring will abstract AudioPlayer behind an interface.")
endif()

message(STATUS "OpenCV library status:")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

############################## 子目录
add_subdirectory(src)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

############################## CMake package export
# 使外部项目能通过 find_package(DigitalHumanSDK) 发现并链接本库。
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/DigitalHumanSDKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/DigitalHumanSDKConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DigitalHumanSDK
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DigitalHumanSDKConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/DigitalHumanSDKConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/DigitalHumanSDKConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DigitalHumanSDK
)
