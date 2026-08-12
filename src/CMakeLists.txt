############################## P1 模块化：单体库解耦
# 按职责将源文件拆分为独立 STATIC 库，每个库只声明自己真正需要的依赖。
# 最终由 digital_human_core SHARED 库组合所有模块，保持单一 DLL/SO 部署。
#
# 依赖方向（禁止反向依赖）：
#   avatar  → OpenCV
#   dialog  → OpenCV (core)               [纯抽象接口，不依赖 HTTP]
#   network → libcurl (可选)               [HTTP 传输原语]
#   http    → dialog + network + libcurl   [HTTP 适配器]
#   media   → FFmpeg                       [编码发布]
#   audio_io→ PortAudio                    [本机音频播放]
#   runtime → OpenCV + ncnn + OpenMP + FFmpeg [Pipeline + 模型 + 音频处理]
#   core(SHARED facade) → 组合上述全部 + sdk_entry + conversation_stream_bridge

# 公共 include 路径：所有模块都需要项目根 include/
set(DH_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# STATIC 库将被链接到 SHARED facade，必须编译为位置无关代码 (-fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#----------------------------------------------------------------------
# digital_human_avatar — 图片校验、解码和画布适配
#----------------------------------------------------------------------
add_library(dh_avatar STATIC
    avatar/avatar_image.cpp
)
target_include_directories(dh_avatar PUBLIC ${DH_INCLUDE_DIR})
target_link_libraries(dh_avatar PUBLIC ${OpenCV_LIBS})

#----------------------------------------------------------------------
# digital_human_dialog — 会话状态机和分句（纯抽象接口，不依赖 libcurl）
#----------------------------------------------------------------------
add_library(dh_dialog STATIC
    dialog/sentence_segmenter.cpp
    dialog/conversation_session.cpp
)
target_include_directories(dh_dialog PUBLIC ${DH_INCLUDE_DIR})
# PUBLIC: cv::Mat 出现在公共 API (UpdateAvatar/PushVideo)
target_link_libraries(dh_dialog PUBLIC ${OpenCV_LIBS})

#----------------------------------------------------------------------
# digital_human_network — HTTP 传输原语与 JSON 工具
#----------------------------------------------------------------------
if(DIGITAL_HUMAN_ENABLE_HTTP AND CURL_FOUND)
    add_library(dh_network STATIC
        network/http_client.cpp
        network/json_utils.cpp
    )
    target_include_directories(dh_network PUBLIC ${DH_INCLUDE_DIR})
    target_include_directories(dh_network PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(dh_network PRIVATE PkgConfig::CURL)
    target_compile_definitions(dh_network PRIVATE DIGITAL_HUMAN_HAS_CURL=1)
endif()

#----------------------------------------------------------------------
# digital_human_http — HTTP 文本生成与 TTS 适配器
#----------------------------------------------------------------------
if(DIGITAL_HUMAN_ENABLE_HTTP AND CURL_FOUND)
    add_library(dh_http STATIC
        dialog/http_text_generation_client.cpp
        dialog/llama_cpp_text_generation_client.cpp
        tts/http_tts_client.cpp
    )
    target_include_directories(dh_http PUBLIC ${DH_INCLUDE_DIR})
    target_include_directories(dh_http PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(dh_http
        PUBLIC dh_dialog           # 继承抽象接口
        PRIVATE dh_network PkgConfig::CURL)
    target_compile_definitions(dh_http PRIVATE DIGITAL_HUMAN_HAS_CURL=1)
endif()

#----------------------------------------------------------------------
# digital_human_media — H.264/AAC 编码和发布
#----------------------------------------------------------------------
if(DIGITAL_HUMAN_ENABLE_MEDIA)
    add_library(dh_media STATIC
        media/stream_publisher.cpp
    )
    target_include_directories(dh_media PUBLIC ${DH_INCLUDE_DIR})
    target_link_libraries(dh_media PUBLIC ${OpenCV_LIBS} PkgConfig::FFMPEG)
endif()

#----------------------------------------------------------------------
# digital_human_audio_io — 本机音频播放
#----------------------------------------------------------------------
if(DIGITAL_HUMAN_ENABLE_AUDIO_IO)
    add_library(dh_audio_io STATIC
        audio/audio_player.cpp
    )
    target_include_directories(dh_audio_io PUBLIC ${DH_INCLUDE_DIR})
    target_include_directories(dh_audio_io PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(dh_audio_io PUBLIC ${PORTAUDIO_LIBRARIES})
endif()

#----------------------------------------------------------------------
# digital_human_runtime — Pipeline、模型推理、同步、渲染和音频处理
#----------------------------------------------------------------------
add_library(dh_runtime STATIC
    # audio processing (不含 audio_player，该文件属于 audio_io)
    audio/audio_cmvn.cpp
    audio/audio_framer.cpp
    audio/audio_loader.cpp
    audio/audio_mel_feature_extract.cpp
    audio/audio_noise_reduction.cpp
    audio/audio_preemphasis.cpp
    audio/audio_ring_buffer.cpp
    audio/audio_rms_normalize.cpp
    audio/audio_vad.cpp
    audio/streaming_vad.cpp
    # model inference
    model/model_inferencer.cpp
    model/model_loader.cpp
    model/output_processor.cpp
    # core pipeline (不含 sdk_entry，该文件属于 facade)
    core/audio_processor.cpp
    core/audio_sync_scheduler.cpp
    core/av_sync.cpp
    core/face_aligner.cpp
    core/face_detector.cpp
    core/face_mask_generator.cpp
    core/frame_scheduler.cpp
    core/image_loader.cpp
    core/inference_worker.cpp
    core/pipeline.cpp
    core/render_thread.cpp
    core/video_processor.cpp
    core/worker_registry.cpp
    # P1: logger + metrics 基础设施
    core/logger.cpp
    core/log_context.cpp
    core/metrics.cpp
)
target_include_directories(dh_runtime PUBLIC ${DH_INCLUDE_DIR})
target_include_directories(dh_runtime PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(dh_runtime
    PUBLIC ${OpenCV_LIBS} ncnn OpenMP::OpenMP_CXX PkgConfig::FFMPEG)
if(NCNN_VULKAN)
    target_link_libraries(dh_runtime PUBLIC Vulkan::Vulkan)
endif()
if(TARGET dh_audio_io)
    target_link_libraries(dh_runtime PUBLIC dh_audio_io)
endif()

#----------------------------------------------------------------------
# digital_human_core — 对外 SHARED facade，组合所有模块
#----------------------------------------------------------------------
# facade 源文件：SDK 入口 + C API + 会话到媒体的桥接（跨层组合代码）
set(DH_FACADE_SOURCES
    core/sdk_entry.cpp
    core/c_api.cpp
)
if(TARGET dh_media)
    list(APPEND DH_FACADE_SOURCES
        media/conversation_stream_bridge.cpp)
endif()

add_library(digital_human_core SHARED ${DH_FACADE_SOURCES})
target_include_directories(digital_human_core PUBLIC
    $<BUILD_INTERFACE:${DH_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

# 编译 facade 时定义导出宏
target_compile_definitions(digital_human_core PRIVATE DIGITAL_HUMAN_CORE_EXPORTS)
set(DH_EXPORT_IMPLEMENTATION_TARGETS dh_runtime dh_avatar dh_dialog)
foreach(optional_target dh_http dh_network dh_media dh_audio_io)
    if(TARGET ${optional_target})
        list(APPEND DH_EXPORT_IMPLEMENTATION_TARGETS ${optional_target})
    endif()
endforeach()
foreach(implementation_target IN LISTS DH_EXPORT_IMPLEMENTATION_TARGETS)
    target_compile_definitions(${implementation_target}
        PRIVATE DIGITAL_HUMAN_CORE_EXPORTS)
endforeach()

# 链接所有模块 STATIC 库。使用 PRIVATE 使消费者只依赖 SHARED facade，
# 不直接链接内部 STATIC 库（STATIC 库有未解析符号，只在 SHARED 内部解析）。
# PUBLIC 依赖：公共头使用 cv::Mat，消费者编译/链接时需要 OpenCV。
# 使用 --whole-archive 强制包含 STATIC 库的全部对象文件，避免链接器
# 因 dead-stripping 丢弃未被 facade 直接引用的公共符号（如 ConversationSession）。
target_link_libraries(digital_human_core
    PUBLIC
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        ncnn
)
if(MSVC)
    target_link_libraries(digital_human_core PRIVATE ${DH_EXPORT_IMPLEMENTATION_TARGETS})
    foreach(implementation_target IN LISTS DH_EXPORT_IMPLEMENTATION_TARGETS)
        target_link_options(digital_human_core PRIVATE
            "/WHOLEARCHIVE:$<TARGET_FILE:${implementation_target}>")
    endforeach()
elseif(APPLE)
    target_link_options(digital_human_core PRIVATE "LINKER:-all_load")
    target_link_libraries(digital_human_core PRIVATE ${DH_EXPORT_IMPLEMENTATION_TARGETS})
else()
    target_link_libraries(digital_human_core PRIVATE
        "-Wl,--whole-archive"
        ${DH_EXPORT_IMPLEMENTATION_TARGETS}
        "-Wl,--no-whole-archive")
endif()

# 版本号与 SOVERSION
set_target_properties(digital_human_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# 符号可见性：默认隐藏，只导出标记 DH_API 的符号
set_target_properties(digital_human_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# 编译选项：按构建类型自动选择，避免全局 -O3 / -march=native 污染 Debug 构建
if(MSVC)
    target_compile_options(digital_human_core PRIVATE /W4)
else()
    target_compile_options(digital_human_core PRIVATE
        $<$<CONFIG:Debug>:-O0;-g;-DDEBUG>
        $<$<CONFIG:Release>:-O2;-DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:-O2;-g;-DNDEBUG>
        $<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>
        -Wall
        -Wextra)
endif()

############################## 安装规则
# 安装 SHARED facade（消费者只需链接此库）
install(TARGETS digital_human_core EXPORT DigitalHumanSDKTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(EXPORT DigitalHumanSDKTargets
    FILE DigitalHumanSDKTargets.cmake
    NAMESPACE DigitalHumanSDK::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DigitalHumanSDK)

# 安装公共头文件（按目录结构保留）
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
