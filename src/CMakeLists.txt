# 自动搜集源文件
# 使用 GLOB_RECURSE 递归查找 src 下的所有子目录的 .cpp 文件
# CONFIGURE_DEPENDS: 当新增/删除源文件时自动重新运行 CMake
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "avatar/*.cpp"
    "audio/*.cpp"
    "video/*.cpp"
    "core/*.cpp"
    "utils/*.cpp"
    "model/*.cpp"
    "network/*.cpp"
    "dialog/*.cpp"
    "tts/*.cpp"
    "media/*.cpp"
)

# 生成 SDK 库
add_library(digital_human_core SHARED ${SRC_FILES})

# 头文件搜索路径（target 级，PUBLIC 让依赖此库的 target 自动继承）
target_include_directories(digital_human_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 第三方库链接（target 级，PUBLIC 让下游 target 自动获得头文件路径和库）
target_link_libraries(digital_human_core PUBLIC
    ${OpenCV_LIBS}
    ncnn
    PkgConfig::FFMPEG
    ${PORTAUDIO_LIBRARIES}
)

if(NCNN_VULKAN)
    # ncnn loads the Vulkan loader at runtime. Link it explicitly so CMake
    # carries the loader and its runtime search path into SDK consumers.
    target_link_libraries(digital_human_core PUBLIC Vulkan::Vulkan)
endif()

# 第三方头文件路径（target 级）
target_include_directories(digital_human_core PRIVATE
    ${PORTAUDIO_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

if(DIGITAL_HUMAN_ENABLE_HTTP_CLIENT AND CURL_FOUND)
    target_link_libraries(digital_human_core PRIVATE PkgConfig::CURL)
    target_compile_definitions(digital_human_core PRIVATE DIGITAL_HUMAN_HAS_CURL=1)
endif()

# OpenMP（target 级）
target_link_libraries(digital_human_core PUBLIC OpenMP::OpenMP_CXX)

# 编译选项：按构建类型自动选择，避免全局 -O3 / -march=native 污染 Debug 构建
# Debug: -O0 -g -DDEBUG；Release: -O2 -DNDEBUG（不使用 -march=native 以保证可移植）
target_compile_options(digital_human_core PRIVATE
    $<$<CONFIG:Debug>:-O0;-g;-DDEBUG>
    $<$<CONFIG:Release>:-O2;-DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O2;-g;-DNDEBUG>
    $<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>
    -Wall
    -Wextra
)
