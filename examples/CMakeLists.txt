# 构建 Demo App（基础演示）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp")
    add_executable(demo_app demo.cpp)
    # 链接 SDK 核心库
    target_link_libraries(demo_app PRIVATE digital_human_core)
endif()

#构建OpenCv测试程序(验证图像功能)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencv_test.cpp")
    add_executable(opencv_test opencv_test.cpp)

    # 链接 SDK 核心库
    # 因为 digital_human_core 在 src 中使用了 PUBLIC 链接 OpenCV，
    # 所以这里会自动获得 OpenCV 的头文件路径和库文件，无需重复 link OpenCV
    target_link_libraries(opencv_test PRIVATE digital_human_core)
endif()

# 3. 构建 ncnn 推理测试
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn_test.cpp")
    add_executable(ncnn_test ncnn_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(ncnn_test PRIVATE digital_human_core)
endif()

# 4. 构建 FFmpeg 测试程序 (验证音频功能)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_audio_test.cpp")
    add_executable(ffmpeg_audio_test ffmpeg_audio_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(ffmpeg_audio_test PRIVATE digital_human_core)
endif()

# 5. 构建 dlib 测试程序 (验证人脸检测和关键点提取)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dlib_test.cpp")
    add_executable(dlib_test dlib_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(dlib_test PRIVATE digital_human_core)
endif()

# 6. 构建 ImageLoader 测试程序
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/image_load_test.cpp")
    add_executable(image_load_test image_load_test.cpp)
    
    # 链接 SDK 核心库
    target_link_libraries(image_load_test PRIVATE digital_human_core)
    
    # 设置输出目录
    set_target_properties(image_load_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 7. 构建 FaceDetector 测试程序
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/face_detector_test.cpp")
    add_executable(face_detector_test face_detector_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(face_detector_test PRIVATE digital_human_core)

    # 设置输出目录
    set_target_properties(face_detector_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 8. 构建 Landmark 测试程序
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/landmark_test.cpp")
    add_executable(landmark_test landmark_test.cpp)

    target_link_libraries(landmark_test PRIVATE digital_human_core)

    set_target_properties(landmark_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

message(STATUS "Configured example targets.")