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

# 9. 构建 FaceAligner 测试程序
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/face_aligner_test.cpp")
    add_executable(face_aligner_test face_aligner_test.cpp)

    target_link_libraries(face_aligner_test PRIVATE digital_human_core)

    set_target_properties(face_aligner_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 10. 构建 FaceMaskGenerator 测试程序
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/face_mask_generator_test.cpp")
    add_executable(face_mask_generator_test face_mask_generator_test.cpp)

    target_link_libraries(face_mask_generator_test PRIVATE digital_human_core)

    set_target_properties(face_mask_generator_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 11. 构建 AudioLoader 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_loader_test.cpp")
    add_executable(audio_loader_test audio_loader_test.cpp)

    target_link_libraries(audio_loader_test PRIVATE digital_human_core)

    set_target_properties(audio_loader_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 12. 构建 AudioFramer 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_framer_test.cpp")
    add_executable(audio_framer_test audio_framer_test.cpp)

    target_link_libraries(audio_framer_test PRIVATE digital_human_core)

    set_target_properties(audio_framer_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 13. 构建 PreEmphasis 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/preemphasis_test.cpp")
    add_executable(preemphasis_test preemphasis_test.cpp)

    target_link_libraries(preemphasis_test PRIVATE digital_human_core)

    set_target_properties(preemphasis_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 14. 构建 VAD 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vad_test.cpp")
    add_executable(vad_test vad_test.cpp)

    target_link_libraries(vad_test PRIVATE digital_human_core)

    set_target_properties(vad_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 15. 构建 RMSNormalize 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm_test.cpp")
    add_executable(rmsnorm_test rmsnorm_test.cpp)

    target_link_libraries(rmsnorm_test PRIVATE digital_human_core)

    set_target_properties(rmsnorm_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 16. 构建 CMVN 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmvn_test.cpp")
    add_executable(cmvn_test cmvn_test.cpp)

    target_link_libraries(cmvn_test PRIVATE digital_human_core)

    set_target_properties(cmvn_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 17. 构建 NoiseReduction 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/noise_reduction_test.cpp")
    add_executable(noise_reduction_test noise_reduction_test.cpp)

    target_link_libraries(noise_reduction_test PRIVATE digital_human_core)

    set_target_properties(noise_reduction_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 18. 构建 RingBuffer 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ring_buffer_test.cpp")
    add_executable(ring_buffer_test ring_buffer_test.cpp)

    target_link_libraries(ring_buffer_test PRIVATE digital_human_core)

    set_target_properties(ring_buffer_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 19. 构建 MelFeatureExtract 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mel_feature_test.cpp")
    add_executable(mel_feature_test mel_feature_test.cpp)

    target_link_libraries(mel_feature_test PRIVATE digital_human_core)

    set_target_properties(mel_feature_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 20. 构建 ModelLoader 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model_loader_test.cpp")
    add_executable(model_loader_test model_loader_test.cpp)

    target_link_libraries(model_loader_test PRIVATE digital_human_core)

    set_target_properties(model_loader_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 21. 构建 ModelInferencer 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model_inferencer_test.cpp")
    add_executable(model_inferencer_test model_inferencer_test.cpp)

    target_link_libraries(model_inferencer_test PRIVATE digital_human_core)

    set_target_properties(model_inferencer_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 22. 构建 OutputProcessor 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/output_processor_test.cpp")
    add_executable(output_processor_test output_processor_test.cpp)

    target_link_libraries(output_processor_test PRIVATE digital_human_core)

    set_target_properties(output_processor_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 23. 构建 AVSync 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/av_sync_test.cpp")
    add_executable(av_sync_test av_sync_test.cpp)

    target_link_libraries(av_sync_test PRIVATE digital_human_core)

    set_target_properties(av_sync_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 24. 构建 FrameScheduler 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/frame_scheduler_test.cpp")
    add_executable(frame_scheduler_test frame_scheduler_test.cpp)

    target_link_libraries(frame_scheduler_test PRIVATE digital_human_core)

    set_target_properties(frame_scheduler_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 25. 构建 AudioSyncScheduler 测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_sync_test.cpp")
    add_executable(audio_sync_test audio_sync_test.cpp)

    target_link_libraries(audio_sync_test PRIVATE digital_human_core)

    set_target_properties(audio_sync_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 26. 构建 Pipeline 多线程流水线测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pipeline_test.cpp")
    add_executable(pipeline_test pipeline_test.cpp)

    target_link_libraries(pipeline_test PRIVATE digital_human_core)

    set_target_properties(pipeline_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 27. 构建 AudioProcessor 音频处理线程测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_processor_test.cpp")
    add_executable(audio_processor_test audio_processor_test.cpp)

    target_link_libraries(audio_processor_test PRIVATE digital_human_core)

    set_target_properties(audio_processor_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

message(STATUS "Configured example targets.")