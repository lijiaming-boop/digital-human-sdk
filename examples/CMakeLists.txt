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

# 28. 构建 InferenceWorker 推理线程测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/inference_worker_test.cpp")
    add_executable(inference_worker_test inference_worker_test.cpp)

    target_link_libraries(inference_worker_test PRIVATE digital_human_core)

    set_target_properties(inference_worker_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 29. 构建 RenderThread 渲染线程测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/render_thread_test.cpp")
    add_executable(render_thread_test render_thread_test.cpp)

    target_link_libraries(render_thread_test PRIVATE digital_human_core)

    set_target_properties(render_thread_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 30. 构建全链路流程集成测试
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/full_pipeline_test.cpp")
    add_executable(full_pipeline_test full_pipeline_test.cpp)

    target_link_libraries(full_pipeline_test PRIVATE digital_human_core)

    set_target_properties(full_pipeline_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 31. 构建推理+渲染联调测试
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/inference_render_pipeline_test.cpp")
    add_executable(inference_render_pipeline_test inference_render_pipeline_test.cpp)

    target_link_libraries(inference_render_pipeline_test PRIVATE digital_human_core)

    set_target_properties(inference_render_pipeline_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 32. 构建离线音频测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/offline_audio_test.cpp")
    add_executable(offline_audio_test offline_audio_test.cpp)

    target_link_libraries(offline_audio_test PRIVATE digital_human_core)

    set_target_properties(offline_audio_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 33. 构建拟合效果测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fit_test.cpp")
    add_executable(fit_test fit_test.cpp)

    target_link_libraries(fit_test PRIVATE digital_human_core)

    set_target_properties(fit_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 34. 构建可视化输出测试程序 — 需要系统 OpenCV highgui，暂跳过
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/visual_output_test.cpp")
#     add_executable(visual_output_test visual_output_test.cpp)
#     target_link_libraries(visual_output_test PRIVATE digital_human_core)
#     target_include_directories(visual_output_test PRIVATE /usr/include/opencv4)
#     set_target_properties(visual_output_test PROPERTIES
#         RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#     )
# endif()

# 35. 构建音视频输出测试程序
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/video_output_test.cpp")
    add_executable(video_output_test video_output_test.cpp)

    target_link_libraries(video_output_test PRIVATE digital_human_core)

    set_target_properties(video_output_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 36. 构建诊断测试程序 — 需要系统 OpenCV highgui，暂跳过
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/diagnose_output.cpp")
#     add_executable(diagnose_output diagnose_output.cpp)
#     target_link_libraries(diagnose_output PRIVATE digital_human_core)
#     target_include_directories(diagnose_output PRIVATE /usr/include/opencv4)
#     set_target_properties(diagnose_output PROPERTIES
#         RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#     )
# endif()

# 37. Bug Fix Verification Test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bugfix_verification_test.cpp")
    add_executable(bugfix_verification_test bugfix_verification_test.cpp)
    target_link_libraries(bugfix_verification_test PRIVATE digital_human_core)
    set_target_properties(bugfix_verification_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 第二轮 Bug 修复验证测试（2026-07-16 代码审查）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bugfix_v2_test.cpp")
    add_executable(bugfix_v2_test bugfix_v2_test.cpp)
    target_link_libraries(bugfix_v2_test PRIVATE digital_human_core)
    set_target_properties(bugfix_v2_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# v1 代码审查报告修复验证测试（7 项主要问题逐条验证）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/v1_review_fixes_test.cpp")
    add_executable(v1_review_fixes_test v1_review_fixes_test.cpp)
    target_link_libraries(v1_review_fixes_test PRIVATE digital_human_core)
    set_target_properties(v1_review_fixes_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 唇形同步诊断测试 — 验证模型输出是否随音频输入变化
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lip_sync_diagnose_test.cpp")
    add_executable(lip_sync_diagnose_test lip_sync_diagnose_test.cpp)
    target_link_libraries(lip_sync_diagnose_test PRIVATE digital_human_core)
    set_target_properties(lip_sync_diagnose_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 端到端性能基准测试 — 测量 fps、延迟分布、缓存命中率
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf_benchmark.cpp")
    add_executable(perf_benchmark perf_benchmark.cpp)
    target_link_libraries(perf_benchmark PRIVATE digital_human_core)
    set_target_properties(perf_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 纯流水线性能基准（不依赖 Wav2Lip 推理，验证缓存/定时）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf_pipeline_only.cpp")
    add_executable(perf_pipeline_only perf_pipeline_only.cpp)
    target_link_libraries(perf_pipeline_only PRIVATE digital_human_core)
    set_target_properties(perf_pipeline_only PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# P0 优化前后对比基准
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf_baseline_compare.cpp")
    add_executable(perf_baseline_compare perf_baseline_compare.cpp)
    target_link_libraries(perf_baseline_compare PRIVATE digital_human_core)
    set_target_properties(perf_baseline_compare PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Pipeline 端到端口型同步 + 帧率验证（修复后验收测试）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pipeline_lipsync_test.cpp")
    add_executable(pipeline_lipsync_test pipeline_lipsync_test.cpp)
    target_link_libraries(pipeline_lipsync_test PRIVATE digital_human_core)
    set_target_properties(pipeline_lipsync_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# P1 性能策略验收：P0/P1 回归 + 线程预算 sweep + GPU 端到端验证 + ROI 收益量化
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/perf_p1_strategy.cpp")
    add_executable(perf_p1_strategy perf_p1_strategy.cpp)
    target_link_libraries(perf_p1_strategy PRIVATE digital_human_core)
    set_target_properties(perf_p1_strategy PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

message(STATUS "Configured example targets.")

# ProcessROI 修复效果验证 (合成数据, 不依赖模型/音频)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/verify_postprocess_fix.cpp")
    add_executable(verify_postprocess_fix verify_postprocess_fix.cpp)
    target_link_libraries(verify_postprocess_fix PRIVATE digital_human_core)
    set_target_properties(verify_postprocess_fix PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 导出 fusion_mask 用于验证渐变带宽度
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dump_fusion_mask.cpp")
    add_executable(dump_fusion_mask dump_fusion_mask.cpp)
    target_link_libraries(dump_fusion_mask PRIVATE digital_human_core)
    set_target_properties(dump_fusion_mask PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Wav2Lip 模型响应诊断 — 静音 vs 语音对照，判断模型是否对音频有响应
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lip_sync_response_test.cpp")
    add_executable(lip_sync_response_test lip_sync_response_test.cpp)
    target_link_libraries(lip_sync_response_test PRIVATE digital_human_core)
    set_target_properties(lip_sync_response_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 统一 SDK 文件接口回归：使用极小队列验证音视频独立生产不会反压死锁
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sdk_process_file_test.cpp")
    add_executable(sdk_process_file_test sdk_process_file_test.cpp)
    target_link_libraries(sdk_process_file_test PRIVATE digital_human_core)
    set_target_properties(sdk_process_file_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 统一 SDK 模型错误传播：不依赖真实模型即可验证错误码与错误消息
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sdk_model_error_test.cpp")
    add_executable(sdk_model_error_test sdk_model_error_test.cpp)
    target_link_libraries(sdk_model_error_test PRIVATE digital_human_core)
    set_target_properties(sdk_model_error_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 对话模块闭环：Fake 文本服务 + Fake TTS + 录制媒体 Sink，不依赖模型文件。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dialog_module_test.cpp")
    add_executable(dialog_module_test dialog_module_test.cpp)
    target_link_libraries(dialog_module_test PRIVATE digital_human_core)
    set_target_properties(dialog_module_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(BUILD_TESTING)
        add_test(
            NAME dialog_module
            COMMAND dialog_module_test
        )
    endif()
endif()

# 真实 HTTP 适配器冒烟测试；运行时传入本地 mock/实际服务 URL。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/http_service_client_test.cpp")
    add_executable(http_service_client_test http_service_client_test.cpp)
    target_link_libraries(http_service_client_test PRIVATE digital_human_core)
    set_target_properties(http_service_client_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# llama.cpp OpenAI-compatible /v1/chat/completions live/mock service test.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp_client_test.cpp")
    add_executable(llama_cpp_client_test llama_cpp_client_test.cpp)
    target_link_libraries(llama_cpp_client_test PRIVATE digital_human_core)
    set_target_properties(llama_cpp_client_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 真实 llama.cpp + HTTP TTS + Wav2Lip + H.264/AAC + 文件/网络发布闭环。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/full_conversation_chain_test.cpp")
    add_executable(full_conversation_chain_test
        full_conversation_chain_test.cpp)
    target_link_libraries(full_conversation_chain_test
        PRIVATE digital_human_core)
    set_target_properties(full_conversation_chain_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 使用真实模型验证会话控制器到 SDK 输出帧的端到端闭环。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/conversation_sdk_integration_test.cpp")
    add_executable(conversation_sdk_integration_test
        conversation_sdk_integration_test.cpp)
    target_link_libraries(conversation_sdk_integration_test
        PRIVATE digital_human_core)
    set_target_properties(conversation_sdk_integration_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 媒体发布层：BGR/PCM 编码为 H.264/AAC 并复用到本地 FLV。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/stream_publisher_test.cpp")
    add_executable(stream_publisher_test stream_publisher_test.cpp)
    target_link_libraries(stream_publisher_test PRIVATE digital_human_core)
    set_target_properties(stream_publisher_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 文本→TTS→真实数字人→H.264/AAC→FLV 的完整发布闭环。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/conversation_stream_integration_test.cpp")
    add_executable(conversation_stream_integration_test
        conversation_stream_integration_test.cpp)
    target_link_libraries(conversation_stream_integration_test
        PRIVATE digital_human_core)
    set_target_properties(conversation_stream_integration_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# 需要外部监听服务的 RTMP/RTSP 网络发布冒烟测试。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/stream_network_publish_test.cpp")
    add_executable(stream_network_publish_test
        stream_network_publish_test.cpp)
    target_link_libraries(stream_network_publish_test PRIVATE digital_human_core)
    set_target_properties(stream_network_publish_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
# Phase 1 lifecycle, shutdown, and configuration validation regression test.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lifecycle_safety_test.cpp")
    add_executable(lifecycle_safety_test lifecycle_safety_test.cpp)
    target_link_libraries(lifecycle_safety_test PRIVATE digital_human_core)
    set_target_properties(lifecycle_safety_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(BUILD_TESTING)
        add_test(
            NAME lifecycle_safety
            COMMAND lifecycle_safety_test
        )
    endif()
endif()


# Uploaded avatar decoding, validation, ownership, and hot-update regression test.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/avatar_upload_test.cpp")
    add_executable(avatar_upload_test avatar_upload_test.cpp)
    target_link_libraries(avatar_upload_test PRIVATE digital_human_core)
    set_target_properties(avatar_upload_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(BUILD_TESTING)
        add_test(
            NAME avatar_upload
            COMMAND avatar_upload_test
        )
    endif()
endif()


# Persistent multi-turn real conversation with uploaded avatar hot replacement.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/realtime_avatar_conversation.cpp")
    add_executable(realtime_avatar_conversation
        realtime_avatar_conversation.cpp)
    target_link_libraries(realtime_avatar_conversation PRIVATE digital_human_core)
    set_target_properties(realtime_avatar_conversation PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

############################## CTest 注册与标签分层
# 目标（P0 测试体系）：所有无外部依赖的测试注册到 CTest，并按 unit / integration /
# model / network / visual / perf 分层，使 CI 能按 label 选择性执行。
#   - unit        : 纯逻辑/合成数据，无模型、无网络、无音频设备
#   - integration : pipeline/编码/文件 IO 集成，可能依赖 assets 但不依赖真实模型
#   - model       : 需要真实 Wav2Lip / 人脸模型文件
#   - network     : 需要外部 HTTP / RTMP / RTSP 服务
#   - perf        : 性能基准，长耗时
# 已注册的测试（dialog_module / lifecycle_safety / avatar_upload）只补标签，不重复 add_test。
if(BUILD_TESTING)
    # 辅助函数：若 target 存在则注册到 CTest 并打标签与超时
    function(dh_register_test target_name labels timeout_sec)
        if(TARGET ${target_name})
            add_test(NAME ${target_name} COMMAND ${target_name})
            set_tests_properties(${target_name} PROPERTIES
                LABELS "${labels}"
                TIMEOUT "${timeout_sec}")
        endif()
    endfunction()

    # 为已通过 add_test 注册的测试补齐标签与超时
    set_tests_properties(dialog_module PROPERTIES LABELS "unit" TIMEOUT 60)
    set_tests_properties(lifecycle_safety PROPERTIES LABELS "unit" TIMEOUT 60)
    set_tests_properties(avatar_upload PROPERTIES LABELS "unit" TIMEOUT 60)

    # ---- unit：纯逻辑 / 合成数据，CI 默认执行 ----
    dh_register_test(ring_buffer_test           "unit" 30)
    dh_register_test(frame_scheduler_test       "unit" 30)
    dh_register_test(render_thread_test         "unit" 30)
    dh_register_test(av_sync_test               "unit" 30)
    dh_register_test(audio_sync_test            "unit" 30)
    dh_register_test(preemphasis_test           "unit" 30)
    dh_register_test(rmsnorm_test               "unit" 30)
    dh_register_test(noise_reduction_test       "unit" 30)
    dh_register_test(vad_test                   "unit" 30)
    dh_register_test(audio_framer_test          "unit" 30)
    dh_register_test(cmvn_test                  "unit" 30)
    dh_register_test(mel_feature_test           "unit" 30)
    dh_register_test(audio_processor_test       "unit" 30)
    dh_register_test(offline_audio_test         "unit" 30)
    dh_register_test(bugfix_v2_test             "unit" 30)
    dh_register_test(bugfix_verification_test   "unit" 30)
    dh_register_test(v1_review_fixes_test       "unit" 30)
    dh_register_test(verify_postprocess_fix     "unit" 30)
    dh_register_test(sdk_model_error_test       "unit" 30)
    dh_register_test(opencv_test                "unit" 30)
    dh_register_test(ncnn_test                  "unit" 30)

    # ---- integration：pipeline / 编码 / 文件 IO，CI 可执行 ----
    dh_register_test(pipeline_test                  "integration" 60)
    dh_register_test(sdk_process_file_test          "integration" 60)
    dh_register_test(stream_publisher_test          "integration" 60)
    dh_register_test(video_output_test              "integration" 60)
    dh_register_test(visual_output_test             "integration" 60)
    dh_register_test(fit_test                       "integration" 60)
    dh_register_test(ffmpeg_audio_test              "integration" 60)
    dh_register_test(audio_loader_test              "integration" 60)
    dh_register_test(inference_render_pipeline_test "integration" 60)

    # ---- model：需要真实模型文件，nightly/manual ----
    dh_register_test(model_loader_test             "model" 120)
    dh_register_test(model_inferencer_test         "model" 120)
    dh_register_test(output_processor_test         "model" 120)
    dh_register_test(face_detector_test            "model" 120)
    dh_register_test(face_aligner_test             "model" 120)
    dh_register_test(face_mask_generator_test      "model" 120)
    dh_register_test(image_load_test               "model" 120)
    dh_register_test(landmark_test                 "model" 120)
    dh_register_test(full_pipeline_test            "model" 120)
    dh_register_test(inference_worker_test         "model" 120)
    dh_register_test(lip_sync_diagnose_test        "model" 120)
    dh_register_test(lip_sync_response_test        "model" 120)
    dh_register_test(pipeline_lipsync_test         "model" 120)
    dh_register_test(conversation_sdk_integration_test "model" 120)

    # ---- network：需要外部 HTTP / RTMP / RTSP 服务，manual/nightly ----
    dh_register_test(http_service_client_test             "network" 90)
    dh_register_test(llama_cpp_client_test                "network" 90)
    dh_register_test(stream_network_publish_test          "network" 90)
    dh_register_test(conversation_stream_integration_test "network" 90)
    dh_register_test(full_conversation_chain_test         "network" 120)

    # ---- perf：性能基准，长耗时，单独 manual job ----
    dh_register_test(perf_benchmark        "perf" 300)
    dh_register_test(perf_pipeline_only    "perf" 300)
    dh_register_test(perf_baseline_compare "perf" 300)
    dh_register_test(perf_p1_strategy      "perf" 300)
endif()
