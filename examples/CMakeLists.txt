############################## P1 模块化：条件构建
# 依赖 HTTP 适配器的示例/测试仅在 ENABLE_HTTP + CURL_FOUND 时构建。
# 依赖 MEDIA 编码发布的示例/测试仅在 ENABLE_MEDIA 时构建。

# 辅助函数：创建可执行文件，链接 facade，设置输出目录
function(dh_add_example target_name source_file)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        return()
    endif()
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE digital_human_core)
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

# 辅助函数：同上，但额外链接条件库（用于直接使用 FFmpeg/PortAudio 头的示例）
function(dh_add_example_with_libs target_name source_file)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        return()
    endif()
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE digital_human_core)
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

############################## 基础演示与测试（无外部依赖）
dh_add_example(demo_app                 demo.cpp)
dh_add_example(opencv_test              opencv_test.cpp)
dh_add_example(ncnn_test                ncnn_test.cpp)

############################## 音频处理单元测试
dh_add_example(ring_buffer_test         ring_buffer_test.cpp)
dh_add_example(audio_framer_test        audio_framer_test.cpp)
dh_add_example(preemphasis_test         preemphasis_test.cpp)
dh_add_example(rmsnorm_test             rmsnorm_test.cpp)
dh_add_example(cmvn_test                cmvn_test.cpp)
dh_add_example(noise_reduction_test     noise_reduction_test.cpp)
dh_add_example(vad_test                 vad_test.cpp)
dh_add_example(mel_feature_test         mel_feature_test.cpp)
dh_add_example(audio_loader_test        audio_loader_test.cpp)
dh_add_example(ffmpeg_audio_test        ffmpeg_audio_test.cpp)
dh_add_example(offline_audio_test       offline_audio_test.cpp)

############################## 视觉/人脸单元测试
dh_add_example(image_load_test          image_load_test.cpp)
dh_add_example(face_detector_test       face_detector_test.cpp)
dh_add_example(landmark_test            landmark_test.cpp)
dh_add_example(face_aligner_test        face_aligner_test.cpp)
dh_add_example(face_mask_generator_test face_mask_generator_test.cpp)

############################## Pipeline 集成测试
dh_add_example(pipeline_test            pipeline_test.cpp)
dh_add_example(audio_processor_test     audio_processor_test.cpp)
dh_add_example(inference_worker_test    inference_worker_test.cpp)
dh_add_example(render_thread_test       render_thread_test.cpp)
dh_add_example(av_sync_test             av_sync_test.cpp)
dh_add_example(frame_scheduler_test     frame_scheduler_test.cpp)
dh_add_example(audio_sync_test          audio_sync_test.cpp)
dh_add_example(full_pipeline_test       full_pipeline_test.cpp)
dh_add_example(inference_render_pipeline_test inference_render_pipeline_test.cpp)
dh_add_example(video_output_test        video_output_test.cpp)
dh_add_example(fit_test                 fit_test.cpp)

############################## 回归与诊断测试
dh_add_example(bugfix_verification_test bugfix_verification_test.cpp)
dh_add_example(bugfix_v2_test           bugfix_v2_test.cpp)
dh_add_example(v1_review_fixes_test     v1_review_fixes_test.cpp)
dh_add_example(lip_sync_diagnose_test   lip_sync_diagnose_test.cpp)
dh_add_example(lip_sync_response_test   lip_sync_response_test.cpp)
dh_add_example(pipeline_lipsync_test    pipeline_lipsync_test.cpp)
dh_add_example(verify_postprocess_fix   verify_postprocess_fix.cpp)
dh_add_example(dump_fusion_mask         dump_fusion_mask.cpp)
dh_add_example(sdk_model_error_test     sdk_model_error_test.cpp)
dh_add_example(sdk_process_file_test    sdk_process_file_test.cpp)

############################## 会话模块测试（仅依赖抽象接口）
dh_add_example(dialog_module_test       dialog_module_test.cpp)
dh_add_example(avatar_upload_test       avatar_upload_test.cpp)
dh_add_example(lifecycle_safety_test    lifecycle_safety_test.cpp)
dh_add_example(conversation_sdk_integration_test conversation_sdk_integration_test.cpp)

############################## 性能基准测试
dh_add_example(perf_benchmark           perf_benchmark.cpp)
dh_add_example(perf_pipeline_only       perf_pipeline_only.cpp)
dh_add_example(perf_baseline_compare    perf_baseline_compare.cpp)
dh_add_example(perf_p1_strategy         perf_p1_strategy.cpp)

############################## HTTP 适配器测试（需要 libcurl + HTTP 模块）
if(DIGITAL_HUMAN_ENABLE_HTTP AND CURL_FOUND)
    dh_add_example(http_service_client_test   http_service_client_test.cpp)
    dh_add_example(llama_cpp_client_test      llama_cpp_client_test.cpp)
endif()

############################## 媒体发布测试（需要 ENABLE_MEDIA）
if(DIGITAL_HUMAN_ENABLE_MEDIA)
    dh_add_example(stream_publisher_test      stream_publisher_test.cpp)
    dh_add_example(stream_network_publish_test stream_network_publish_test.cpp)
    dh_add_example(conversation_stream_integration_test conversation_stream_integration_test.cpp)
endif()

############################## 完整链路测试（需要 HTTP + MEDIA）
if(DIGITAL_HUMAN_ENABLE_HTTP AND CURL_FOUND AND DIGITAL_HUMAN_ENABLE_MEDIA)
    dh_add_example(full_conversation_chain_test full_conversation_chain_test.cpp)
    dh_add_example(realtime_avatar_conversation  realtime_avatar_conversation.cpp)
endif()

# 这些测试直接调用 FFmpeg C API，不能只依赖 facade 的 PRIVATE 媒体依赖。
foreach(ffmpeg_example
        ffmpeg_audio_test
        stream_publisher_test
        conversation_stream_integration_test
        full_conversation_chain_test)
    if(TARGET ${ffmpeg_example})
        target_link_libraries(${ffmpeg_example} PRIVATE PkgConfig::FFMPEG)
    endif()
endforeach()

############################## CTest 注册与标签分层
# 目标：所有无外部依赖的测试注册到 CTest，并按 unit / integration / model /
# network / perf 分层，使 CI 能按 label 选择性执行。
if(BUILD_TESTING)
    function(dh_register_test target_name labels timeout_sec)
        if(TARGET ${target_name})
            add_test(NAME ${target_name} COMMAND ${target_name})
            set_tests_properties(${target_name} PROPERTIES
                LABELS "${labels}"
                TIMEOUT "${timeout_sec}")
        endif()
    endfunction()

    # ---- unit：纯逻辑 / 合成数据，CI 默认执行 ----
    dh_register_test(dialog_module_test         "unit" 60)
    dh_register_test(lifecycle_safety_test      "unit" 60)
    dh_register_test(avatar_upload_test         "unit" 60)
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
    dh_register_test(bugfix_v2_test             "unit" 30)
    dh_register_test(bugfix_verification_test   "unit" 30)
    dh_register_test(v1_review_fixes_test       "unit" 30)
    dh_register_test(verify_postprocess_fix     "unit" 30)
    dh_register_test(sdk_model_error_test       "unit" 30)
    dh_register_test(opencv_test                "unit" 30)

    # ---- integration：pipeline / 编码 / 文件 IO，CI 可执行 ----
    dh_register_test(pipeline_test                  "integration" 60)
    dh_register_test(sdk_process_file_test          "integration" 60)
    dh_register_test(stream_publisher_test          "integration" 60)
    dh_register_test(video_output_test              "integration" 60)
    dh_register_test(fit_test                       "integration" 60)
    dh_register_test(ffmpeg_audio_test              "integration" 60)
    dh_register_test(audio_loader_test              "integration" 60)
    dh_register_test(offline_audio_test             "integration" 60)
    dh_register_test(inference_render_pipeline_test "integration" 60)

    # ---- model：需要真实模型文件，nightly/manual ----
    dh_register_test(model_loader_test             "model" 120)
    dh_register_test(ncnn_test                     "model" 120)
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
