#include "digital_human/c_api.h"
#include "digital_human_sdk.h"

#include <cstring>
#include <new>
#include <string>
#include <utility>

namespace {

dh_error_code_t sdk_error_to_c(digital_human::SDKError err) {
    switch (err) {
        case digital_human::SDKError::OK:                  return DH_ERROR_OK;
        case digital_human::SDKError::INVALID_CONFIG:      return DH_ERROR_CONFIG;
        case digital_human::SDKError::NOT_INITIALIZED:     return DH_ERROR_LIFECYCLE;
        case digital_human::SDKError::ALREADY_RUNNING:     return DH_ERROR_LIFECYCLE;
        case digital_human::SDKError::NOT_RUNNING:         return DH_ERROR_LIFECYCLE;
        case digital_human::SDKError::ALREADY_TERMINATED:  return DH_ERROR_LIFECYCLE;
        case digital_human::SDKError::MODEL_LOAD_FAILED:   return DH_ERROR_MODEL;
        case digital_human::SDKError::FACE_MODEL_LOAD_FAILED: return DH_ERROR_MODEL;
        case digital_human::SDKError::PIPELINE_START_FAILED: return DH_ERROR_LIFECYCLE;
        case digital_human::SDKError::GPU_NOT_AVAILABLE:   return DH_ERROR_MODEL;
        case digital_human::SDKError::INVALID_INPUT:       return DH_ERROR_AUDIO;
        case digital_human::SDKError::AUDIO_LOAD_FAILED:   return DH_ERROR_AUDIO;
        case digital_human::SDKError::IMAGE_LOAD_FAILED:   return DH_ERROR_VIDEO;
        case digital_human::SDKError::TIMEOUT:             return DH_ERROR_TIMEOUT;
        case digital_human::SDKError::SHUTDOWN_TIMEOUT:    return DH_ERROR_TIMEOUT;
        case digital_human::SDKError::UNKNOWN:             return DH_ERROR_UNKNOWN;
    }
    return DH_ERROR_UNKNOWN;
}

dh_state_t sdk_state_to_c(digital_human::SDKState state) {
    switch (state) {
        case digital_human::SDKState::UNINITIALIZED: return DH_STATE_UNINITIALIZED;
        case digital_human::SDKState::INITIALIZED:   return DH_STATE_INITIALIZED;
        case digital_human::SDKState::RUNNING:       return DH_STATE_RUNNING;
        case digital_human::SDKState::PAUSED:        return DH_STATE_PAUSED;
        case digital_human::SDKState::STOPPING:      return DH_STATE_STOPPING;
        case digital_human::SDKState::STOPPED:       return DH_STATE_STOPPED;
    }
    return DH_STATE_UNINITIALIZED;
}

void set_error(dh_error_t* out, dh_error_code_t code, const char* module,
               const char* message, const char* cause) {
    if (!out) return;
    if (out->struct_size != 0 && out->struct_size < sizeof(dh_error_t)) return;
    struct ErrorStorage {
        std::string module;
        std::string message;
        std::string cause;
    };
    static thread_local ErrorStorage storage;
    storage.module = module ? module : "";
    storage.message = message ? message : "";
    storage.cause = cause ? cause : "";
    if (out->struct_size == 0) out->struct_size = sizeof(dh_error_t);
    out->error_code = code;
    out->module = storage.module.c_str();
    out->message = storage.message.c_str();
    out->cause = cause ? storage.cause.c_str() : nullptr;
}

}  // namespace

extern "C" {

uint32_t dh_sdk_api_version(void) {
    return 0x00020000;  // 0.2.0
}

dh_sdk_t* dh_sdk_create(const dh_sdk_config_t* config, dh_error_t* error) {
    if (!config) {
        set_error(error, DH_ERROR_CONFIG, "c_api", "config is null", nullptr);
        return nullptr;
    }
    if (config->struct_size < sizeof(dh_sdk_config_t)) {
        set_error(error, DH_ERROR_CONFIG, "c_api",
                  "config struct_size mismatch", nullptr);
        return nullptr;
    }

    digital_human::SDKConfig cpp_config;
    if (config->model_path)
        cpp_config.lipsync_model_dir = config->model_path;
    if (config->face_model_path)
        cpp_config.face_model_dir = config->face_model_path;
    if (config->width > 0 && config->height > 0) {
        // width/height 在 SDKConfig 中没有直接字段，
        // 但 face_size 和 target_fps 等有。这里只映射已有字段。
    }
    if (config->fps > 0)
        cpp_config.target_fps = static_cast<double>(config->fps);
    if (config->audio_sample_rate > 0)
        cpp_config.audio_sample_rate = config->audio_sample_rate;
    if (config->audio_channels > 0)
        cpp_config.audio_channels = config->audio_channels;
    if (config->inference_threads > 0)
        cpp_config.inference_threads = config->inference_threads;
    cpp_config.enable_gpu = config->enable_gpu != 0;

    auto* sdk = new (std::nothrow) digital_human::DigitalHumanSDK();
    if (!sdk) {
        set_error(error, DH_ERROR_UNKNOWN, "c_api", "memory allocation failed", nullptr);
        return nullptr;
    }

    digital_human::SDKError err = sdk->Init(cpp_config);
    if (err != digital_human::SDKError::OK) {
        set_error(error, sdk_error_to_c(err), "pipeline",
                  sdk->GetLastError().c_str(), nullptr);
        delete sdk;
        return nullptr;
    }

    set_error(error, DH_ERROR_OK, "", "", nullptr);
    return reinterpret_cast<dh_sdk_t*>(sdk);
}

void dh_sdk_destroy(dh_sdk_t* sdk) {
    if (!sdk) return;
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    cpp->Stop();
    delete cpp;
}

int dh_sdk_start(dh_sdk_t* sdk, dh_error_t* error) {
    if (!sdk) {
        set_error(error, DH_ERROR_LIFECYCLE, "c_api", "sdk is null", nullptr);
        return 0;
    }
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    digital_human::SDKError err = cpp->Start();
    if (err != digital_human::SDKError::OK) {
        set_error(error, sdk_error_to_c(err), "pipeline",
                  cpp->GetLastError().c_str(), nullptr);
        return 0;
    }
    set_error(error, DH_ERROR_OK, "", "", nullptr);
    return 1;
}

int dh_sdk_stop(dh_sdk_t* sdk, int timeout_ms, dh_error_t* error) {
    if (!sdk) {
        set_error(error, DH_ERROR_LIFECYCLE, "c_api", "sdk is null", nullptr);
        return 0;
    }
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    digital_human::SDKError err = cpp->Stop(timeout_ms);
    if (err != digital_human::SDKError::OK) {
        set_error(error, sdk_error_to_c(err), "pipeline",
                  cpp->GetLastError().c_str(), nullptr);
        return 0;
    }
    set_error(error, DH_ERROR_OK, "", "", nullptr);
    return 1;
}

int dh_sdk_pause(dh_sdk_t* sdk, dh_error_t* error) {
    if (!sdk) {
        set_error(error, DH_ERROR_LIFECYCLE, "c_api", "sdk is null", nullptr);
        return 0;
    }
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    digital_human::SDKError err = cpp->Pause();
    if (err != digital_human::SDKError::OK) {
        set_error(error, sdk_error_to_c(err), "pipeline",
                  cpp->GetLastError().c_str(), nullptr);
        return 0;
    }
    set_error(error, DH_ERROR_OK, "", "", nullptr);
    return 1;
}

int dh_sdk_resume(dh_sdk_t* sdk, dh_error_t* error) {
    if (!sdk) {
        set_error(error, DH_ERROR_LIFECYCLE, "c_api", "sdk is null", nullptr);
        return 0;
    }
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    digital_human::SDKError err = cpp->Resume();
    if (err != digital_human::SDKError::OK) {
        set_error(error, sdk_error_to_c(err), "pipeline",
                  cpp->GetLastError().c_str(), nullptr);
        return 0;
    }
    set_error(error, DH_ERROR_OK, "", "", nullptr);
    return 1;
}

int dh_sdk_push_audio(dh_sdk_t* sdk, const float* samples,
                      int64_t sample_count, int64_t pts_ms,
                      dh_error_t* error) {
    if (!sdk) {
        set_error(error, DH_ERROR_LIFECYCLE, "c_api", "sdk is null", nullptr);
        return 0;
    }
    if (!samples || sample_count <= 0) {
        set_error(error, DH_ERROR_AUDIO, "c_api", "invalid audio buffer", nullptr);
        return 0;
    }
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    std::vector<float> pcm(samples, samples + sample_count);
    digital_human::SDKError err = cpp->PushAudio(pcm, pts_ms);
    if (err != digital_human::SDKError::OK) {
        set_error(error, sdk_error_to_c(err), "pipeline",
                  cpp->GetLastError().c_str(), nullptr);
        return 0;
    }
    set_error(error, DH_ERROR_OK, "", "", nullptr);
    return 1;
}

int dh_sdk_push_video(dh_sdk_t* sdk, const uint8_t* bgr_data,
                      int width, int height, int stride,
                      int64_t pts_ms, dh_error_t* error) {
    if (!sdk) {
        set_error(error, DH_ERROR_LIFECYCLE, "c_api", "sdk is null", nullptr);
        return 0;
    }
    if (!bgr_data || width <= 0 || height <= 0) {
        set_error(error, DH_ERROR_VIDEO, "c_api", "invalid video buffer", nullptr);
        return 0;
    }
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    // 构造 cv::Mat：BGR uint8，stride 作为行步长
    cv::Mat frame(height, width, CV_8UC3,
                  const_cast<uint8_t*>(bgr_data),
                  stride > 0 ? stride : width * 3);
    digital_human::SDKError err = cpp->PushVideo(frame, pts_ms);
    if (err != digital_human::SDKError::OK) {
        set_error(error, sdk_error_to_c(err), "pipeline",
                  cpp->GetLastError().c_str(), nullptr);
        return 0;
    }
    set_error(error, DH_ERROR_OK, "", "", nullptr);
    return 1;
}

void dh_sdk_mark_audio_eos(dh_sdk_t* sdk) {
    if (!sdk) return;
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    cpp->MarkAudioEOS();
}

void dh_sdk_mark_video_eos(dh_sdk_t* sdk) {
    if (!sdk) return;
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    cpp->MarkVideoEOS();
}

dh_state_t dh_sdk_get_state(dh_sdk_t* sdk) {
    if (!sdk) return DH_STATE_UNINITIALIZED;
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    return sdk_state_to_c(cpp->GetState());
}

const char* dh_sdk_get_last_error(dh_sdk_t* sdk) {
    if (!sdk) return "";
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    static thread_local std::string last;
    last = cpp->GetLastError();
    return last.c_str();
}

int dh_sdk_get_metrics(dh_sdk_t* sdk, dh_metrics_t* metrics) {
    if (!sdk || !metrics) return 0;
    if (metrics->struct_size != 0
        && metrics->struct_size < sizeof(dh_metrics_t)) return 0;
    if (metrics->struct_size == 0) metrics->struct_size = sizeof(dh_metrics_t);
    auto* cpp = reinterpret_cast<digital_human::DigitalHumanSDK*>(sdk);
    digital_human::SDKMetrics m = cpp->GetMetrics();
    metrics->video_frames_in = m.total_frames_in;
    metrics->video_frames_encoded = m.total_frames_out;
    metrics->video_frames_dropped = m.frames_dropped;
    metrics->audio_samples_in = m.audio_packets_in;
    metrics->packets_written = 0;  // publisher 层指标
    metrics->actual_fps = m.actual_fps;
    return 1;
}

}  // extern "C"
