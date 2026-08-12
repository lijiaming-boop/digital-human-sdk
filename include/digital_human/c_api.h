#pragma once

/// @file c_api.h
/// @brief 版本化 C API（P1-3.8d）
///
/// 提供 opaque handle + POD 配置，保证跨编译器 ABI。
/// C++ API 保留为便捷封装，但不承诺跨编译器 ABI。

#include <stdint.h>
#include "digital_human/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/// API 版本号（0x00020000 = 0.2.0），用于运行时 ABI 检查
DH_API uint32_t dh_sdk_api_version(void);

/// Opaque handle
typedef struct dh_sdk_t dh_sdk_t;

/// POD 配置（纯 C 兼容，无 STL 类型）
typedef struct {
    uint32_t   struct_size;    ///< 至少为当前版本 sizeof(dh_sdk_config_t)
    const char* model_path;    ///< Wav2Lip 模型路径
    const char* face_model_path; ///< 人脸检测模型路径
    int   width;               ///< 输出视频宽
    int   height;              ///< 输出视频高
    int   fps;                 ///< 输出帧率
    int   audio_sample_rate;   ///< 音频采样率
    int   audio_channels;      ///< 音频通道数
    int   inference_threads;   ///< 推理线程数
    int   enable_gpu;          ///< 0=CPU, 1=Vulkan
} dh_sdk_config_t;

/// 错误码
typedef enum {
    DH_ERROR_OK = 0,
    DH_ERROR_CONFIG,
    DH_ERROR_LIFECYCLE,
    DH_ERROR_MODEL,
    DH_ERROR_AUDIO,
    DH_ERROR_VIDEO,
    DH_ERROR_SYNC,
    DH_ERROR_DIALOG,
    DH_ERROR_TTS,
    DH_ERROR_NETWORK,
    DH_ERROR_MEDIA,
    DH_ERROR_TIMEOUT,
    DH_ERROR_CANCELLED,
    DH_ERROR_UNKNOWN = 99,
} dh_error_code_t;

/// 错误结构
typedef struct {
    uint32_t   struct_size;    ///< 设为调用方可用的结构体大小
    dh_error_code_t error_code;
    const char* module;        ///< "pipeline" / "dialog" / "media" / ...
    const char* message;       ///< 可读消息；有效至同线程下一次 C API 调用
    const char* cause;         ///< 底层原因；有效期同 message，可为 NULL
} dh_error_t;

/// 生命周期
DH_API dh_sdk_t* dh_sdk_create(const dh_sdk_config_t* config, dh_error_t* error);
DH_API void      dh_sdk_destroy(dh_sdk_t* sdk);
DH_API int       dh_sdk_start(dh_sdk_t* sdk, dh_error_t* error);
DH_API int       dh_sdk_stop(dh_sdk_t* sdk, int timeout_ms, dh_error_t* error);
DH_API int       dh_sdk_pause(dh_sdk_t* sdk, dh_error_t* error);
DH_API int       dh_sdk_resume(dh_sdk_t* sdk, dh_error_t* error);

/// 音视频供料（buffer 生命周期由调用者管理）
DH_API int dh_sdk_push_audio(dh_sdk_t* sdk,
                              const float* samples,
                              int64_t sample_count,
                              int64_t pts_ms,
                              dh_error_t* error);

DH_API int dh_sdk_push_video(dh_sdk_t* sdk,
                              const uint8_t* bgr_data,
                              int width, int height, int stride,
                              int64_t pts_ms,
                              dh_error_t* error);

/// 标记输入结束
DH_API void dh_sdk_mark_audio_eos(dh_sdk_t* sdk);
DH_API void dh_sdk_mark_video_eos(dh_sdk_t* sdk);

/// 获取状态
typedef enum {
    DH_STATE_UNINITIALIZED = 0,
    DH_STATE_INITIALIZED,
    DH_STATE_RUNNING,
    DH_STATE_PAUSED,
    DH_STATE_STOPPING,
    DH_STATE_STOPPED,
} dh_state_t;

DH_API dh_state_t dh_sdk_get_state(dh_sdk_t* sdk);
DH_API const char* dh_sdk_get_last_error(dh_sdk_t* sdk);

/// 指标快照（简化版，完整指标通过 C++ API 获取）
typedef struct {
    uint32_t struct_size;
    int64_t  video_frames_in;
    int64_t  video_frames_encoded;
    int64_t  video_frames_dropped;
    int64_t  audio_samples_in;
    int64_t  packets_written;
    double   actual_fps;
} dh_metrics_t;

DH_API int dh_sdk_get_metrics(dh_sdk_t* sdk, dh_metrics_t* metrics);

#ifdef __cplusplus
}
#endif
