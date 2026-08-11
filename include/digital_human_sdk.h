#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <functional>

#include <opencv2/core.hpp>

namespace digital_human {

// ============================================================================
// 错误码
// ============================================================================

/// @brief SDK 错误码
enum class SDKError {
    OK                  = 0,    ///< 成功
    INVALID_CONFIG      = 1,    ///< 无效配置
    NOT_INITIALIZED     = 2,    ///< 未初始化
    ALREADY_RUNNING     = 3,    ///< 已在运行
    NOT_RUNNING         = 4,    ///< 未运行
    ALREADY_TERMINATED  = 5,    ///< 已终止（不可重启）
    MODEL_LOAD_FAILED   = 6,    ///< Wav2Lip 模型加载失败
    FACE_MODEL_LOAD_FAILED = 7, ///< 人脸模型加载失败
    PIPELINE_START_FAILED  = 8, ///< 流水线启动失败
    GPU_NOT_AVAILABLE   = 9,    ///< GPU 不可用
    INVALID_INPUT       = 10,   ///< 无效输入数据
    AUDIO_LOAD_FAILED   = 11,   ///< 音频文件加载失败
    IMAGE_LOAD_FAILED   = 12,   ///< 图像文件加载失败
    TIMEOUT             = 13,   ///< 等待超时
    UNKNOWN             = 99,   ///< 未知错误
};

/// @brief 错误码字符串描述
inline const char* SDKErrorToString(SDKError err) {
    switch (err) {
        case SDKError::OK:                     return "OK";
        case SDKError::INVALID_CONFIG:         return "INVALID_CONFIG";
        case SDKError::NOT_INITIALIZED:        return "NOT_INITIALIZED";
        case SDKError::ALREADY_RUNNING:        return "ALREADY_RUNNING";
        case SDKError::NOT_RUNNING:            return "NOT_RUNNING";
        case SDKError::ALREADY_TERMINATED:     return "ALREADY_TERMINATED";
        case SDKError::MODEL_LOAD_FAILED:      return "MODEL_LOAD_FAILED";
        case SDKError::FACE_MODEL_LOAD_FAILED: return "FACE_MODEL_LOAD_FAILED";
        case SDKError::PIPELINE_START_FAILED:  return "PIPELINE_START_FAILED";
        case SDKError::GPU_NOT_AVAILABLE:      return "GPU_NOT_AVAILABLE";
        case SDKError::INVALID_INPUT:          return "INVALID_INPUT";
        case SDKError::AUDIO_LOAD_FAILED:      return "AUDIO_LOAD_FAILED";
        case SDKError::IMAGE_LOAD_FAILED:      return "IMAGE_LOAD_FAILED";
        case SDKError::TIMEOUT:                return "TIMEOUT";
        case SDKError::UNKNOWN:                return "UNKNOWN";
        default:                               return "UNKNOWN";
    }
}

// ============================================================================
// 状态枚举
// ============================================================================

/// @brief SDK 运行状态
enum class SDKState {
    UNINITIALIZED,  ///< 未初始化
    INITIALIZED,    ///< 已初始化，未启动
    RUNNING,        ///< 运行中
    PAUSED,         ///< 已暂停
    STOPPED,        ///< 已停止（不可重启）
};

// ============================================================================
// 配置
// ============================================================================

/// @brief SDK 配置
struct SDKConfig {
    // ---- 模型路径（Init 时自动加载，留空则需手动调用 Load* 接口）----
    std::string lipsync_model_dir;  ///< Wav2Lip 模型目录（含 .param/.bin）
    std::string face_model_dir;     ///< SCRFD + 2D106 人脸模型目录

    // ---- 音频参数 ----
    int    audio_sample_rate   = 16000;  ///< 音频采样率（Hz）
    int    audio_channels      = 1;      ///< 音频声道数
    int    audio_frame_size    = 800;    ///< 音频帧大小（样本数）
    int    audio_hop_size      = 200;    ///< 音频帧移（样本数）

    // ---- 视频参数 ----
    double target_fps          = 25.0;   ///< 目标帧率
    int    face_size           = 96;     ///< 对齐人脸尺寸

    // ---- 同步参数 ----
    double sync_threshold_ms   = 30.0;   ///< 同步告警阈值（ms）
    double max_drift_ms        = 100.0;  ///< 严重偏移丢帧阈值（ms）

    // ---- 性能开关 ----
    bool   enable_frame_pacing = true;   ///< 渲染帧间隔调节
    int    opencv_num_threads  = 4;      ///< OpenCV 并行线程数
    int    inference_threads   = 0;      ///< 推理线程数（0=autoTune）
    bool   enable_gpu          = false;  ///< 启用 Vulkan GPU 推理

    // ---- 队列容量 ----
    int    audio_raw_queue_size = 30;
    int    mel_queue_size       = 60;
    int    video_raw_queue_size = 30;
    int    face_queue_size      = 30;
    int    infer_queue_size     = 30;
    int    output_queue_size    = 10;

    // ---- 超时 ----
    int    pop_timeout_ms       = 100;   ///< 队列弹出超时（ms）
    int    shutdown_timeout_ms  = 2000;  ///< 关闭超时（ms）
};

// ============================================================================
// 运行时指标
// ============================================================================

/// @brief SDK 运行时指标
struct SDKMetrics {
    int64_t total_frames_in   = 0;   ///< 输入帧数
    int64_t total_frames_out  = 0;   ///< 输出帧数
    int64_t frames_dropped    = 0;   ///< 丢弃帧数
    int64_t frames_skipped    = 0;   ///< 跳过帧数
    int64_t audio_packets_in  = 0;   ///< 音频包数
    int64_t video_packets_in  = 0;   ///< 视频包数
    int64_t inference_count   = 0;   ///< 推理次数
    double  avg_audio_process_ms = 0.0;  ///< 平均音频处理耗时（ms）
    double  avg_video_process_ms = 0.0;  ///< 平均视频处理耗时（ms）
    double  avg_inference_ms   = 0.0;    ///< 平均推理耗时（ms）
    double  avg_output_ms      = 0.0;    ///< 平均输出耗时（ms）
    double  actual_fps         = 0.0;    ///< 实际帧率
    double  drift_ms           = 0.0;    ///< 音视频偏移（ms）
};

// ============================================================================
// 帧回调
// ============================================================================

/// @brief 输出帧回调
/// @param frame  输出帧（BGR uint8）
/// @param pts_ms 呈现时间戳（毫秒）
using FrameCallback = std::function<void(const cv::Mat& frame, int64_t pts_ms)>;

// ============================================================================
// 主 SDK 类
// ============================================================================

/**
 * @brief 数字人 SDK 主入口
 *
 * 封装 Pipeline + 模型加载 + 音频/图像加载，提供统一生命周期管理。
 *
 * 典型用法（流式）：
 * @code
 *   digital_human::SDKConfig cfg;
 *   cfg.lipsync_model_dir = "models/Wav2Lip-SD-GAN-opt";
 *   cfg.face_model_dir    = "models/face";
 *
 *   digital_human::DigitalHumanSDK sdk;
 *   sdk.Init(cfg);
 *   sdk.Start();
 *
 *   sdk.PushAudio(pcm_data, pts_ms);
 *   sdk.PushVideo(frame, pts_ms);
 *
 *   cv::Mat out;
 *   int64_t out_pts;
 *   while (sdk.GetOutputFrame(out, out_pts, 100) == SDKError::OK) {
 *       // 渲染 out
 *   }
 *   sdk.Stop();
 * @endcode
 *
 * 典型用法（文件批处理）：
 * @code
 *   digital_human::DigitalHumanSDK sdk;
 *   sdk.Init(cfg);
 *   sdk.ProcessFile("audio.wav", "face.jpg", [](const cv::Mat& f, int64_t) {
 *       // 处理输出帧
 *   });
 * @endcode
 *
 * 线程安全：公有接口线程安全，内部状态通过原子变量与队列隔离。
 * 一次性对象语义：Stop() 后不可重启，需销毁重建。
 */
class DigitalHumanSDK {
public:
    DigitalHumanSDK();
    ~DigitalHumanSDK();

    DigitalHumanSDK(const DigitalHumanSDK&) = delete;
    DigitalHumanSDK& operator=(const DigitalHumanSDK&) = delete;
    DigitalHumanSDK(DigitalHumanSDK&&) = delete;
    DigitalHumanSDK& operator=(DigitalHumanSDK&&) = delete;

    // ========================================================================
    // 生命周期
    // ========================================================================

    /// @brief 初始化 SDK
    ///
    /// 创建内部 Pipeline，按 config 加载模型（若路径非空）。
    /// 不启动线程，需调用 Start()。
    ///
    /// @param config SDK 配置
    /// @return SDKError::OK 成功；其他见错误码
    SDKError Init(const SDKConfig& config);

    /// @brief 启动流水线
    SDKError Start();

    /// @brief 停止流水线并释放线程资源
    ///
    /// 幂等。停止后 SDK 不可重启（一次性对象语义）。
    SDKError Stop();

    /// @brief 暂停（停止消费输入，已入队数据保留）
    SDKError Pause();

    /// @brief 从暂停恢复
    SDKError Resume();

    // ========================================================================
    // 模型管理
    // ========================================================================

    /// @brief 加载 Wav2Lip 推理模型（Init 后可单独调用，覆盖 Init 时的加载）
    SDKError LoadLipSyncModel(const std::string& model_dir);

    /// @brief 加载人脸检测+关键点模型
    SDKError LoadFaceModel(const std::string& model_dir);

    /// @brief 启用/关闭 GPU 推理（须在 Start 前调用）
    SDKError EnableGPU(bool enable);

    /// @brief 设置推理线程数（须在 Start 前调用；0=autoTune）
    SDKError SetInferenceThreads(int n);

    // ========================================================================
    // 数据输入（流式）
    // ========================================================================

    /// @brief 输入一帧音频
    /// @param pcm     PCM float 样本（interleaved，[-1.0, 1.0]）
    /// @param pts_ms  呈现时间戳（毫秒）
    SDKError PushAudio(const std::vector<float>& pcm, int64_t pts_ms);

    /// @brief 输入一帧视频
    /// @param frame   BGR 图像
    /// @param pts_ms  呈现时间戳（毫秒）
    SDKError PushVideo(const cv::Mat& frame, int64_t pts_ms);

    /// @brief 标记音频流结束
    SDKError MarkAudioEOS();

    /// @brief 标记视频流结束
    SDKError MarkVideoEOS();

    // ========================================================================
    // 数据输出
    // ========================================================================

    /// @brief 获取一帧处理结果（阻塞）
    /// @param[out] frame       输出帧
    /// @param[out] pts_ms      输出时间戳
    /// @param      timeout_ms  超时（ms），-1 无限等待
    /// @return SDKError::OK 成功；TIMEOUT 超时；NOT_RUNNING 流水线已停止
    SDKError GetOutputFrame(cv::Mat& frame, int64_t& pts_ms, int timeout_ms = -1);

    // ========================================================================
    // 文件便捷接口
    // ========================================================================

    /// @brief 端到端文件处理：音频 + 单张图片 → 帧回调
    ///
    /// 内部完成：加载文件 → 启动 Pipeline → 按音频帧率推送数据
    /// → 拉取输出帧调用 callback → 停止 Pipeline。
    /// 图片作为静态人脸帧重复推送，适配 talking head 场景。
    ///
    /// @param audio_path  音频文件路径（wav/mp3 等 FFmpeg 支持格式）
    /// @param image_path  人脸图片路径
    /// @param callback    输出帧回调
    /// @return SDKError::OK 成功；其他见错误码
    SDKError ProcessFile(const std::string& audio_path,
                         const std::string& image_path,
                         const FrameCallback& callback);

    // ========================================================================
    // 查询
    // ========================================================================

    /// @brief 获取当前状态
    SDKState GetState() const;

    /// @brief 获取最后一次错误消息
    std::string GetLastError() const;

    /// @brief 获取运行时指标
    SDKMetrics GetMetrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace digital_human
