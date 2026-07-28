#pragma once

#include <memory>
#include <string>
#include <vector>
#include <atomic>

#include "core/packet.h"
#include "core/thread_safe_queue.h"
#include "core/thread_base.h"
#include "core/frame_scheduler.h"
#include "core/av_sync.h"

namespace digital_human {
namespace core {

// ============================================================================
// 配置结构
// ============================================================================

/// @brief Pipeline 配置
struct PipelineConfig {
    // ---- 音频参数 ----
    int audio_sample_rate         = 16000;   ///< 音频采样率（特征提取用）
    int audio_channels            = 1;        ///< 音频声道数
    int audio_frame_size          = 400;      ///< 音频帧大小
    int audio_hop_size            = 160;      ///< 音频帧移

    // ---- 视频参数 ----
    double target_fps             = 30.0;     ///< 目标帧率
    int    face_size              = 96;       ///< 对齐人脸尺寸

    // ---- 同步参数 ----
    double sync_threshold_ms      = 30.0;     ///< 同步阈值
    double max_drift_ms           = 100.0;    ///< 最大允许漂移
    double av_match_threshold_ms  = 40.0;     ///< 音视频匹配阈值

    // ---- 口型驱动参数 ----
    int    mel_window_frames      = 16;       ///< Wav2Lip mel 时序窗长度（mel 帧数，syncnet_mel_step_size）
    int    mel_context_frames     = 300;      ///< 归一化滚动上下文长度（mel 帧数，~3s）

    // ---- 性能开关 ----
    bool   enable_frame_pacing    = true;     ///< 渲染帧间隔调节（离线批处理建议关闭以跑满吞吐）
    int    opencv_num_threads     = 4;        ///< OpenCV 并行线程数（0=不设置；小图操作线程过多会因同步开销拖慢整体并与推理争抢核）

    // ---- 队列容量 ----
    int    audio_raw_queue_size   = 30;       ///< 音频原始数据队列容量
    int    mel_queue_size         = 60;       ///< Mel 特征队列容量
    int    video_raw_queue_size   = 30;       ///< 视频原始帧队列容量
    int    face_queue_size        = 30;       ///< 处理后人脸队列容量
    int    infer_queue_size       = 30;       ///< 推理输出队列容量
    int    output_queue_size      = 10;       ///< 最终输出队列容量

    // ---- 超时 ----
    int    pop_timeout_ms         = 100;      ///< 队列弹出超时（毫秒）
    int    shutdown_timeout_ms    = 2000;     ///< 关闭超时（毫秒）
};

// ============================================================================
// Pipeline 指标
// ============================================================================

/// @brief Pipeline 运行时指标
struct PipelineMetrics {
    int64_t total_frames_in       = 0;
    int64_t total_frames_out      = 0;
    int64_t frames_dropped        = 0;
    int64_t frames_skipped        = 0;
    int64_t audio_packets_in      = 0;
    int64_t video_packets_in      = 0;
    int64_t inference_count       = 0;
    double  avg_audio_process_ms  = 0.0;
    double  avg_video_process_ms  = 0.0;
    double  avg_inference_ms      = 0.0;
    double  avg_output_ms         = 0.0;
    double  actual_fps            = 0.0;

    std::string ToString() const;
};

// ============================================================================
// Pipeline 主类
// ============================================================================

/**
 * @brief 多线程媒体处理管道
 *
 * 编排音频/视频/推理/渲染 7 个线程的流水线处理。
 *
 * 架构：
 * ```
 * AudioProducer ─► AudioProcessor ─► ┌─ SyncPoint ─┐
 *                                     │ (AV匹配)     │
 * VideoProducer ─► VideoProcessor ──►└──────────────┘
 *                                              │
 *                                              ▼
 *                                        InferenceWorker
 *                                              │
 *                                              ▼
 *                                        OutputProcessor
 *                                              │
 *                                              ▼
 *                                          RenderThread
 * ```
 *
 * 线程安全：Pipeline 本身线程安全，所有内部状态通过队列和原子变量隔离。
 */
class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;

    // ========================================================================
    // 生命周期
    // ========================================================================

    /**
     * @brief 初始化 Pipeline
     *
     * 创建所有内部队列和辅助模块（AVSync, FrameScheduler），
     * 但不启动线程。
     *
     * @param config Pipeline 配置
     * @return true  初始化成功
     * @return false 配置无效
     */
    bool Init(const PipelineConfig& config);

    /**
     * @brief 启动所有线程
     *
     * 按从下游到上游的顺序启动线程：
     * RenderThread → OutputProcessor → InferenceWorker
     * → VideoProcessor → AudioProcessor → Producers
     *
     * @return true  启动成功
     * @return false 未初始化或已启动
     */
    bool Start();

    /**
     * @brief 停止所有线程并清理资源
     *
     * 按从上游到下游的顺序停止线程，超时后强制终止。
     * 可在任意时刻调用，幂等。
     */
    void Stop();

    /// @brief 检查 Pipeline 是否在运行
    bool IsRunning() const;

    // ========================================================================
    // 数据输入
    // ========================================================================

    /**
     * @brief 输入一帧音频数据
     *
     * @param pcm_data  PCM float 样本
     * @param pts_ms    时间戳（毫秒）
     * @return true     入队成功
     */
    bool PushAudio(const std::vector<float>& pcm_data, int64_t pts_ms);

    /**
     * @brief 输入一帧视频数据
     *
     * @param frame    BGR 图像
     * @param pts_ms   时间戳（毫秒）
     * @return true    入队成功
     */
    bool PushVideo(const cv::Mat& frame, int64_t pts_ms);

    /**
     * @brief 标记音频流结束
     */
    void MarkAudioEOS();

    /**
     * @brief 标记视频流结束
     */
    void MarkVideoEOS();

    // ========================================================================
    // 数据输出
    // ========================================================================

    /**
     * @brief 获取一帧处理结果（阻塞）
     *
     * @param[out] frame      输出帧
     * @param timeout_ms      超时时间（毫秒），-1 无限等待
     * @return true           成功获取一帧
     * @return false          超时或流水线结束
     */
    bool GetOutputFrame(OutputFramePacket& frame, int timeout_ms = -1);

    // ========================================================================
    // 控制
    // ========================================================================

    /// @brief 暂停处理
    void Pause();

    /// @brief 恢复处理
    void Resume();

    /// @brief 检查是否已暂停
    bool IsPaused() const;

    // ========================================================================
    // 查询
    // ========================================================================

    /// @brief 获取运行时指标
    PipelineMetrics GetMetrics() const;

    /// @brief 获取帧调度统计
    FrameStats GetFrameStats() const;

    /// @brief 获取同步状态
    SyncStatus GetSyncStatus() const;

    /// @brief 获取音视频偏移
    double GetDriftMs() const;

    /// @brief 获取音频时钟
    double GetAudioClockMs() const;

    /// @brief 设置 dlib 人脸关键点模型路径（拟合图片/视频前必须调用）
    /// @param path 模型文件路径（shape_predictor_68_face_landmarks.dat）
    void SetLandmarkModelPath(const std::string& path);

    /// @brief 初始化内置 ModelInferencer（Wav2Lip 推理模型）
    /// @param model_dir 模型目录（含 Wav2Lip-SD-GAN-opt.param/.bin）
    /// @return true 初始化成功
    bool InitModelInferencer(const std::string& model_dir);

    /// @brief 设置推理线程数（0=保持 autoTune 结果）
    ///
    /// 注意：autoTune 在空闲环境下测得最优线程数，但在流水线并发负载下，
    /// 推理线程与其他阶段线程在物理核/SMT 上相互争抢，最优值往往更小。
    /// 建议在目标部署场景下实测后设定。
    /// Call before Start(). If already initialized, the model is reloaded so
    /// ncnn's load-time convolution pipeline uses the requested thread count.
    void SetInferenceThreads(int n);


private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
