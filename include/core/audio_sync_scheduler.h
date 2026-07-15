#pragma once

#include <memory>
#include <cstdint>
#include <vector>
#include <string>

#include "core/frame_scheduler.h"
#include "core/av_sync.h"

namespace digital_human {
namespace core {

/// @brief 音频同步调度器配置
struct AudioSyncConfig {
    // ---- 音频参数 ----
    int audio_sample_rate        = 48000;   ///< 采样率（Hz）
    int audio_channels           = 2;        ///< 声道数
    int audio_frames_per_buffer  = 512;      ///< PortAudio 每缓冲区帧数
    int audio_buffer_capacity_ms = 500;      ///< 音频缓冲区容量（毫秒）

    // ---- 视频参数 ----
    double target_fps            = 25.0;     ///< 目标帧率

    // ---- 同步参数 ----
    double sync_threshold_ms     = 30.0;     ///< 同步阈值（毫秒）
    double max_drift_ms          = 100.0;    ///< 最大允许漂移（毫秒）
};

/// @brief 播放状态
enum class PlaybackState {
    STOPPED,    ///< 已停止
    PLAYING,    ///< 播放中
    PAUSED,     ///< 已暂停
};

/**
 * @brief 音频驱动的音视频同步调度器
 *
 * 采用音频主时钟（Audio Master Clock）策略，以 PortAudio 的音频输出位置
 * 作为时间基准，驱动视频帧的调度和同步。
 *
 * 工作流程：
 *   1. 加载 PCM 音频数据
 *   2. 调用 Play() 启动音频播放（PortAudio 输出音频到扬声器）
 *   3. 每帧调用 ScheduleFrame(frameId, videoPtsMs) 获取调度决策
 *      - 调度器内部查询 AudioPlayer 的播放位置作为音频时钟
 *      - 与视频帧 PTS 比较，计算漂移量 drift
 *      - |drift| ≥ max_drift  → DROP（严重偏移）
 *      - drift > threshold    → DUPLICATE（视频超前，重复上一帧）
 *      - drift < -threshold   → DROP（视频滞后，丢弃帧）
 *      - 否则                 → DISPLAY
 *   4. 调用 OnFrameDisplayed() 更新帧统计
 *   5. 调用 Stop() 停止播放
 *
 * 组合了 AudioPlayer、AVSync 和 FrameScheduler 的功能。
 */
class AudioSyncScheduler {
public:
    AudioSyncScheduler();
    ~AudioSyncScheduler();
    AudioSyncScheduler(const AudioSyncScheduler&) = delete;
    AudioSyncScheduler& operator=(const AudioSyncScheduler&) = delete;
    AudioSyncScheduler(AudioSyncScheduler&&) noexcept;
    AudioSyncScheduler& operator=(AudioSyncScheduler&&) noexcept;

    // ==================== 初始化 ====================

    /**
     * @brief 使用指定配置初始化同步调度器
     *
     * 内部初始化 AudioPlayer、AVSync 和 FrameScheduler。
     *
     * @param config 同步调度器配置
     * @return true  成功
     * @return false 失败（例如无音频设备）
     */
    bool Init(const AudioSyncConfig& config);

    /// @brief 检查是否已初始化
    bool IsInitialized() const;

    /// @brief 销毁调度器，释放所有资源
    void Destroy();

    // ==================== 音频数据加载 ====================

    /**
     * @brief 加载 PCM float 音频数据
     *
     * @param samples    音频样本数据（interleaved，范围 [-1.0, 1.0]）
     * @param numSamples 样本总数
     * @param channels   声道数
     * @return true  成功
     * @return false 失败
     */
    bool LoadAudio(const float* samples, int numSamples, int channels);

    /// @brief 加载 PCM float 音频数据（vector 版本）
    bool LoadAudio(const std::vector<float>& samples, int channels);

    // ==================== 播放控制 ====================

    /// @brief 开始播放（启动音频流）
    bool Play();

    /// @brief 暂停播放（暂停音频流，冻结调度）
    bool Pause();

    /// @brief 从暂停状态恢复播放
    bool Resume();

    /// @brief 停止播放（重置所有状态）
    bool Stop();

    /// @brief 获取当前播放状态
    PlaybackState GetPlaybackState() const;

    // ==================== 视频帧调度 ====================

    /**
     * @brief 调度下一帧
     *
     * 内部执行：
     *   1. 从 AudioPlayer 获取已消耗音频帧数 → 音频时钟
     *   2. 用 AVSync 比较音频时钟与视频帧 PTS
     *   3. 用 FrameScheduler 做出最终调度决策
     *
     * @param frame_id    帧序号（单调递增）
     * @param video_pts_ms 视频帧呈现时间戳（毫秒）
     * @return ScheduleResult 调度决策（含 action、wait_time 等）
     */
    ScheduleResult ScheduleFrame(int frame_id, double video_pts_ms);

    /**
     * @brief 通知调度器帧已显示
     *
     * 更新 FrameScheduler 的 EMA 平滑统计。
     *
     * @param actual_display_time_ms 实际显示时间（毫秒）
     */
    void OnFrameDisplayed(double actual_display_time_ms);

    // ==================== 同步信息查询 ====================

    /**
     * @brief 获取当前音视频偏移量（毫秒）
     *
     * drift = video_pts - audio_clock
     * 正值表示视频超前于音频，负值表示视频滞后。
     *
     * @return double 偏移量（毫秒）
     */
    double GetDriftMs() const;

    /// @brief 获取当前音频时钟（毫秒）
    double GetAudioClockMs() const;

    /// @brief 获取当前同步状态
    SyncStatus GetSyncStatus() const;

    /// @brief 获取帧调度统计信息
    FrameStats GetFrameStats() const;

    /// @brief 获取详细统计信息的字符串描述
    std::string GetStatsString() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
