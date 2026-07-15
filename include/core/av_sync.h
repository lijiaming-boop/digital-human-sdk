#pragma once

#include <memory>
#include <string>

namespace digital_human {
namespace core {

/// @brief 音画同步状态枚举
enum class SyncStatus {
    SYNCED,           ///< 音画同步：|drift| ≤ threshold
    VIDEO_AHEAD,      ///< 视频超前：video_pts > audio_clock + threshold
    VIDEO_BEHIND,     ///< 视频滞后：video_pts < audio_clock - threshold
    SEVERE_OFFSET     ///< 严重偏移：|drift| ≥ max_drift
};

/// @brief 音画同步配置
struct SyncConfig {
    int    audio_sample_rate = 16000;     ///< 音频采样率（Hz）
    double sync_threshold_ms = 30.0;      ///< 同步阈值（毫秒），超出此值视为不同步
    double max_drift_ms      = 100.0;     ///< 最大允许漂移（毫秒），超此值视为严重偏移
};

/// @brief 音画同步判定结果
struct SyncResult {
    SyncStatus status        = SyncStatus::SYNCED;   ///< 同步状态
    double     audio_clock_ms = 0.0;                  ///< 当前音频时钟（毫秒）
    double     video_pts_ms   = 0.0;                  ///< 当前视频帧 PTS（毫秒）
    double     drift_ms       = 0.0;                  ///< 偏移量 = video_pts - audio_clock（毫秒）
    double     wait_time_ms   = 0.0;                  ///< 建议等待时间（仅 VIDEO_AHEAD 时有效）
    bool       should_drop_frame = false;             ///< 是否应丢弃当前帧

    /// @brief 返回同步状态的字符串描述
    std::string StatusString() const;
};

/**
 * @brief 音视频同步器
 *
 * 采用音频基准同步策略（Audio Master Clock），以音频时钟为基准，
 * 视频帧通过时间戳（PTS）与之对齐，通过比较 drift 值判定同步状态。
 *
 * 同步判定逻辑：
 *   drift = video_pts - audio_clock
 *   - |drift| ≥ max_drift   → SEVERE_OFFSET，应丢弃帧
 *   - drift > threshold      → VIDEO_AHEAD，视频超前
 *   - drift < -threshold     → VIDEO_BEHIND，视频滞后，应丢弃帧
 *   - 否则                   → SYNCED
 *
 * 采用 PIMPL（Pointer to Implementation）模式。
 */
class AVSync {
public:
    AVSync();
    ~AVSync();
    AVSync(const AVSync&) = delete;
    AVSync& operator=(const AVSync&) = delete;
    AVSync(AVSync&&) noexcept;
    AVSync& operator=(AVSync&&) noexcept;

    // ==================== 初始化 ====================

    /// @brief 使用指定配置初始化同步器
    void Init(const SyncConfig& config);

    /// @brief 检查是否已初始化
    bool IsInitialized() const;

    // ==================== 音频时钟 ====================

    /**
     * @brief 根据已消耗的音频样本数更新音频时钟
     *
     * 音频时钟 = samples_consumed / sample_rate * 1000（毫秒）
     * 多次调用会累加样本数，clock 单调递增。
     *
     * @param samples_consumed 本次消耗的音频样本数
     */
    void UpdateAudioClock(int64_t samples_consumed);

    /// @brief 直接设置音频时钟（毫秒），用于跳转/seek 场景
    void SetAudioClockMs(double ms);

    /// @brief 获取当前音频时钟（毫秒）
    double GetAudioClockMs() const;

    // ==================== 同步判定 ====================

    /**
     * @brief 根据视频帧 PTS 获取同步判定结果
     *
     * @param video_pts_ms 视频帧的呈现时间戳（毫秒）
     * @return SyncResult  同步判定结果
     */
    SyncResult GetSyncStatus(double video_pts_ms) const;

    /**
     * @brief 综合同步操作：更新音频时钟 + 同步判定
     *
     * 等效于依次调用 UpdateAudioClock 和 GetSyncStatus。
     *
     * @param samples_consumed 本次消耗的音频样本数
     * @param video_pts_ms     视频帧 PTS（毫秒）
     * @return SyncResult      同步判定结果
     */
    SyncResult Sync(int64_t samples_consumed, double video_pts_ms);

    // ==================== 控制 ====================

    /// @brief 重置同步器（音频时钟归零，清除所有状态）
    void Reset();

    /// @brief 设置同步阈值（毫秒）
    void SetSyncThresholdMs(double ms);

    /// @brief 设置最大允许漂移（毫秒）
    void SetMaxDriftMs(double ms);

    /// @brief 获取当前同步配置
    SyncConfig GetConfig() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
