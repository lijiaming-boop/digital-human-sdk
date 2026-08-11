#pragma once

#include <memory>
#include <cstdint>
#include <string>

namespace digital_human {
namespace core {

/// @brief 帧调度动作枚举
enum class FrameAction {
    DISPLAY,    ///< 正常显示当前帧
    DROP,       ///< 丢弃当前帧（视频滞后）
    DUPLICATE,  ///< 重复上一帧（视频超前，填补间隙）
    WAIT        ///< 等待（视频超前，需等待音频追赶）
};

/// @brief 帧调度配置
struct SchedulerConfig {
    double target_fps        = 30.0;  ///< 目标帧率
    int    max_pending_frames = 10;    ///< 最大待处理帧数
    double smoothing_factor  = 0.5;    ///< EMA 平滑因子 [0, 1]
    bool   enable_smoothing  = true;  ///< 是否启用帧间隔平滑
};

/// @brief 调度决策结果
struct ScheduleResult {
    FrameAction action          = FrameAction::DISPLAY; ///< 调度动作
    int         frame_id        = 0;   ///< 帧序号
    double      pts_ms          = 0.0; ///< 帧原始 PTS（毫秒）
    double      scheduled_pts_ms = 0.0;///< 调度后的 PTS（毫秒）
    double      wait_time_ms    = 0.0; ///< 需等待时间（仅 WAIT 时有效）
};

/// @brief 帧调度统计信息
struct FrameStats {
    int64_t total_frames        = 0;  ///< 总输入帧数
    int64_t frames_displayed    = 0;  ///< 实际显示帧数
    int64_t frames_dropped      = 0;  ///< 丢弃帧数
    int64_t frames_duplicated   = 0;  ///< 重复帧数
    double  actual_fps          = 0.0;///< 实际帧率
    double  avg_frame_interval_ms = 0.0; ///< 平均帧间隔（毫秒）

    /// @brief 返回统计信息的字符串描述
    std::string ToString() const;
};

/**
 * @brief 帧调度器
 *
 * 负责按目标帧率调度视频帧的显示，处理掉帧和重复帧场景，
 * 并通过 EMA 平滑策略稳定实际输出帧率。
 *
 * 调度逻辑：
 *   1. 计算期望 PTS = 上一显示帧 PTS + 帧间隔
 *   2. 比较实际 PTS 与期望 PTS
 *      - 实际 PTS 落后超过半帧间隔 → DROP
 *      - 实际 PTS 超前超过半帧间隔 → DUPLICATE
 *      - 否则 → DISPLAY
 *   3. EMA 平滑实际帧间隔，更新统计
 *
 * 采用 PIMPL（Pointer to Implementation）模式。
 */
class FrameScheduler {
public:
    FrameScheduler();
    ~FrameScheduler();
    FrameScheduler(const FrameScheduler&) = delete;
    FrameScheduler& operator=(const FrameScheduler&) = delete;
    FrameScheduler(FrameScheduler&&) noexcept;
    FrameScheduler& operator=(FrameScheduler&&) noexcept;

    // ==================== 初始化 ====================

    /// @brief 使用指定配置初始化调度器
    void Init(const SchedulerConfig& config);

    /// @brief 检查是否已初始化
    bool IsInitialized() const;

    // ==================== 调度 ====================

    /**
     * @brief 输入一帧并获取调度决策
     *
     * @param frame_id 帧序号（单调递增）
     * @param pts_ms   帧呈现时间戳（毫秒）
     * @return ScheduleResult 调度决策结果
     */
    ScheduleResult ScheduleFrame(int frame_id, double pts_ms);

    /**
     * @brief 通知调度器帧已显示（更新 EMA 平滑统计）
     *
     * @param actual_display_time_ms 实际显示时间（毫秒）
     */
    void OnFrameDisplayed(double actual_display_time_ms);

    // ==================== 统计 ====================

    /// @brief 获取帧调度统计信息
    FrameStats GetStats() const;

    // ==================== 控制 ====================

    /// @brief 重置调度器（清除所有状态和统计）
    void Reset();

    /// @brief 设置目标帧率
    void SetTargetFps(double fps);

    /// @brief 获取目标帧率
    double GetTargetFps() const;

    /// @brief 获取帧间隔（毫秒）
    double GetFrameIntervalMs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
