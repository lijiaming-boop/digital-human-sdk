#include "core/frame_scheduler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <limits>

namespace digital_human {
namespace core {

// ============================================================================
// Impl 结构体
// ============================================================================
struct FrameScheduler::Impl {
    SchedulerConfig config;                 ///< 调度配置
    bool initialized = false;               ///< 是否已初始化

    // ---- 调度状态 ----
    double last_displayed_pts_ms = 0.0;     ///< 上一显示帧的 PTS（毫秒）
    int    last_displayed_frame_id = -1;    ///< 上一显示帧的序号
    bool   has_first_frame = false;         ///< 是否已显示第一帧

    // ---- 统计信息 ----
    int64_t total_frames      = 0;          ///< 总输入帧数
    int64_t frames_displayed  = 0;          ///< 实际显示帧数
    int64_t frames_dropped    = 0;          ///< 丢弃帧数
    int64_t frames_duplicated = 0;          ///< 重复帧数
    double  avg_interval_ms   = 0.0;        ///< EMA 平均帧间隔（毫秒）
    double  last_actual_time_ms = 0.0;      ///< 上一帧的实际显示时间
    bool    has_first_display = false;      ///< 是否已有首次显示时间记录

    // ========================================================================
    // 调度核心逻辑
    // ========================================================================

    /**
     * @brief 核心调度判定
     *
     * expected_pts = last_displayed_pts + frame_interval
     * half_interval = frame_interval / 2
     *
     * pts < expected_pts - half_interval → DROP（滞后太多）
     * pts > expected_pts + half_interval → DUPLICATE（超前太多）
     * 否则 → DISPLAY
     */
    ScheduleResult doSchedule(int frame_id, double pts_ms) {
        total_frames++;
        ScheduleResult result;
        result.frame_id = frame_id;
        result.pts_ms   = pts_ms;

        double frame_interval = 1000.0 / config.target_fps;
        double half_interval  = frame_interval * 0.5;

        if (!has_first_frame) {
            // 第一帧始终显示
            result.action = FrameAction::DISPLAY;
            result.scheduled_pts_ms = pts_ms;
            applyDisplay(frame_id, pts_ms);
            return result;
        }

        double expected_pts = last_displayed_pts_ms + frame_interval;
        double diff = pts_ms - expected_pts;

        if (diff < -half_interval) {
            // 视频滞后超过半帧：丢弃
            result.action = FrameAction::DROP;
            result.scheduled_pts_ms = last_displayed_pts_ms;
            frames_dropped++;
        } else if (diff > half_interval) {
            // 视频超前超过半帧：重复上一帧
            result.action = FrameAction::DUPLICATE;
            result.scheduled_pts_ms = last_displayed_pts_ms + frame_interval;
            frames_duplicated++;
        } else {
            // 正常显示
            result.action = FrameAction::DISPLAY;
            result.scheduled_pts_ms = pts_ms;
            applyDisplay(frame_id, pts_ms);
        }

        return result;
    }

    /// @brief 更新显示状态
    void applyDisplay(int frame_id, double pts_ms) {
        last_displayed_pts_ms   = pts_ms;
        last_displayed_frame_id = frame_id;
        has_first_frame = true;
        frames_displayed++;
    }

    /// @brief 更新 EMA 平滑统计
    void applySmoothing(double actual_time_ms) {
        if (!has_first_display) {
            last_actual_time_ms = actual_time_ms;
            avg_interval_ms     = 1000.0 / config.target_fps;
            has_first_display   = true;
            return;
        }

        double interval = actual_time_ms - last_actual_time_ms;
        last_actual_time_ms = actual_time_ms;

        if (interval <= 0.0) return;

        if (config.enable_smoothing) {
            double alpha = std::clamp(config.smoothing_factor, 0.01, 0.99);
            avg_interval_ms = alpha * interval + (1.0 - alpha) * avg_interval_ms;
        } else {
            avg_interval_ms = interval;
        }
    }
};

// ============================================================================
// FrameStats 实现
// ============================================================================

std::string FrameStats::ToString() const {
    std::ostringstream oss;
    oss << "FrameStats { total=" << total_frames
        << " displayed=" << frames_displayed
        << " dropped=" << frames_dropped
        << " duplicated=" << frames_duplicated
        << " actual_fps=" << actual_fps
        << " avg_interval=" << avg_frame_interval_ms << "ms }";
    return oss.str();
}

// ============================================================================
// FrameScheduler 公有接口实现
// ============================================================================

FrameScheduler::FrameScheduler()
    : impl_(std::make_unique<Impl>()) {}

FrameScheduler::~FrameScheduler() = default;

FrameScheduler::FrameScheduler(FrameScheduler&&) noexcept = default;

FrameScheduler& FrameScheduler::operator=(FrameScheduler&&) noexcept = default;

// ==================== 初始化 ====================

void FrameScheduler::Init(const SchedulerConfig& config) {
    impl_->config = config;
    if (impl_->config.target_fps <= 0.0) {
        std::cerr << "[FrameScheduler] Init: target_fps <= 0，使用默认 30fps" << std::endl;
        impl_->config.target_fps = 30.0;
    }
    if (impl_->config.max_pending_frames <= 0) {
        impl_->config.max_pending_frames = 10;
    }
    impl_->config.smoothing_factor = std::clamp(config.smoothing_factor, 0.01, 0.99);
    impl_->avg_interval_ms = 1000.0 / impl_->config.target_fps;
    impl_->initialized = true;
}

bool FrameScheduler::IsInitialized() const {
    return impl_->initialized;
}

// ==================== 调度 ====================

ScheduleResult FrameScheduler::ScheduleFrame(int frame_id, double pts_ms) {
    if (!impl_->initialized) {
        std::cerr << "[FrameScheduler] ScheduleFrame: 未初始化" << std::endl;
        ScheduleResult err;
        err.action = FrameAction::DROP;
        return err;
    }
    return impl_->doSchedule(frame_id, pts_ms);
}

void FrameScheduler::OnFrameDisplayed(double actual_display_time_ms) {
    if (!impl_->initialized) return;
    impl_->applySmoothing(actual_display_time_ms);
}

// ==================== 统计 ====================

FrameStats FrameScheduler::GetStats() const {
    FrameStats stats;
    stats.total_frames         = impl_->total_frames;
    stats.frames_displayed     = impl_->frames_displayed;
    stats.frames_dropped       = impl_->frames_dropped;
    stats.frames_duplicated    = impl_->frames_duplicated;
    stats.avg_frame_interval_ms = impl_->avg_interval_ms;
    stats.actual_fps = (impl_->avg_interval_ms > 0.0)
                       ? 1000.0 / impl_->avg_interval_ms
                       : 0.0;
    return stats;
}

// ==================== 控制 ====================

void FrameScheduler::Reset() {
    impl_->last_displayed_pts_ms   = 0.0;
    impl_->last_displayed_frame_id = -1;
    impl_->has_first_frame         = false;
    impl_->total_frames            = 0;
    impl_->frames_displayed        = 0;
    impl_->frames_dropped          = 0;
    impl_->frames_duplicated       = 0;
    impl_->avg_interval_ms         = 1000.0 / impl_->config.target_fps;
    impl_->last_actual_time_ms     = 0.0;
    impl_->has_first_display       = false;
}

void FrameScheduler::SetTargetFps(double fps) {
    if (fps <= 0.0) {
        std::cerr << "[FrameScheduler] SetTargetFps: fps <= 0" << std::endl;
        return;
    }
    impl_->config.target_fps = fps;
}

double FrameScheduler::GetTargetFps() const {
    return impl_->config.target_fps;
}

double FrameScheduler::GetFrameIntervalMs() const {
    return 1000.0 / impl_->config.target_fps;
}

}  // namespace core
}  // namespace digital_human
