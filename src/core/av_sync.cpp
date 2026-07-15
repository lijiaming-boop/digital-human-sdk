#include "core/av_sync.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace digital_human {
namespace core {

// ============================================================================
// Impl 结构体
// ============================================================================
struct AVSync::Impl {
    SyncConfig config;                      ///< 同步配置
    bool       initialized   = false;       ///< 是否已初始化
    double     audio_clock_ms = 0.0;        ///< 音频时钟（毫秒），单调递增

    // ========================================================================
    // 同步判定
    // ========================================================================

    /**
     * @brief 核心同步判定逻辑
     *
     * drift = video_pts - audio_clock
     *
     * |drift| ≥ max_drift  → SEVERE_OFFSET, drop frame
     * drift > threshold    → VIDEO_AHEAD, wait
     * drift < -threshold   → VIDEO_BEHIND, drop frame
     * 否则                 → SYNCED
     */
    SyncResult doGetSyncStatus(double video_pts_ms) const {
        SyncResult result;
        result.audio_clock_ms = audio_clock_ms;
        result.video_pts_ms   = video_pts_ms;
        result.drift_ms       = video_pts_ms - audio_clock_ms;
        result.wait_time_ms   = 0.0;
        result.should_drop_frame = false;
        result.status = SyncStatus::SYNCED;

        double abs_drift = std::abs(result.drift_ms);
        double threshold = config.sync_threshold_ms;
        double max_drift = config.max_drift_ms;

        if (abs_drift >= max_drift) {
            // 严重偏移：丢弃帧
            result.status = SyncStatus::SEVERE_OFFSET;
            result.should_drop_frame = true;
        } else if (result.drift_ms > threshold) {
            // 视频超前：等待
            result.status = SyncStatus::VIDEO_AHEAD;
            result.wait_time_ms = result.drift_ms;
            result.should_drop_frame = false;
        } else if (result.drift_ms < -threshold) {
            // 视频滞后：丢弃帧
            result.status = SyncStatus::VIDEO_BEHIND;
            result.should_drop_frame = true;
        } else {
            // 音画同步
            result.status = SyncStatus::SYNCED;
            result.should_drop_frame = false;
        }

        return result;
    }
};

// ============================================================================
// SyncResult 成员实现
// ============================================================================

std::string SyncResult::StatusString() const {
    switch (status) {
        case SyncStatus::SYNCED:       return "SYNCED";
        case SyncStatus::VIDEO_AHEAD:  return "VIDEO_AHEAD";
        case SyncStatus::VIDEO_BEHIND: return "VIDEO_BEHIND";
        case SyncStatus::SEVERE_OFFSET: return "SEVERE_OFFSET";
        default:                       return "UNKNOWN";
    }
}

// ============================================================================
// AVSync 公有接口实现
// ============================================================================

AVSync::AVSync()
    : impl_(std::make_unique<Impl>()) {}

AVSync::~AVSync() = default;

AVSync::AVSync(AVSync&&) noexcept = default;

AVSync& AVSync::operator=(AVSync&&) noexcept = default;

// ==================== 初始化 ====================

void AVSync::Init(const SyncConfig& config) {
    impl_->config       = config;
    impl_->audio_clock_ms = 0.0;
    impl_->initialized  = true;

    if (impl_->config.audio_sample_rate <= 0) {
        std::cerr << "[AVSync] Init: 无效的 audio_sample_rate ("
                  << impl_->config.audio_sample_rate << ")，使用默认 16000" << std::endl;
        impl_->config.audio_sample_rate = 16000;
    }
    if (impl_->config.sync_threshold_ms <= 0.0) {
        std::cerr << "[AVSync] Init: sync_threshold_ms <= 0，使用默认 30ms" << std::endl;
        impl_->config.sync_threshold_ms = 30.0;
    }
    if (impl_->config.max_drift_ms <= 0.0) {
        std::cerr << "[AVSync] Init: max_drift_ms <= 0，使用默认 100ms" << std::endl;
        impl_->config.max_drift_ms = 100.0;
    }
}

bool AVSync::IsInitialized() const {
    return impl_->initialized;
}

// ==================== 音频时钟 ====================

void AVSync::UpdateAudioClock(int64_t samples_consumed) {
    if (!impl_->initialized) {
        std::cerr << "[AVSync] UpdateAudioClock: 未初始化" << std::endl;
        return;
    }
    if (samples_consumed < 0) {
        std::cerr << "[AVSync] UpdateAudioClock: samples_consumed 不能为负" << std::endl;
        return;
    }

    double delta_ms = static_cast<double>(samples_consumed)
                    / impl_->config.audio_sample_rate * 1000.0;
    impl_->audio_clock_ms += delta_ms;
}

void AVSync::SetAudioClockMs(double ms) {
    if (!impl_->initialized) {
        std::cerr << "[AVSync] SetAudioClockMs: 未初始化" << std::endl;
        return;
    }
    impl_->audio_clock_ms = ms;
}

double AVSync::GetAudioClockMs() const {
    return impl_->audio_clock_ms;
}

// ==================== 同步判定 ====================

SyncResult AVSync::GetSyncStatus(double video_pts_ms) const {
    if (!impl_->initialized) {
        std::cerr << "[AVSync] GetSyncStatus: 未初始化" << std::endl;
        return SyncResult();
    }
    return impl_->doGetSyncStatus(video_pts_ms);
}

SyncResult AVSync::Sync(int64_t samples_consumed, double video_pts_ms) {
    UpdateAudioClock(samples_consumed);
    return GetSyncStatus(video_pts_ms);
}

// ==================== 控制 ====================

void AVSync::Reset() {
    impl_->audio_clock_ms = 0.0;
}

void AVSync::SetSyncThresholdMs(double ms) {
    impl_->config.sync_threshold_ms = std::max(ms, 1.0);
}

void AVSync::SetMaxDriftMs(double ms) {
    impl_->config.max_drift_ms = std::max(ms, 1.0);
}

SyncConfig AVSync::GetConfig() const {
    return impl_->config;
}

}  // namespace core
}  // namespace digital_human
