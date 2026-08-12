#include "core/media_clock.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <mutex>
#include <vector>

namespace digital_human {
namespace core {
namespace {

double Percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<size_t>(std::ceil(
        quantile * static_cast<double>(values.size() - 1)));
    return values[std::min(index, values.size() - 1)];
}

bool ValidConfig(const MediaClockConfig& config) {
    return config.audio_sample_rate > 0
        && config.expected_video_interval_ms > 0.0
        && config.drift_correction_threshold_ms >= 0.0
        && config.max_correction_step_ms >= 0.0
        && config.discontinuity_threshold_ms > 0.0
        && config.jitter_window_size > 0;
}

}  // namespace

struct MediaClock::Impl {
    mutable std::mutex mutex;
    MediaClockConfig config;
    int64_t audio_frames = 0;
    double correction_ms = 0.0;
    double last_video_pts_ms = std::numeric_limits<double>::quiet_NaN();
    double last_drift_ms = 0.0;
    double max_abs_drift_ms = 0.0;
    std::deque<double> jitter;
    uint64_t observations = 0;
    uint64_t corrections = 0;
    uint64_t discontinuities = 0;
    bool pts_monotonic = true;

    double AudioClockMs() const {
        return static_cast<double>(audio_frames) * 1000.0
            / static_cast<double>(config.audio_sample_rate);
    }

    void ResetState() {
        audio_frames = 0;
        correction_ms = 0.0;
        last_video_pts_ms = std::numeric_limits<double>::quiet_NaN();
        last_drift_ms = 0.0;
        max_abs_drift_ms = 0.0;
        jitter.clear();
        observations = 0;
        corrections = 0;
        discontinuities = 0;
        pts_monotonic = true;
    }
};

MediaClock::MediaClock(const MediaClockConfig& config)
    : impl_(std::make_unique<Impl>()) {
    if (ValidConfig(config)) impl_->config = config;
}

MediaClock::~MediaClock() = default;
MediaClock::MediaClock(MediaClock&&) noexcept = default;
MediaClock& MediaClock::operator=(MediaClock&&) noexcept = default;

bool MediaClock::Configure(const MediaClockConfig& config) {
    if (!ValidConfig(config)) return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config = config;
    impl_->ResetState();
    return true;
}

void MediaClock::Reset() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->ResetState();
}

void MediaClock::AdvanceAudioFrames(int64_t frames) {
    if (frames <= 0) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->audio_frames += frames;
}

void MediaClock::ObserveVideoPts(double pts_ms) {
    if (!std::isfinite(pts_ms) || pts_ms < 0.0) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (std::isfinite(impl_->last_video_pts_ms)) {
        const double delta = pts_ms - impl_->last_video_pts_ms;
        if (delta <= 0.0) impl_->pts_monotonic = false;
        if (std::abs(delta) >= impl_->config.discontinuity_threshold_ms) {
            ++impl_->discontinuities;
        }
        impl_->jitter.push_back(std::abs(
            delta - impl_->config.expected_video_interval_ms));
        while (impl_->jitter.size() > impl_->config.jitter_window_size) {
            impl_->jitter.pop_front();
        }
    }

    impl_->last_video_pts_ms = pts_ms;
    impl_->last_drift_ms = pts_ms - impl_->correction_ms
        - impl_->AudioClockMs();
    impl_->max_abs_drift_ms = std::max(
        impl_->max_abs_drift_ms, std::abs(impl_->last_drift_ms));
    if (std::abs(impl_->last_drift_ms)
        >= impl_->config.drift_correction_threshold_ms) {
        const double step = std::clamp(
            impl_->last_drift_ms,
            -impl_->config.max_correction_step_ms,
            impl_->config.max_correction_step_ms);
        impl_->correction_ms += step;
        ++impl_->corrections;
    }
    ++impl_->observations;
}

double MediaClock::CorrectVideoPts(double pts_ms) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return pts_ms - impl_->correction_ms;
}

MediaClockSnapshot MediaClock::Snapshot() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    MediaClockSnapshot result;
    result.audio_clock_ms = impl_->AudioClockMs();
    result.last_video_pts_ms = std::isfinite(impl_->last_video_pts_ms)
        ? impl_->last_video_pts_ms : 0.0;
    result.av_drift_ms = impl_->last_drift_ms;
    result.video_correction_ms = impl_->correction_ms;
    result.max_abs_drift_ms = impl_->max_abs_drift_ms;
    result.observations = impl_->observations;
    result.corrections = impl_->corrections;
    result.discontinuities = impl_->discontinuities;
    result.pts_monotonic = impl_->pts_monotonic;
    const std::vector<double> jitter(impl_->jitter.begin(), impl_->jitter.end());
    result.jitter_p50_ms = Percentile(jitter, 0.50);
    result.jitter_p95_ms = Percentile(jitter, 0.95);
    result.jitter_p99_ms = Percentile(jitter, 0.99);
    return result;
}

}  // namespace core
}  // namespace digital_human
