#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "digital_human/export.h"

namespace digital_human {
namespace core {

struct MediaClockConfig {
    int audio_sample_rate = 48000;
    double expected_video_interval_ms = 40.0;
    double drift_correction_threshold_ms = 80.0;
    double max_correction_step_ms = 5.0;
    double discontinuity_threshold_ms = 250.0;
    size_t jitter_window_size = 256;
};

struct MediaClockSnapshot {
    double audio_clock_ms = 0.0;
    double last_video_pts_ms = 0.0;
    double av_drift_ms = 0.0;
    double video_correction_ms = 0.0;
    double jitter_p50_ms = 0.0;
    double jitter_p95_ms = 0.0;
    double jitter_p99_ms = 0.0;
    double max_abs_drift_ms = 0.0;
    uint64_t observations = 0;
    uint64_t corrections = 0;
    uint64_t discontinuities = 0;
    bool pts_monotonic = true;
};

/// Audio-master media clock with bounded long-session video correction.
class DH_API MediaClock {
public:
    explicit MediaClock(const MediaClockConfig& config = {});
    ~MediaClock();
    MediaClock(MediaClock&&) noexcept;
    MediaClock& operator=(MediaClock&&) noexcept;
    MediaClock(const MediaClock&) = delete;
    MediaClock& operator=(const MediaClock&) = delete;

    bool Configure(const MediaClockConfig& config);
    void Reset();
    void AdvanceAudioFrames(int64_t frames);
    void ObserveVideoPts(double pts_ms);
    double CorrectVideoPts(double pts_ms) const;
    MediaClockSnapshot Snapshot() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
