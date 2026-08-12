#include "digital_human/quality_gate.h"

#include <cmath>

namespace digital_human {
namespace {

void RequireAtLeast(double value, double limit, const char* name,
                    std::vector<std::string>& violations) {
    if (!std::isfinite(value) || value < limit) {
        violations.emplace_back(std::string(name) + " below threshold");
    }
}

void RequireAtMost(double value, double limit, const char* name,
                   std::vector<std::string>& violations) {
    if (!std::isfinite(value) || value > limit || value < 0.0) {
        violations.emplace_back(std::string(name) + " exceeds threshold");
    }
}

}  // namespace

QualityGateResult QualityGate::Evaluate(
    const QualityGateSample& sample,
    const QualityGateThresholds& thresholds) {
    QualityGateResult result;
    RequireAtLeast(sample.fps_p95, thresholds.min_fps_p95,
                   "fps_p95", result.violations);
    RequireAtMost(sample.frame_latency_p95_ms,
                  thresholds.max_frame_latency_p95_ms,
                  "frame_latency_p95_ms", result.violations);
    RequireAtMost(sample.drop_rate, thresholds.max_drop_rate,
                  "drop_rate", result.violations);
    RequireAtMost(sample.av_sync_p95_ms, thresholds.max_av_sync_p95_ms,
                  "av_sync_p95_ms", result.violations);
    RequireAtMost(std::abs(sample.av_duration_delta_ms),
                  thresholds.max_av_duration_delta_ms,
                  "av_duration_delta_ms", result.violations);
    RequireAtLeast(sample.mouth_psnr_db, thresholds.min_mouth_psnr_db,
                   "mouth_psnr_db", result.violations);
    RequireAtLeast(sample.mouth_ssim, thresholds.min_mouth_ssim,
                   "mouth_ssim", result.violations);
    if (thresholds.require_monotonic_pts && !sample.pts_monotonic) {
        result.violations.emplace_back("PTS is not strictly monotonic");
    }
    result.passed = result.violations.empty();
    return result;
}

}  // namespace digital_human
