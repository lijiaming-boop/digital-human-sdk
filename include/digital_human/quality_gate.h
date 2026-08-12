#pragma once

#include <string>
#include <vector>

#include "digital_human/export.h"

namespace digital_human {

struct QualityGateThresholds {
    double min_fps_p95 = 24.0;
    double max_frame_latency_p95_ms = 80.0;
    double max_drop_rate = 0.02;
    double max_av_sync_p95_ms = 80.0;
    double max_av_duration_delta_ms = 100.0;
    double min_mouth_psnr_db = 28.0;
    double min_mouth_ssim = 0.90;
    bool require_monotonic_pts = true;
};

struct QualityGateSample {
    double fps_p95 = 0.0;
    double frame_latency_p95_ms = 0.0;
    double drop_rate = 0.0;
    double av_sync_p95_ms = 0.0;
    double av_duration_delta_ms = 0.0;
    double mouth_psnr_db = 0.0;
    double mouth_ssim = 0.0;
    bool pts_monotonic = true;
};

struct QualityGateResult {
    bool passed = false;
    std::vector<std::string> violations;
};

class DH_API QualityGate {
public:
    static QualityGateResult Evaluate(
        const QualityGateSample& sample,
        const QualityGateThresholds& thresholds = {});
};

}  // namespace digital_human
