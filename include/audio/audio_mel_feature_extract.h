#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace digital_human {
namespace audio {

struct MelConfig {
    int nFFT = 512;
    int nMels = 80;
    int sampleRate = 16000;
    float fMin = 0.0f;
    float fMax = 8000.0f;
};

class MelFeatureExtract {
public:
    MelFeatureExtract();
    ~MelFeatureExtract();
    MelFeatureExtract(const MelFeatureExtract&) = delete;
    MelFeatureExtract& operator=(const MelFeatureExtract&) = delete;
    MelFeatureExtract(MelFeatureExtract&&) noexcept;
    MelFeatureExtract& operator=(MelFeatureExtract&&) noexcept;

    cv::Mat extract(const std::vector<std::vector<float>>& frames,
                    const MelConfig& config = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
