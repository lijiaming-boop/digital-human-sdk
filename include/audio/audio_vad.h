#pragma once

#include <vector>
#include <memory>
#include "digital_human/export.h"

namespace digital_human {
namespace audio {

class DH_API VoiceActivityDetector {
public:
    VoiceActivityDetector(float energyThresh = 0.01f,
                          float zcrMin = 0.0f,
                          float zcrMax = 0.5f,
                          int hangover = 3);
    ~VoiceActivityDetector();
    VoiceActivityDetector(const VoiceActivityDetector&) = delete;
    VoiceActivityDetector& operator=(const VoiceActivityDetector&) = delete;
    VoiceActivityDetector(VoiceActivityDetector&&) noexcept;
    VoiceActivityDetector& operator=(VoiceActivityDetector&&) noexcept;

    std::vector<std::vector<float>> filter(
        const std::vector<std::vector<float>>& frames) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
