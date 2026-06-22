#include "audio/audio_rms_normalize.h"

#include <cmath>

namespace digital_human {
namespace audio {

struct RMSNormalize::Impl {
    float targetRMS;

    std::vector<float> process(const std::vector<float>& pcm) const {
        if (pcm.empty()) return {};

        float sumSq = 0.0f;
        for (float x : pcm) {
            sumSq += x * x;
        }

        float currentRMS = std::sqrt(sumSq / pcm.size());
        if (currentRMS < 1e-10f) {
            return pcm;  // silence, skip normalization
        }

        float scale = targetRMS / currentRMS;
        std::vector<float> result(pcm.size());
        for (size_t i = 0; i < pcm.size(); i++) {
            result[i] = pcm[i] * scale;
        }
        return result;
    }
};

RMSNormalize::RMSNormalize(float targetRMS)
    : impl_(std::make_unique<Impl>()) {
    impl_->targetRMS = targetRMS;
}

RMSNormalize::~RMSNormalize() = default;

RMSNormalize::RMSNormalize(RMSNormalize&&) noexcept = default;

RMSNormalize& RMSNormalize::operator=(RMSNormalize&&) noexcept = default;

std::vector<float> RMSNormalize::process(const std::vector<float>& pcm) const {
    return impl_->process(pcm);
}

}  // namespace audio
}  // namespace digital_human
