#include "audio/audio_preemphasis.h"

#include <algorithm>

namespace digital_human {
namespace audio {

struct PreEmphasis::Impl {
    float alpha;

    std::vector<float> process(const std::vector<float>& pcm) const {
        if (pcm.empty()) {
            return {};
        }

        std::vector<float> result(pcm.size());
        result[0] = pcm[0];
        for (size_t i = 1; i < pcm.size(); i++) {
            result[i] = pcm[i] - alpha * pcm[i - 1];
        }
        return result;
    }
};

PreEmphasis::PreEmphasis(float alpha)
    : impl_(std::make_unique<Impl>()) {
    impl_->alpha = std::clamp(alpha, 0.0f, 1.0f);
}

PreEmphasis::~PreEmphasis() = default;

PreEmphasis::PreEmphasis(PreEmphasis&&) noexcept = default;

PreEmphasis& PreEmphasis::operator=(PreEmphasis&&) noexcept = default;

std::vector<float> PreEmphasis::process(const std::vector<float>& pcm) const {
    return impl_->process(pcm);
}

}  // namespace audio
}  // namespace digital_human
