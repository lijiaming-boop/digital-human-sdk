#include "audio/audio_vad.h"

#include <algorithm>
#include <cmath>

namespace digital_human {
namespace audio {

struct VoiceActivityDetector::Impl {
    float energyThresh;
    float zcrMin;
    float zcrMax;
    int hangover;

    static float computeEnergy(const std::vector<float>& frame) {
        float energy = 0.0f;
        for (float x : frame) {
            energy += x * x;
        }
        return energy;
    }

    static float computeZCR(const std::vector<float>& frame) {
        if (frame.size() < 2) return 0.0f;
        int crossings = 0;
        for (size_t i = 1; i < frame.size(); i++) {
            if ((frame[i - 1] >= 0.0f) != (frame[i] >= 0.0f)) {
                crossings++;
            }
        }
        return static_cast<float>(crossings) / (frame.size() - 1);
    }

    std::vector<std::vector<float>> filter(
        const std::vector<std::vector<float>>& frames) const {

        if (frames.empty()) return {};

        int N = static_cast<int>(frames.size());
        std::vector<bool> isSpeech(N, false);

        for (int i = 0; i < N; i++) {
            float energy = computeEnergy(frames[i]);
            float zcr = computeZCR(frames[i]);
            isSpeech[i] = (energy >= energyThresh) &&
                          (zcr >= zcrMin) && (zcr <= zcrMax);
        }

        // Apply hangover: extend speech segments
        std::vector<bool> extended = isSpeech;
        for (int i = 0; i < N; i++) {
            if (isSpeech[i]) {
                int lo = std::max(0, i - hangover);
                int hi = std::min(N - 1, i + hangover);
                for (int j = lo; j <= hi; j++) {
                    extended[j] = true;
                }
            }
        }

        std::vector<std::vector<float>> result;
        for (int i = 0; i < N; i++) {
            if (extended[i]) {
                result.push_back(frames[i]);
            }
        }
        return result;
    }
};

VoiceActivityDetector::VoiceActivityDetector(float energyThresh,
                                             float zcrMin,
                                             float zcrMax,
                                             int hangover)
    : impl_(std::make_unique<Impl>()) {
    impl_->energyThresh = energyThresh;
    impl_->zcrMin = zcrMin;
    impl_->zcrMax = zcrMax;
    impl_->hangover = std::max(0, hangover);
}

VoiceActivityDetector::~VoiceActivityDetector() = default;

VoiceActivityDetector::VoiceActivityDetector(VoiceActivityDetector&&) noexcept = default;

VoiceActivityDetector& VoiceActivityDetector::operator=(VoiceActivityDetector&&) noexcept = default;

std::vector<std::vector<float>> VoiceActivityDetector::filter(
    const std::vector<std::vector<float>>& frames) const {
    return impl_->filter(frames);
}

}  // namespace audio
}  // namespace digital_human
