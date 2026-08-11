#include "audio/audio_framer.h"

#include <cmath>
#include <algorithm>

namespace digital_human {
namespace audio {

struct AudioFramer::Impl {
    std::vector<double> hammingWindow_;

    std::vector<double> buildHammingWindow(int frameSize) {
        constexpr double kPi = 3.14159265358979323846;
        if (frameSize < 2) {
            // frameSize==1 时 denom=0 会除零，强制最小为 2
            return {1.0};
        }
        std::vector<double> window(frameSize);
        double denom = static_cast<double>(frameSize - 1);
        for (int i = 0; i < frameSize; i++) {
            window[i] = 0.54 - 0.46 * std::cos(2.0 * kPi * i / denom);
        }
        return window;
    }

    std::vector<std::vector<float>> frame(const std::vector<float>& pcm,
                                          const FrameConfig& config) {
        int frameSize = config.frameSize;
        int hopSize = config.hopSize;

        if (frameSize <= 0 || hopSize <= 0) {
            throw AudioFramerException("frameSize and hopSize must be positive");
        }
        if (pcm.empty()) {
            throw AudioFramerException("pcm data is empty");
        }

        int N = static_cast<int>(pcm.size());
        int numFrames;
        if (N < frameSize) {
            numFrames = 1;
        } else {
            numFrames = (N - frameSize + hopSize - 1) / hopSize + 1;
        }

        if (hammingWindow_.size() != static_cast<size_t>(frameSize)) {
            hammingWindow_ = buildHammingWindow(frameSize);
        }

        std::vector<std::vector<float>> featureMap(numFrames,
                                                   std::vector<float>(frameSize, 0.0f));

        for (int i = 0; i < numFrames; i++) {
            int start = i * hopSize;
            for (int j = 0; j < frameSize; j++) {
                int idx = start + j;
                float sample = (idx < N) ? pcm[idx] : 0.0f;
                featureMap[i][j] = static_cast<float>(sample * hammingWindow_[j]);
            }
        }

        return featureMap;
    }
};

AudioFramer::AudioFramer() : impl_(std::make_unique<Impl>()) {}

AudioFramer::~AudioFramer() = default;
AudioFramer::AudioFramer(AudioFramer&& other) noexcept = default;
AudioFramer& AudioFramer::operator=(AudioFramer&& other) noexcept = default;

std::vector<std::vector<float>> AudioFramer::frame(const std::vector<float>& pcm,
                                                    const FrameConfig& config) {
    return impl_->frame(pcm, config);
}

}  // namespace audio
}  // namespace digital_human
