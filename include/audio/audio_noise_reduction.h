#pragma once

#include <vector>
#include <memory>

namespace digital_human {
namespace audio {

class NoiseReduction {
public:
    NoiseReduction(int noiseFrames = 10, float oversubtraction = 0.02f);
    ~NoiseReduction();
    NoiseReduction(const NoiseReduction&) = delete;
    NoiseReduction& operator=(const NoiseReduction&) = delete;
    NoiseReduction(NoiseReduction&&) noexcept;
    NoiseReduction& operator=(NoiseReduction&&) noexcept;

    std::vector<float> process(const std::vector<float>& pcm, int sampleRate);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
