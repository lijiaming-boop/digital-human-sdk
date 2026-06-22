#pragma once

#include <vector>
#include <memory>

namespace digital_human {
namespace audio {

class RMSNormalize {
public:
    explicit RMSNormalize(float targetRMS = 0.056f);
    ~RMSNormalize();
    RMSNormalize(const RMSNormalize&) = delete;
    RMSNormalize& operator=(const RMSNormalize&) = delete;
    RMSNormalize(RMSNormalize&&) noexcept;
    RMSNormalize& operator=(RMSNormalize&&) noexcept;

    std::vector<float> process(const std::vector<float>& pcm) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
