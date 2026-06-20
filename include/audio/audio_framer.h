#pragma once

#include <vector>
#include <memory>
#include <exception>
#include <cstdint>

namespace digital_human {
namespace audio {

struct FrameConfig {
    int frameSize = 400;
    int hopSize = 160;
};

class AudioFramer {
public:
    AudioFramer();
    ~AudioFramer();
    AudioFramer(const AudioFramer& other) = delete;
    AudioFramer& operator=(const AudioFramer& other) = delete;
    AudioFramer(AudioFramer&& other) noexcept;
    AudioFramer& operator=(AudioFramer&& other) noexcept;

    std::vector<std::vector<float>> frame(const std::vector<float>& pcm,
                                          const FrameConfig& config = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class AudioFramerException : public std::runtime_error {
public:
    explicit AudioFramerException(const std::string& message)
        : std::runtime_error(message) {}
};

}  // namespace audio
}  // namespace digital_human
