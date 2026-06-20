#pragma once

#include <string>
#include <vector>
#include <memory>
#include <exception>
#include <cstdint>

namespace digital_human {
namespace audio {

struct AudioData {
    std::vector<float> samples;
    int sampleRate = 16000;
    int channels = 1;
    double duration = 0.0;
};

class AudioLoader {
public:
    AudioLoader();
    ~AudioLoader();
    AudioLoader(const AudioLoader& other) = delete;
    AudioLoader& operator=(const AudioLoader& other) = delete;
    AudioLoader(AudioLoader&& other) noexcept;
    AudioLoader& operator=(AudioLoader&& other) noexcept;

    AudioData load(const std::string& filePath);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class AudioLoaderException : public std::runtime_error {
public:
    explicit AudioLoaderException(const std::string& message)
        : std::runtime_error(message) {}
};

}  // namespace audio
}  // namespace digital_human
