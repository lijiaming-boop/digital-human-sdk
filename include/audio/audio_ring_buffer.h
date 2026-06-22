#pragma once

#include <cstddef>
#include <memory>

namespace digital_human {
namespace audio {

class RingBuffer {
public:
    explicit RingBuffer(size_t capacity);
    ~RingBuffer();

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&) noexcept;
    RingBuffer& operator=(RingBuffer&&) noexcept;

    // Producer API
    size_t write(const float* data, size_t count);
    size_t availableWrite() const;

    // Consumer API
    size_t read(float* data, size_t count);
    size_t skip(size_t count);
    size_t availableRead() const;

    // Common API
    size_t capacity() const;
    void reset();
    bool empty() const;
    bool full() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
