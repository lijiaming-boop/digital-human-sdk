#include "audio/audio_ring_buffer.h"

#include <atomic>
#include <cstdint>
#include <algorithm>
#include <cstring>

namespace digital_human {
namespace audio {

struct RingBuffer::Impl {
    size_t capacity;
    std::unique_ptr<float[]> buffer;

    alignas(64) std::atomic<uint64_t> writeIdx;
    alignas(64) std::atomic<uint64_t> readIdx;

    explicit Impl(size_t cap)
        : capacity(cap)
        , buffer(std::make_unique<float[]>(cap))
        , writeIdx(0)
        , readIdx(0) {}

    size_t write(const float* data, size_t count) {
        size_t avail = availableWrite();
        if (avail == 0 || count == 0) return 0;

        size_t n = std::min(count, avail);
        uint64_t w = writeIdx.load(std::memory_order_relaxed);
        size_t pos = static_cast<size_t>(w % capacity);

        size_t first = std::min(n, capacity - pos);
        std::memcpy(buffer.get() + pos, data, first * sizeof(float));

        if (first < n) {
            std::memcpy(buffer.get(), data + first, (n - first) * sizeof(float));
        }

        writeIdx.store(w + n, std::memory_order_release);
        return n;
    }

    size_t availableWrite() const {
        uint64_t w = writeIdx.load(std::memory_order_relaxed);
        uint64_t r = readIdx.load(std::memory_order_acquire);
        return static_cast<size_t>(capacity - (w - r));
    }

    size_t read(float* data, size_t count) {
        size_t avail = availableRead();
        if (avail == 0 || count == 0) return 0;

        size_t n = std::min(count, avail);
        uint64_t r = readIdx.load(std::memory_order_relaxed);
        size_t pos = static_cast<size_t>(r % capacity);

        size_t first = std::min(n, capacity - pos);
        std::memcpy(data, buffer.get() + pos, first * sizeof(float));

        if (first < n) {
            std::memcpy(data + first, buffer.get(), (n - first) * sizeof(float));
        }

        readIdx.store(r + n, std::memory_order_release);
        return n;
    }

    size_t skip(size_t count) {
        size_t avail = availableRead();
        if (avail == 0 || count == 0) return 0;

        size_t n = std::min(count, avail);
        uint64_t r = readIdx.load(std::memory_order_relaxed);
        readIdx.store(r + n, std::memory_order_release);
        return n;
    }

    size_t availableRead() const {
        uint64_t w = writeIdx.load(std::memory_order_acquire);
        uint64_t r = readIdx.load(std::memory_order_relaxed);
        return static_cast<size_t>(w - r);
    }

    void reset() {
        readIdx.store(writeIdx.load(std::memory_order_acquire),
                      std::memory_order_release);
    }
};

RingBuffer::RingBuffer(size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}

RingBuffer::~RingBuffer() = default;

RingBuffer::RingBuffer(RingBuffer&&) noexcept = default;

RingBuffer& RingBuffer::operator=(RingBuffer&&) noexcept = default;

size_t RingBuffer::write(const float* data, size_t count) {
    return impl_->write(data, count);
}

size_t RingBuffer::availableWrite() const {
    return impl_->availableWrite();
}

size_t RingBuffer::read(float* data, size_t count) {
    return impl_->read(data, count);
}

size_t RingBuffer::skip(size_t count) {
    return impl_->skip(count);
}

size_t RingBuffer::availableRead() const {
    return impl_->availableRead();
}

size_t RingBuffer::capacity() const {
    return impl_->capacity;
}

void RingBuffer::reset() {
    impl_->reset();
}

bool RingBuffer::empty() const {
    return impl_->availableRead() == 0;
}

bool RingBuffer::full() const {
    return impl_->availableWrite() == 0;
}

}  // namespace audio
}  // namespace digital_human
