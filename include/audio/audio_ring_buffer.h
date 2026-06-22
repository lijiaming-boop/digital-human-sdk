#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <algorithm>

namespace digital_human {
namespace audio {

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity)
        , buffer_(std::make_unique<T[]>(capacity))
        , writeIdx_(0)
        , readIdx_(0) {}

    ~RingBuffer() = default;

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    RingBuffer(RingBuffer&& other) noexcept
        : capacity_(other.capacity_)
        , buffer_(std::move(other.buffer_))
        , writeIdx_(other.writeIdx_.load())
        , readIdx_(other.readIdx_.load()) {}

    RingBuffer& operator=(RingBuffer&& other) noexcept {
        if (this != &other) {
            capacity_ = other.capacity_;
            buffer_ = std::move(other.buffer_);
            writeIdx_.store(other.writeIdx_.load());
            readIdx_.store(other.readIdx_.load());
        }
        return *this;
    }

    // === Producer API ===

    size_t write(const T* data, size_t count) {
        size_t avail = availableWrite();
        if (avail == 0 || count == 0) return 0;

        size_t n = std::min(count, avail);
        uint64_t w = writeIdx_.load(std::memory_order_relaxed);
        size_t pos = static_cast<size_t>(w % capacity_);

        // First segment: from pos to end of buffer
        size_t first = std::min(n, capacity_ - pos);
        std::copy(data, data + first, buffer_.get() + pos);

        // Second segment: wrap to beginning
        if (first < n) {
            std::copy(data + first, data + n, buffer_.get());
        }

        writeIdx_.store(w + n, std::memory_order_release);
        return n;
    }

    size_t availableWrite() const {
        uint64_t w = writeIdx_.load(std::memory_order_relaxed);
        uint64_t r = readIdx_.load(std::memory_order_acquire);
        return static_cast<size_t>(capacity_ - (w - r));
    }

    // === Consumer API ===

    size_t read(T* data, size_t count) {
        size_t avail = availableRead();
        if (avail == 0 || count == 0) return 0;

        size_t n = std::min(count, avail);
        uint64_t r = readIdx_.load(std::memory_order_relaxed);
        size_t pos = static_cast<size_t>(r % capacity_);

        // First segment: from pos to end of buffer
        size_t first = std::min(n, capacity_ - pos);
        std::copy(buffer_.get() + pos, buffer_.get() + pos + first, data);

        // Second segment: wrap to beginning
        if (first < n) {
            std::copy(buffer_.get(), buffer_.get() + (n - first), data + first);
        }

        readIdx_.store(r + n, std::memory_order_release);
        return n;
    }

    size_t skip(size_t count) {
        size_t avail = availableRead();
        if (avail == 0 || count == 0) return 0;

        size_t n = std::min(count, avail);
        uint64_t r = readIdx_.load(std::memory_order_relaxed);
        readIdx_.store(r + n, std::memory_order_release);
        return n;
    }

    size_t availableRead() const {
        uint64_t w = writeIdx_.load(std::memory_order_acquire);
        uint64_t r = readIdx_.load(std::memory_order_relaxed);
        return static_cast<size_t>(w - r);
    }

    // === Common API ===

    size_t capacity() const { return capacity_; }

    void reset() {
        readIdx_.store(writeIdx_.load(std::memory_order_acquire),
                       std::memory_order_release);
    }

    bool empty() const { return availableRead() == 0; }

    bool full() const { return availableWrite() == 0; }

private:
    size_t capacity_;
    std::unique_ptr<T[]> buffer_;

    // Cache-line padding to avoid false sharing
    alignas(64) std::atomic<uint64_t> writeIdx_;
    alignas(64) std::atomic<uint64_t> readIdx_;
};

}  // namespace audio
}  // namespace digital_human
