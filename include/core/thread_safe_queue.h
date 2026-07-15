#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <queue>

namespace digital_human {
namespace core {

/**
 * @brief 线程安全队列
 *
 * 支持多生产者/多消费者模型，提供阻塞和非阻塞两种弹出方式。
 * 通过 Stop() 信号唤醒所有等待线程，支持优雅退出。
 *
 * 线程安全保证：
 * - Push / TryPop / WaitAndPop 均可从任意线程调用
 * - 所有公有方法内部加锁，外部无需额外同步
 * - 单个队列实例内部不会死锁（不嵌套加锁）
 *
 * @tparam T 元素类型，要求可移动构造
 */
template <typename T>
class ThreadSafeQueue {
public:
    using size_type = size_t;

    /// @brief 构造一个有界或无界队列
    /// @param max_capacity 最大容量，0 表示无界
    explicit ThreadSafeQueue(size_type max_capacity = 0)
        : max_capacity_(max_capacity) {}

    /// @brief 禁止拷贝
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    /// @brief 允许移动构造（不推荐并发时使用）
    ThreadSafeQueue(ThreadSafeQueue&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mutex_);
        queue_        = std::move(other.queue_);
        stopped_      = other.stopped_.load();
        max_capacity_ = other.max_capacity_;
    }

    // ========================================================================
    // 生产者接口
    // ========================================================================

    /**
     * @brief 入队
     *
     * 有界队列在队列满时会阻塞，直到有空间或队列被停止。
     *
     * @param item 待入队元素（移动）
     * @return true  入队成功
     * @return false 队列已停止
     */
    bool Push(T item) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            not_full_cv_.wait(lock, [this]() {
                return stopped_.load(std::memory_order_acquire) ||
                       max_capacity_ == 0 ||
                       queue_.size() < max_capacity_;
            });

            if (stopped_.load(std::memory_order_acquire)) {
                return false;
            }

            queue_.push(std::move(item));
        }
        not_empty_cv_.notify_one();
        return true;
    }

    /**
     * @brief 尝试入队（非阻塞）
     *
     * @param item 待入队元素
     * @return true  入队成功
     * @return false 队列满或已停止
     */
    bool TryPush(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_.load(std::memory_order_acquire)) {
                return false;
            }
            if (max_capacity_ > 0 && queue_.size() >= max_capacity_) {
                return false;
            }
            queue_.push(std::move(item));
        }
        not_empty_cv_.notify_one();
        return true;
    }

    // ========================================================================
    // 消费者接口
    // ========================================================================

    /**
     * @brief 阻塞出队（带超时）
     *
     * @param[out] item      出队元素
     * @param timeout_ms     超时时间（毫秒），-1 表示无限等待
     * @return true  出队成功
     * @return false 超时或队列已停止
     */
    bool WaitAndPop(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (timeout_ms < 0) {
            not_empty_cv_.wait(lock, [this]() {
                return stopped_.load(std::memory_order_acquire) || !queue_.empty();
            });
        } else if (!not_empty_cv_.wait_for(lock,
                       std::chrono::milliseconds(timeout_ms), [this]() {
                           return stopped_.load(std::memory_order_acquire) ||
                                  !queue_.empty();
                       })) {
            return false;  // 超时
        }

        if (stopped_.load(std::memory_order_acquire) && queue_.empty()) {
            return false;  // 已停止且队列为空
        }

        item = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        not_full_cv_.notify_one();
        return true;
    }

    /**
     * @brief 非阻塞出队
     *
     * @param[out] item 出队元素
     * @return true  出队成功
     * @return false 队列空或已停止
     */
    bool TryPop(T& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) return false;
            item = std::move(queue_.front());
            queue_.pop();
        }
        not_full_cv_.notify_one();
        return true;
    }

    // ========================================================================
    // 控制接口
    // ========================================================================

    /**
     * @brief 停止队列
     *
     * 唤醒所有等待 Push/WaitAndPop 的线程。
     * 停止后 Push/TryPush 返回 false，WaitAndPop 在队列清空后返回 false。
     */
    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_.store(true, std::memory_order_release);
        }
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();
    }

    /// @brief 检查队列是否已停止
    bool IsStopped() const {
        return stopped_.load(std::memory_order_acquire);
    }

    /// @brief 清空队列
    void Clear() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::queue<T> empty;
            std::swap(queue_, empty);
        }
        not_full_cv_.notify_all();
    }

    // ========================================================================
    // 查询接口
    // ========================================================================

    /// @brief 获取当前队列大小
    size_type Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /// @brief 队列是否为空
    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /// @brief 队列是否已满（有界队列）
    bool Full() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return max_capacity_ > 0 && queue_.size() >= max_capacity_;
    }

    /// @brief 获取最大容量
    size_type Capacity() const { return max_capacity_; }

private:
    mutable std::mutex           mutex_;
    std::queue<T, std::deque<T>> queue_;
    std::condition_variable      not_empty_cv_;  ///< 非空条件变量
    std::condition_variable      not_full_cv_;   ///< 非满条件变量（有界队列）
    std::atomic<bool>            stopped_{false};
    size_type                    max_capacity_ = 0;
};

}  // namespace core
}  // namespace digital_human
