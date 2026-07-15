#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace digital_human {
namespace core {

// ============================================================================
// 队列运行时指标
// ============================================================================

/// @brief 队列运行时指标
struct QueueMetrics {
    std::string name;             ///< 队列名称
    size_t      current_size  = 0; ///< 当前大小
    size_t      capacity      = 0; ///< 最大容量（0=无界）
    size_t      peak_size     = 0; ///< 历史峰值
    size_t      total_pushes  = 0; ///< 累计入队次数
    size_t      total_pops    = 0; ///< 累计出队次数
    size_t      total_overflows = 0; ///< 累计溢出次数
    bool        is_stopped    = false; ///< 是否已停止
    bool        is_healthy    = false; ///< 心跳是否健康
    int64_t     last_push_ms  = 0;  ///< 上次入队时间戳
    int64_t     last_pop_ms   = 0;  ///< 上次出队时间戳
    int64_t     last_heartbeat_ms = 0; ///< 上次心跳时间戳

    std::string ToString() const {
        std::ostringstream oss;
        oss << "Queue[" << name << "] {"
            << " size=" << current_size << "/" << (capacity ? std::to_string(capacity) : "unbounded")
            << " peak=" << peak_size
            << " pushes=" << total_pushes
            << " pops=" << total_pops
            << " overflow=" << total_overflows
            << " healthy=" << (is_healthy ? "yes" : "no")
            << " stopped=" << (is_stopped ? "yes" : "no")
            << " }";
        return oss.str();
    }
};

// ============================================================================
// ThreadSafeQueue
// ============================================================================

/**
 * @brief 高级线程安全队列
 *
 * 特性：
 * - 有界/无界队列，防止内存溢出
 * - 同步与互斥操作（mutex + condition_variable）
 * - 超时等待，防止线程永久阻塞
 * - 流水线心跳检测 + 死锁检测
 * - 批量出队优化
 * - 移动语义 + 原位构造，避免拷贝
 * - 运行时指标收集
 *
 * @tparam T 元素类型，要求可移动构造
 */
template <typename T>
class ThreadSafeQueue {
public:
    using size_type = size_t;

    /**
     * @brief 构造线程安全队列
     *
     * @param max_capacity         最大容量，0 表示无界
     * @param name                 队列名称（用于日志和指标）
     * @param heartbeat_timeout_ms 心跳超时（毫秒），0 表示不启用心跳检测
     */
    explicit ThreadSafeQueue(size_type max_capacity = 0,
                             const std::string& name = "queue",
                             size_type heartbeat_timeout_ms = 0)
        : max_capacity_(max_capacity)
        , name_(name)
        , heartbeat_timeout_ms_(heartbeat_timeout_ms) {}

    /// @brief 禁止拷贝
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    /// @brief 允许移动构造（不推荐并发时使用）
    ThreadSafeQueue(ThreadSafeQueue&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mutex_);
        queue_               = std::move(other.queue_);
        stopped_.store(other.stopped_.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
        max_capacity_        = other.max_capacity_;
        name_                = std::move(other.name_);
        heartbeat_timeout_ms_ = other.heartbeat_timeout_ms_;
        total_pushes_.store(other.total_pushes_.load(std::memory_order_relaxed));
        total_pops_.store(other.total_pops_.load(std::memory_order_relaxed));
    }

    // ========================================================================
    // 生产者接口
    // ========================================================================

    /**
     * @brief 入队（移动语义）
     *
     * 有界队列满时会阻塞，直到有空间或队列被停止。
     * 使用移动语义避免拷贝。
     *
     * @param item 待入队元素（右值引用）
     * @return true  入队成功
     * @return false 队列已停止
     */
    bool Push(T&& item) {
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

            queue_.emplace_back(std::move(item));
            UpdatePushStats();
        }
        not_empty_cv_.notify_one();
        return true;
    }

    /**
     * @brief 入队（const 左值引用 → 拷贝）
     *
     * 仅在需要保留原值时使用。优先使用 Push(T&&) 避免拷贝。
     *
     * @param item 待入队元素（const 左值引用）
     * @return true  入队成功
     * @return false 队列已停止
     */
    bool Push(const T& item) {
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

            queue_.push_back(item);
            UpdatePushStats();
        }
        not_empty_cv_.notify_one();
        return true;
    }

    /**
     * @brief 原位构造入队
     *
     * 直接在队列中构造对象，避免任何拷贝/移动。
     *
     * @tparam Args 构造参数类型
     * @param args  构造参数
     * @return true  入队成功
     * @return false 队列已停止
     */
    template <typename... Args>
    bool Emplace(Args&&... args) {
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

            queue_.emplace_back(std::forward<Args>(args)...);
            UpdatePushStats();
        }
        not_empty_cv_.notify_one();
        return true;
    }

    /**
     * @brief 尝试入队（非阻塞）
     *
     * @param item 待入队元素（右值引用）
     * @return true  入队成功
     * @return false 队列满或已停止
     */
    bool TryPush(T&& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_.load(std::memory_order_acquire)) return false;
            if (max_capacity_ > 0 && queue_.size() >= max_capacity_) {
                total_overflows_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            queue_.emplace_back(std::move(item));
            UpdatePushStats();
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
     * @param[out] item      出队元素（引用）
     * @param timeout_ms     超时时间（毫秒），-1 无限等待，0 非阻塞
     * @return true  出队成功
     * @return false 超时或队列已停止
     */
    bool WaitAndPop(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (timeout_ms == 0) {
            // 非阻塞模式
            if (queue_.empty()) return false;
            return doPop(item, lock);
        }

        if (timeout_ms < 0) {
            // 无限等待
            not_empty_cv_.wait(lock, [this]() {
                return stopped_.load(std::memory_order_acquire) || !queue_.empty();
            });
        } else {
            // 带超时等待
            auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::milliseconds(timeout_ms);

            if (!not_empty_cv_.wait_until(lock, deadline, [this]() {
                    return stopped_.load(std::memory_order_acquire) || !queue_.empty();
                })) {
                return false;  // 超时
            }
        }

        if (stopped_.load(std::memory_order_acquire) && queue_.empty()) {
            return false;
        }

        return doPop(item, lock);
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
            queue_.pop_front();
            UpdatePopStats();
        }
        not_full_cv_.notify_one();
        return true;
    }

    /**
     * @brief 批量出队
     *
     * 一次性取出最多 max_count 个元素，减少锁竞争。
     *
     * @param[out] items     出队元素容器
     * @param max_count      最大出队数量，0 表示全部取出
     * @return size_t        实际出队数量
     */
    size_t TryPopBatch(std::vector<T>& items, size_t max_count = 0) {
        size_t count = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) return 0;

            size_t available = queue_.size();
            count = (max_count == 0) ? available : std::min(max_count, available);
            items.reserve(items.size() + count);

            for (size_t i = 0; i < count; ++i) {
                items.emplace_back(std::move(queue_.front()));
                queue_.pop_front();
            }
            UpdatePopStats(count);
        }
        if (count > 0) {
            not_full_cv_.notify_all();
        }
        return count;
    }

    // ========================================================================
    // 心跳与死锁检测
    // ========================================================================

    /**
     * @brief 发送心跳信号
     *
     * 每个消费者线程应定期调用此方法，通知队列该线程仍在运行。
     */
    void Heartbeat() {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        last_heartbeat_.store(now, std::memory_order_release);
    }

    /**
     * @brief 检查队列是否健康
     *
     * 基于心跳超时检测。如果启用心跳但超过 timeout 未收到心跳，则不健康。
     * 如果未启用心跳检测，始终返回 true。
     *
     * @return true  健康
     * @return false 心跳超时（疑似死锁或线程挂起）
     */
    bool IsHealthy() const {
        if (heartbeat_timeout_ms_ == 0) return true;  // 未启用

        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        auto last = last_heartbeat_.load(std::memory_order_acquire);
        if (last == 0) return true;  // 从未心跳，视为健康

        return (now - last) < static_cast<int64_t>(heartbeat_timeout_ms_);
    }

    /**
     * @brief 检查是否发生死锁（数据停滞）
     *
     * 如果队列非空但长时间没有出队，或者队列未满但长时间没有入队，
     * 则可能存在死锁或生产者/消费者停滞。
     *
     * @param stall_timeout_ms 停滞超时（毫秒）
     * @return true  检测到可能的死锁
     * @return false 正常
     */
    bool CheckDeadlock(size_t stall_timeout_ms) const {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        auto last_push = last_push_time_.load(std::memory_order_acquire);
        auto last_pop  = last_pop_time_.load(std::memory_order_acquire);

        bool no_push_stall = (last_push == 0) ||
            (now - last_push) < static_cast<int64_t>(stall_timeout_ms);
        bool no_pop_stall  = (last_pop == 0) ||
            (now - last_pop) < static_cast<int64_t>(stall_timeout_ms);

        // 队列非空但消费者停滞：消费者可能死锁
        // 队列未满但生产者停滞：生产者可能死锁
        bool size_nonzero = false;
        bool size_nonfull = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            size_nonzero  = !queue_.empty();
            size_nonfull  = (max_capacity_ == 0) || (queue_.size() < max_capacity_);
        }

        if (size_nonzero && !no_pop_stall) {
            std::cerr << "[Queue:" << name_ << "] 死锁警告: 队列非空但消费者停滞 "
                      << (now - last_pop) << "ms" << std::endl;
            return true;
        }

        if (size_nonfull && !no_push_stall && !Empty()) {
            std::cerr << "[Queue:" << name_ << "] 死锁警告: 队列未满但生产者停滞 "
                      << (now - last_push) << "ms" << std::endl;
            return true;
        }

        return false;
    }

    // ========================================================================
    // 控制接口
    // ========================================================================

    /**
     * @brief 停止队列
     *
     * 唤醒所有等待中的线程。
     * 停止后 Push/TryPush 返回 false，WaitAndPop 在清空后返回 false。
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
            queue_.clear();
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

    /// @brief 获取队列名称
    const std::string& GetName() const { return name_; }

    // ========================================================================
    // 指标收集
    // ========================================================================

    /// @brief 获取队列运行时指标
    QueueMetrics GetMetrics() const {
        QueueMetrics m;
        m.name              = name_;
        m.capacity           = max_capacity_;
        m.total_pushes       = total_pushes_.load(std::memory_order_relaxed);
        m.total_pops         = total_pops_.load(std::memory_order_relaxed);
        m.total_overflows    = total_overflows_.load(std::memory_order_relaxed);
        m.peak_size          = peak_size_.load(std::memory_order_relaxed);
        m.is_stopped         = stopped_.load(std::memory_order_acquire);
        m.is_healthy         = IsHealthy();
        m.last_push_ms       = last_push_time_.load(std::memory_order_relaxed);
        m.last_pop_ms        = last_pop_time_.load(std::memory_order_relaxed);
        m.last_heartbeat_ms  = last_heartbeat_.load(std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            m.current_size = queue_.size();
        }
        return m;
    }

private:
    // ========================================================================
    // 内部方法
    // ========================================================================

    /// @brief 内部出队（调用方已持有锁）
    bool doPop(T& item, std::unique_lock<std::mutex>& lock) {
        item = std::move(queue_.front());
        queue_.pop_front();
        UpdatePopStats();
        lock.unlock();
        not_full_cv_.notify_one();
        return true;
    }

    /// @brief 更新入队统计
    void UpdatePushStats() {
        total_pushes_.fetch_add(1, std::memory_order_relaxed);
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        last_push_time_.store(now, std::memory_order_release);
        // 更新峰值
        size_t cur = queue_.size();
        size_t peak = peak_size_.load(std::memory_order_relaxed);
        while (cur > peak) {
            peak_size_.compare_exchange_weak(peak, cur,
                std::memory_order_release, std::memory_order_relaxed);
        }
    }

    /// @brief 更新出队统计
    void UpdatePopStats(size_t count = 1) {
        total_pops_.fetch_add(count, std::memory_order_relaxed);
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        last_pop_time_.store(now, std::memory_order_release);
    }

    // ---- 同步原语 ----
    mutable std::mutex           mutex_;
    std::deque<T>                queue_;
    std::condition_variable      not_empty_cv_;     ///< 非空条件变量
    std::condition_variable      not_full_cv_;      ///< 非满条件变量

    // ---- 状态 ----
    std::atomic<bool>            stopped_{false};
    size_t                       max_capacity_ = 0;
    std::string                  name_;

    // ---- 心跳检测 ----
    size_t                       heartbeat_timeout_ms_ = 0;
    std::atomic<int64_t>         last_push_time_{0};
    std::atomic<int64_t>         last_pop_time_{0};
    std::atomic<int64_t>         last_heartbeat_{0};

    // ---- 运行时统计 ----
    std::atomic<size_t>          total_pushes_{0};
    std::atomic<size_t>          total_pops_{0};
    std::atomic<size_t>          total_overflows_{0};
    std::atomic<size_t>          peak_size_{0};
};

}  // namespace core
}  // namespace digital_human
