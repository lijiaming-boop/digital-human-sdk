#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

// WinBase.h defines ERROR as a macro. Keep the enum value usable on Windows.
#ifdef ERROR
#undef ERROR
#endif

namespace digital_human {
namespace core {

/// @brief 线程状态
enum class ThreadState : int8_t {
    INIT,       ///< 已创建，未启动
    RUNNING,    ///< 运行中
    STOPPING,   ///< 收到停止信号，正在退出
    STOPPED,    ///< 已停止
    ERROR       ///< 异常退出
};

/// @brief 线程状态字符串
inline const char* ThreadStateToString(ThreadState s) {
    switch (s) {
        case ThreadState::INIT:     return "INIT";
        case ThreadState::RUNNING:  return "RUNNING";
        case ThreadState::STOPPING: return "STOPPING";
        case ThreadState::STOPPED:  return "STOPPED";
        case ThreadState::ERROR:    return "ERROR";
        default:                    return "UNKNOWN";
    }
}

/**
 * @brief 线程基类
 *
 * 提供统一的线程生命周期管理：
 *   INIT → Start() → RUNNING → Stop() → STOPPING → Wait() → STOPPED
 *                                              ↘ ERROR
 *
 * 继承该类并实现 Run() 方法，即可获得完整的线程控制能力。
 */
class ThreadBase {
public:
    /// @brief 构造函数
    /// @param name 线程名称（用于日志和调试）
    explicit ThreadBase(std::string name)
        : name_(std::move(name)) {}

    virtual ~ThreadBase() { Wait(); }

    ThreadBase(const ThreadBase&) = delete;
    ThreadBase& operator=(const ThreadBase&) = delete;
    ThreadBase(ThreadBase&&) = delete;
    ThreadBase& operator=(ThreadBase&&) = delete;

    // ========================================================================
    // 生命周期控制
    // ========================================================================

    /**
     * @brief 启动线程
     *
     * 调用后线程状态变为 RUNNING，并开始执行 Run()。
     * 重复调用无效（只能启动一次）。
     *
     * @return true  启动成功
     * @return false 线程已在运行
     */
    bool Start() {
        std::lock_guard<std::mutex> join_lock(join_mutex_);

        ThreadState expected = ThreadState::INIT;
        if (!state_.compare_exchange_strong(expected, ThreadState::RUNNING)) {
            return false;
        }

        run_exited_.store(false, std::memory_order_release);
        try {
            thread_ = std::thread([this]() {
                try {
                    Run();
                } catch (const std::exception& e) {
                    LogError(std::string("Unhandled exception: ") + e.what());
                    state_.store(ThreadState::ERROR, std::memory_order_release);
                } catch (...) {
                    LogError("Unhandled unknown exception");
                    state_.store(ThreadState::ERROR, std::memory_order_release);
                }

                run_exited_.store(true, std::memory_order_release);
                run_exited_cv_.notify_all();
            });
        } catch (...) {
            state_.store(ThreadState::ERROR, std::memory_order_release);
            throw;
        }

        return true;
    }

    /**
     * @brief 请求停止线程
     *
     * 设置停止标志，Run() 循环应定期检查 IsStopping()。
     * 非阻塞，不等待线程实际退出。
     */
    void Stop() {
        ThreadState expected = ThreadState::RUNNING;
        state_.compare_exchange_strong(expected, ThreadState::STOPPING);
    }

    /**
     * @brief 等待线程退出
     *
     * 阻塞直到线程完全退出。
     *
     * 协作式停止：调用方应先调用 Stop() 并唤醒线程可能阻塞的队列/条件变量，
     * 然后再调用 Wait()。Run() 主循环必须定期检查 IsStopping() 才能保证退出。
     *
     * 超时语义：
     *   - timeout_ms < 0：无限等待，直接 join()。
     *   - timeout_ms >= 0：使用条件变量等待 Run() 退出，超时返回 false。
     *     不会 detach 线程（避免对同一 std::thread 对象并发操作导致的 UB），
     *     超时后线程仍由本对象持有，下一次 Wait() 或析构函数会重新尝试回收。
     *
     * @param timeout_ms 超时（毫秒），-1 表示无限等待
     * @return true  线程已正常退出
     * @return false 超时（线程仍在运行，需稍后再次调用 Wait() 或由析构回收）
     */
    bool Wait(int timeout_ms = -1) {
        // std::thread::join() is not safe to call concurrently. Serialize Wait
        // so only one caller can inspect and consume the joinable state.
        std::lock_guard<std::mutex> join_lock(join_mutex_);

        if (!thread_.joinable()) {
            FinalizeState();
            return true;
        }

        if (timeout_ms < 0) {
            // Destruction must eventually join; never detach an owned worker.
            thread_.join();
            FinalizeState();
            return true;
        }

        const auto deadline = std::chrono::steady_clock::now()
                            + std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::mutex> exited_lock(run_exited_mutex_);
        const bool exited = run_exited_cv_.wait_until(
            exited_lock,
            deadline,
            [this]() {
                return run_exited_.load(std::memory_order_acquire);
            });
        exited_lock.unlock();

        if (!exited) {
            const auto s = state_.load(std::memory_order_acquire);
            std::cerr << "[ThreadBase] Wait timed out (" << timeout_ms
                      << "ms), worker is still running (state="
                      << ThreadStateToString(s)
                      << "), thread remains owned; retry Wait() later" << std::endl;
            return false;
        }

        // run_exited_ is published, so join only reclaims thread resources.
        thread_.join();
        FinalizeState();
        return true;
    }

    /**
     * @brief 启动 → 停止 → 等待（快捷方法）
     *
     * 在销毁前调用，确保线程干净退出。
     */
    void Shutdown() {
        Stop();
        Wait();
    }

    // ========================================================================
    // 状态查询
    // ========================================================================

    /// @brief 获取当前线程状态
    ThreadState GetState() const {
        return state_.load(std::memory_order_acquire);
    }

    /// @brief 检查线程是否正在运行
    bool IsRunning() const {
        return state_.load(std::memory_order_acquire) == ThreadState::RUNNING;
    }

    /// @brief 检查线程是否应停止（Run() 循环判断条件）
    bool IsStopping() const {
        auto s = state_.load(std::memory_order_acquire);
        return s == ThreadState::STOPPING || s == ThreadState::ERROR;
    }

    /// @brief 是否已完全停止
    bool IsStopped() const {
        auto s = state_.load(std::memory_order_acquire);
        return s == ThreadState::STOPPED || s == ThreadState::ERROR;
    }

    /// @brief 是否发生错误
    bool IsError() const {
        return state_.load(std::memory_order_acquire) == ThreadState::ERROR;
    }

    /// @brief 获取线程名称
    const std::string& GetName() const { return name_; }

protected:
    // ========================================================================
    // 子类接口
    // ========================================================================

    /**
     * @brief 线程主循环（子类实现）
     *
     * 应定期检查 IsStopping() 以支持优雅退出。
     * 抛出异常会被基类捕获并转换为 ERROR 状态。
     */
    virtual void Run() = 0;

    /// @brief 记录错误日志
    void LogError(const std::string& msg) {
        std::cerr << "[" << name_ << "] ERROR: " << msg << std::endl;
    }

    /// @brief 记录信息日志
    void LogInfo(const std::string& msg) {
        std::cout << "[" << name_ << "] " << msg << std::endl;
    }

private:
    /// @brief 最终化线程状态
    void FinalizeState() {
        auto s = state_.load(std::memory_order_acquire);
        if (s == ThreadState::RUNNING || s == ThreadState::STOPPING) {
            state_.store(ThreadState::STOPPED, std::memory_order_release);
        }
    }

    std::string              name_;
    std::thread              thread_;
    std::atomic<ThreadState> state_{ThreadState::INIT};
    /// 由线程自身在 Run() 退出时设置，供 Wait(timeout) 条件等待使用，
    /// 避免依赖 state_（state_ 由 FinalizeState 推进，存在循环依赖）。
    std::atomic<bool>        run_exited_{false};
    mutable std::mutex       join_mutex_;
    mutable std::mutex       run_exited_mutex_;
    std::condition_variable  run_exited_cv_;
};

}  // namespace core
}  // namespace digital_human
