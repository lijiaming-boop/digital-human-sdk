#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

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
        ThreadState expected = ThreadState::INIT;
        if (!state_.compare_exchange_strong(expected, ThreadState::RUNNING)) {
            return false;
        }

        thread_ = std::thread([this]() {
            try {
                Run();
            } catch (const std::exception& e) {
                LogError(std::string("未捕获异常: ") + e.what());
                state_.store(ThreadState::ERROR, std::memory_order_release);
            } catch (...) {
                LogError("未捕获未知异常");
                state_.store(ThreadState::ERROR, std::memory_order_release);
            }
        });

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
     * std::thread 不支持超时 join，timeout_ms 为预留参数（当前始终阻塞等待）。
     *
     * @param timeout_ms 预留参数，当前无效（始终完全等待）
     * @return true  线程正常退出
     */
    bool Wait(int /*timeout_ms*/ = -1) {
        if (thread_.joinable()) {
            thread_.join();
        }
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
};

}  // namespace core
}  // namespace digital_human
