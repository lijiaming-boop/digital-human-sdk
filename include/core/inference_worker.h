#pragma once

#include <memory>
#include <atomic>
#include <cstdint>

#include "core/thread_base.h"
#include "core/thread_safe_queue.h"
#include "core/packet.h"

namespace digital_human {
namespace model {
class ModelInferencer;
}

namespace core {

// ============================================================================
// 推理线程配置
// ============================================================================

/// @brief 推理线程配置
struct InferenceWorkerConfig {
    int   input_queue_warn_threshold  = 10;   ///< 输入队列深度警告阈值
    int   input_queue_error_threshold = 30;   ///< 输入队列深度错误阈值
    int   max_retries                 = 3;    ///< 最大重试次数
    int   pop_timeout_ms              = 100;  ///< 队列弹出超时（毫秒）
    int   backlog_check_interval      = 1000; ///< 队列积压检测间隔（毫秒）
    float latency_warn_threshold_ms   = 100.0f; ///< 推理延迟警告阈值（毫秒）
};

// ============================================================================
// 推理指标
// ============================================================================

/// @brief 推理线程运行时指标
struct InferenceMetrics {
    int64_t total_inferences       = 0;    ///< 总推理次数
    int64_t total_success          = 0;    ///< 成功次数
    int64_t total_failures         = 0;    ///< 失败次数
    int64_t total_retries          = 0;    ///< 重试次数
    int64_t skipped_due_to_backlog = 0;    ///< 因积压跳过的帧数
    double  avg_latency_ms         = 0.0;  ///< 平均推理延迟（毫秒）
    double  min_latency_ms         = 0.0;  ///< 最小延迟
    double  max_latency_ms         = 0.0;  ///< 最大延迟
    double  ewma_latency_ms        = 0.0;  ///< 指数移动平均延迟
    int     input_queue_depth      = 0;    ///< 当前输入队列深度
    int     retry_queue_depth      = 0;    ///< 重试队列深度
    bool    backlog_warning        = false;///< 是否处于积压警告状态

    std::string ToString() const;
};

// ============================================================================
// 推理线程
// ============================================================================

/**
 * @brief Wav2Lip 模型推理线程
 *
 * 从输入队列获取 InferenceTask，执行模型前向传播，
 * 将生成的唇形同步图像推送到输出队列。
 *
 * 特性：
 * - 自动失败重试（最多 kMaxRetries 次）
 * - 队列积压检测与反压
 * - 推理延迟测量（EWMA 平滑）
 * - 输入张量转换（cv::Mat → ncnn::Mat）
 * - 线程安全退出
 */
class InferenceWorker : public ThreadBase {
public:
    /// @brief 构造推理线程
    /// @param name 线程名称
    explicit InferenceWorker(const std::string& name = "InferenceWorker");

    ~InferenceWorker() override;

    InferenceWorker(const InferenceWorker&) = delete;
    InferenceWorker& operator=(const InferenceWorker&) = delete;
    InferenceWorker(InferenceWorker&&) = delete;
    InferenceWorker& operator=(InferenceWorker&&) = delete;

    // ========================================================================
    // 配置
    // ========================================================================

    /// @brief 设置推理线程配置
    void SetConfig(const InferenceWorkerConfig& config);

    /// @brief 获取当前配置
    const InferenceWorkerConfig& GetConfig() const;

    // ========================================================================
    // 模型
    // ========================================================================

    /// @brief 设置模型推理器（外部管理生命周期）
    void SetModelInferencer(model::ModelInferencer* inferencer);

    // ========================================================================
    // 队列
    // ========================================================================

    /// @brief 设置输入任务队列
    void SetInputQueue(ThreadSafeQueue<InferenceTask>* queue);

    /// @brief 设置输出结果队列
    void SetOutputQueue(ThreadSafeQueue<InferenceOutputPacket>* queue);

    // ========================================================================
    // 线程主循环
    // ========================================================================

    void Run() override;

    // ========================================================================
    // 控制
    // ========================================================================

    /// @brief 重置所有统计和状态
    void ResetStats();

    /// @brief 标记输入结束（无更多任务）
    void MarkInputEOS();

    // ========================================================================
    // 查询
    // ========================================================================

    /// @brief 获取运行时指标
    InferenceMetrics GetMetrics() const;

    /// @brief 获取最近 N 次推理的平均延迟
    double GetRecentAvgLatencyMs() const;

    /// @brief 检查是否处于积压状态
    bool IsBacklogged() const;

    /// @brief 累计成功推理次数
    int64_t GetSuccessCount() const;

    /// @brief 累计失败次数
    int64_t GetFailureCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
