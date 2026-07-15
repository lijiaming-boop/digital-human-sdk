#include "core/inference_worker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <ncnn/mat.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "model/model_inferencer.h"

namespace digital_human {
namespace core {

using model::ModelInferencer;

// ============================================================================
// 常量
// ============================================================================

/// @brief EWMA 平滑因子
static constexpr double kEwmaAlpha = 0.3;

/// @brief 指标滑动窗口大小
static constexpr int kMetricsWindowSize = 100;

// ============================================================================
// InferenceMetrics 实现
// ============================================================================

std::string InferenceMetrics::ToString() const {
    std::ostringstream oss;
    oss << "InferenceMetrics {"
        << " total=" << total_inferences
        << " success=" << total_success
        << " fail=" << total_failures
        << " retry=" << total_retries
        << " backlog_skip=" << skipped_due_to_backlog
        << " avg=" << avg_latency_ms << "ms"
        << " ewma=" << ewma_latency_ms << "ms"
        << " queue=" << input_queue_depth
        << " backlog=" << (backlog_warning ? "yes" : "no")
        << " }";
    return oss.str();
}

// ============================================================================
// Impl 结构体
// ============================================================================

struct InferenceWorker::Impl {
    // ---- 配置 ----
    InferenceWorkerConfig config;

    // ---- 模型 ----
    ModelInferencer* model_ = nullptr;

    // ---- 队列 ----
    ThreadSafeQueue<InferenceTask>*         input_queue_  = nullptr;
    ThreadSafeQueue<InferenceOutputPacket>* output_queue_ = nullptr;

    // ---- 重试队列（用于失败重试） ----
    std::vector<InferenceTask> retry_queue_;

    // ---- 统计 ----
    std::atomic<int64_t> total_inferences_{0};
    std::atomic<int64_t> total_success_{0};
    std::atomic<int64_t> total_failures_{0};
    std::atomic<int64_t> total_retries_{0};
    std::atomic<int64_t> skipped_backlog_{0};

    // ---- 延迟统计 ----
    mutable std::mutex stats_mutex_;
    double      min_latency_ms_      = std::numeric_limits<double>::max();
    double      max_latency_ms_      = 0.0;
    double      sum_latency_ms_      = 0.0;
    int64_t     latency_sample_count_ = 0;
    double      ewma_latency_ms_     = 0.0;

    // ---- 输入结束标记 ----
    bool input_eos_ = false;

    // ---- 积压检测 ----
    std::atomic<bool> backlogged_{false};
    int64_t last_backlog_check_ = 0;

    // ---- 最近延迟滑动窗口 ----
    std::vector<double> recent_latencies_;
    size_t recent_max_samples_ = 200;

    // ========================================================================
    // 张量转换
    // ========================================================================

    /**
     * @brief Mel 频谱 cv::Mat → ncnn::Mat
     *
     * cv::Mat: (rows=T, cols=mel_bins, type=CV_32F)
     * ncnn::Mat: (w=mel_bins, h=T, c=1)
     */
    ncnn::Mat MelToNCNN(const cv::Mat& mel) const {
        if (mel.empty()) return ncnn::Mat();

        int mel_bins = mel.cols;
        int T        = mel.rows;

        ncnn::Mat out(mel_bins, T, 1);
        // 逐行拷贝: cv::Mat 行优先 → ncnn::Mat (w, h, c)
        for (int y = 0; y < T; ++y) {
            const float* src_row = mel.ptr<float>(y);
            float* dst_row = out.channel(0).row(y);
            std::memcpy(dst_row, src_row, mel_bins * sizeof(float));
        }
        return out;
    }

    /**
     * @brief 对齐人脸 cv::Mat → ncnn::Mat (6通道)
     *
     * Wav2Lip 要求 6 通道输入：前 3 通道 = 人脸 BGR，
     * 后 3 通道 = 副本（对口型优化）。
     */
    ncnn::Mat FaceToNCNN(const cv::Mat& face) const {
        if (face.empty()) return ncnn::Mat();

        int w = face.cols;
        int h = face.rows;

        if (face.channels() == 3) {
            // 单帧: BGR → RGB 并复制为 6 通道
            ncnn::Mat out(w, h, 6);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    cv::Vec3b bgr = face.at<cv::Vec3b>(y, x);
                    // 前 3 通道: RGB
                    out.channel(0).row(y)[x] = static_cast<float>(bgr[2]) / 255.0f;
                    out.channel(1).row(y)[x] = static_cast<float>(bgr[1]) / 255.0f;
                    out.channel(2).row(y)[x] = static_cast<float>(bgr[0]) / 255.0f;
                    // 后 3 通道: 副本
                    out.channel(3).row(y)[x] = static_cast<float>(bgr[2]) / 255.0f;
                    out.channel(4).row(y)[x] = static_cast<float>(bgr[1]) / 255.0f;
                    out.channel(5).row(y)[x] = static_cast<float>(bgr[0]) / 255.0f;
                }
            }
            return out;
        }

        if (face.channels() == 6) {
            // 已经是 6 通道，直接 float 化
            ncnn::Mat out(w, h, 6);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    for (int c = 0; c < 6; ++c) {
                        out.channel(c).row(y)[x] =
                            face.at<cv::Vec<float, 6>>(y, x)[c];
                    }
                }
            }
            return out;
        }

        // 其他格式
        cv::Mat rgb;
        cv::cvtColor(face, rgb, cv::COLOR_BGR2RGB);
        ncnn::Mat out(w, h, 3);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                cv::Vec3b rgb_px = rgb.at<cv::Vec3b>(y, x);
                out.channel(0).row(y)[x] = static_cast<float>(rgb_px[0]) / 255.0f;
                out.channel(1).row(y)[x] = static_cast<float>(rgb_px[1]) / 255.0f;
                out.channel(2).row(y)[x] = static_cast<float>(rgb_px[2]) / 255.0f;
            }
        }
        return out;
    }

    // ========================================================================
    // 队列积压检测
    // ========================================================================

    /// @brief 检查输入队列深度，输出警告
    void CheckBacklog(int64_t now_ms) {
        if (last_backlog_check_ == 0) {
            last_backlog_check_ = now_ms;
            return;
        }

        if (now_ms - last_backlog_check_ < config.backlog_check_interval) {
            return;
        }
        last_backlog_check_ = now_ms;

        if (!input_queue_) return;

        int depth = static_cast<int>(input_queue_->Size());
        // 加上重试队列中的任务
        depth += static_cast<int>(retry_queue_.size());

        if (depth >= config.input_queue_error_threshold) {
            backlogged_.store(true, std::memory_order_release);
            std::cerr << "[InferenceWorker] 严重积压: 输入队列深度="
                      << depth << " (阈值=" << config.input_queue_error_threshold
                      << ")" << std::endl;
        } else if (depth >= config.input_queue_warn_threshold) {
            backlogged_.store(true, std::memory_order_release);
            std::cout << "[InferenceWorker] 积压警告: 输入队列深度="
                      << depth << " (阈值=" << config.input_queue_warn_threshold
                      << ")" << std::endl;
        } else {
            backlogged_.store(false, std::memory_order_release);
        }
    }

    // ========================================================================
    // 推理执行
    // ========================================================================

    /**
     * @brief 执行单次推理
     *
     * @param task  推理任务
     * @param[out] output 推理输出
     * @param[out] latency_ms 推理延迟
     * @return true  推理成功
     * @return false 推理失败
     */
    bool DoInfer(const InferenceTask& task,
                 ncnn::Mat& output, double& latency_ms) {
        if (!model_ || !model_->IsInitialized()) {
            // 无模型时返回 false（测试模式下触发重试机制）
            std::cerr << "[InferenceWorker] 模型未初始化，返回失败"
                      << " pts=" << task.mel.header.pts_ms << "ms"
                      << std::endl;
            return false;
        }

        // 张量转换
        ncnn::Mat audio_ncnn = MelToNCNN(task.mel.payload);
        ncnn::Mat face_ncnn  = FaceToNCNN(task.face.payload.aligned_face);

        if (audio_ncnn.empty() || face_ncnn.empty()) {
            std::cerr << "[InferenceWorker] 张量转换失败: "
                      << "audio_empty=" << audio_ncnn.empty()
                      << " face_empty=" << face_ncnn.empty()
                      << std::endl;
            return false;
        }

        // 推理 + 计时
        auto t0 = std::chrono::steady_clock::now();
        output = model_->Infer(audio_ncnn, face_ncnn);
        auto t1 = std::chrono::steady_clock::now();

        latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (output.empty()) {
            std::cerr << "[InferenceWorker] 推理返回空输出"
                      << " pts=" << task.mel.header.pts_ms << "ms"
                      << " seq=" << task.mel.header.seq_id
                      << std::endl;
            return false;
        }

        return true;
    }

    // ========================================================================
    // 更新统计
    // ========================================================================

    void UpdateStats(double latency_ms, bool success) {
        total_inferences_.fetch_add(1, std::memory_order_relaxed);
        if (success) {
            total_success_.fetch_add(1, std::memory_order_relaxed);
        } else {
            total_failures_.fetch_add(1, std::memory_order_relaxed);
        }

        // 延迟统计
        std::lock_guard<std::mutex> lock(stats_mutex_);
        sum_latency_ms_ += latency_ms;
        latency_sample_count_++;

        if (latency_ms < min_latency_ms_) min_latency_ms_ = latency_ms;
        if (latency_ms > max_latency_ms_) max_latency_ms_ = latency_ms;

        // EWMA
        if (ewma_latency_ms_ == 0.0) {
            ewma_latency_ms_ = latency_ms;
        } else {
            ewma_latency_ms_ = kEwmaAlpha * latency_ms
                             + (1.0 - kEwmaAlpha) * ewma_latency_ms_;
        }

        // 滑动窗口
        recent_latencies_.push_back(latency_ms);
        if (recent_latencies_.size() > recent_max_samples_) {
            recent_latencies_.erase(
                recent_latencies_.begin(),
                recent_latencies_.begin()
                    + (recent_latencies_.size() - recent_max_samples_));
        }
    }

    /// @brief 记录失败日志
    void LogFailure(const InferenceTask& task, const std::string& reason) {
        std::cerr << "[InferenceWorker] 推理失败"
                  << " pts=" << task.mel.header.pts_ms << "ms"
                  << " seq=" << task.mel.header.seq_id
                  << " retry=" << task.retry_count << "/" << task.kMaxRetries
                  << " 原因: " << reason
                  << std::endl;
    }
};

// ============================================================================
// 构造 / 析构
// ============================================================================

InferenceWorker::InferenceWorker(const std::string& name)
    : ThreadBase(name)
    , impl_(std::make_unique<Impl>()) {}

InferenceWorker::~InferenceWorker() {
    Stop();
}

// ============================================================================
// 配置
// ============================================================================

void InferenceWorker::SetConfig(const InferenceWorkerConfig& config) {
    impl_->config = config;
}

const InferenceWorkerConfig& InferenceWorker::GetConfig() const {
    return impl_->config;
}

// ============================================================================
// 模型
// ============================================================================

void InferenceWorker::SetModelInferencer(ModelInferencer* inferencer) {
    impl_->model_ = inferencer;
}

// ============================================================================
// 队列
// ============================================================================

void InferenceWorker::SetInputQueue(
    ThreadSafeQueue<InferenceTask>* queue) {
    impl_->input_queue_ = queue;
}

void InferenceWorker::SetOutputQueue(
    ThreadSafeQueue<InferenceOutputPacket>* queue) {
    impl_->output_queue_ = queue;
}

// ============================================================================
// 线程主循环
// ============================================================================

void InferenceWorker::Run() {
    LogInfo("[InferenceWorker] 启动");

    if (!impl_->input_queue_) {
        LogError("输入队列未设置");
        return;
    }
    if (!impl_->output_queue_) {
        LogError("输出队列未设置");
        return;
    }

    if (!impl_->model_) {
        LogInfo("模型推理器未设置，将透传所有任务");
    }

    while (!IsStopping()) {
        // ---- 1. 优先处理重试队列 ----
        if (!impl_->retry_queue_.empty()) {
            InferenceTask task = impl_->retry_queue_.front();
            impl_->retry_queue_.erase(impl_->retry_queue_.begin());

            ncnn::Mat output;
            double latency_ms = 0.0;
            bool ok = impl_->DoInfer(task, output, latency_ms);

            impl_->UpdateStats(latency_ms, ok);

            if (ok) {
                // 成功: 推送到输出队列
                InferenceOutputPacket pkt;
                pkt.InheritHeader(task.mel.header);
                pkt.payload = output;
                pkt.header.status = StatusCode::OK;
                pkt.header.cost_ms = latency_ms;
                impl_->output_queue_->Push(std::move(pkt));
                LogInfo("[InferenceWorker] 重试成功"
                        " pts=" + std::to_string(task.mel.header.pts_ms) + "ms"
                        " latency=" + std::to_string(latency_ms) + "ms");
            } else {
                // 失败: 重试或丢弃
                impl_->LogFailure(task, "推理返回空");
                if (task.CanRetry()) {
                    impl_->total_retries_.fetch_add(1, std::memory_order_relaxed);
                    impl_->retry_queue_.push_back(task.Retry());
                    LogInfo("[InferenceWorker] 加入重试队列"
                            " retry=" + std::to_string(task.retry_count + 1));
                } else {
                    LogError("重试耗尽，丢弃帧");
                }
            }
            continue;
        }

        // ---- 2. 从输入队列取任务 ----
        InferenceTask task;
        if (!impl_->input_queue_->WaitAndPop(task,
                impl_->config.pop_timeout_ms)) {
            // 超时或停止
            continue;
        }

        // ---- 3. 检查终止信号 ----
        if (task.IsEOS()) {
            LogInfo("[InferenceWorker] 收到 EOS");
            impl_->output_queue_->Push(InferenceOutputPacket::EOS());
            break;
        }
        if (task.IsFatal()) {
            LogError("收到致命错误信号");
            impl_->output_queue_->Push(InferenceOutputPacket::Fatal());
            break;
        }

        // ---- 4. 检查任务有效性 ----
        if (!task.IsValid()) {
            impl_->LogFailure(task, "任务数据无效");
            continue;
        }

        // ---- 5. 积压检测 ----
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        impl_->CheckBacklog(now);

        // ---- 6. 执行推理 ----
        ncnn::Mat output;
        double latency_ms = 0.0;
        bool ok = impl_->DoInfer(task, output, latency_ms);

        impl_->UpdateStats(latency_ms, ok);

        if (ok) {
            // 成功
            InferenceOutputPacket pkt;
            pkt.InheritHeader(task.mel.header);
            pkt.payload = output;
            pkt.header.status = StatusCode::OK;
            pkt.header.cost_ms = latency_ms;
            impl_->output_queue_->Push(std::move(pkt));
        } else {
            // 失败: 重试或丢弃
            impl_->LogFailure(task, "推理返回空");
            if (task.CanRetry()) {
                impl_->total_retries_.fetch_add(1, std::memory_order_relaxed);
                impl_->retry_queue_.push_back(task.Retry());
                LogInfo("[InferenceWorker] 加入重试队列"
                        " retry=" + std::to_string(task.retry_count + 1));
            } else {
                LogError("重试耗尽，丢弃帧"
                         " pts=" + std::to_string(task.mel.header.pts_ms) + "ms");
            }
        }
    }

    LogInfo("[InferenceWorker] 退出 (成功="
            + std::to_string(impl_->total_success_.load())
            + " 失败=" + std::to_string(impl_->total_failures_.load())
            + " 重试=" + std::to_string(impl_->total_retries_.load())
            + ")");
}

// ============================================================================
// 控制
// ============================================================================

void InferenceWorker::ResetStats() {
    impl_->total_inferences_.store(0, std::memory_order_relaxed);
    impl_->total_success_.store(0, std::memory_order_relaxed);
    impl_->total_failures_.store(0, std::memory_order_relaxed);
    impl_->total_retries_.store(0, std::memory_order_relaxed);
    impl_->skipped_backlog_.store(0, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(impl_->stats_mutex_);
    impl_->min_latency_ms_       = std::numeric_limits<double>::max();
    impl_->max_latency_ms_       = 0.0;
    impl_->sum_latency_ms_       = 0.0;
    impl_->latency_sample_count_ = 0;
    impl_->ewma_latency_ms_      = 0.0;
    impl_->recent_latencies_.clear();
}

void InferenceWorker::MarkInputEOS() {
    impl_->input_eos_ = true;
}

// ============================================================================
// 查询
// ============================================================================

InferenceMetrics InferenceWorker::GetMetrics() const {
    InferenceMetrics m;
    m.total_inferences        = impl_->total_inferences_.load(std::memory_order_relaxed);
    m.total_success           = impl_->total_success_.load(std::memory_order_relaxed);
    m.total_failures          = impl_->total_failures_.load(std::memory_order_relaxed);
    m.total_retries           = impl_->total_retries_.load(std::memory_order_relaxed);
    m.skipped_due_to_backlog  = impl_->skipped_backlog_.load(std::memory_order_relaxed);
    m.backlog_warning         = impl_->backlogged_.load(std::memory_order_acquire);

    {
        std::lock_guard<std::mutex> lock(impl_->stats_mutex_);
        if (impl_->latency_sample_count_ > 0) {
            m.avg_latency_ms = impl_->sum_latency_ms_
                             / static_cast<double>(impl_->latency_sample_count_);
        }
        m.min_latency_ms = (impl_->min_latency_ms_
                           == std::numeric_limits<double>::max())
                               ? 0.0 : impl_->min_latency_ms_;
        m.max_latency_ms      = impl_->max_latency_ms_;
        m.ewma_latency_ms     = impl_->ewma_latency_ms_;
    }

    if (impl_->input_queue_) {
        m.input_queue_depth = static_cast<int>(impl_->input_queue_->Size())
                            + static_cast<int>(impl_->retry_queue_.size());
    }
    m.retry_queue_depth = static_cast<int>(impl_->retry_queue_.size());

    return m;
}

double InferenceWorker::GetRecentAvgLatencyMs() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex_);
    if (impl_->recent_latencies_.empty()) return 0.0;
    double sum = 0.0;
    for (double v : impl_->recent_latencies_) sum += v;
    return sum / static_cast<double>(impl_->recent_latencies_.size());
}

bool InferenceWorker::IsBacklogged() const {
    return impl_->backlogged_.load(std::memory_order_acquire);
}

int64_t InferenceWorker::GetSuccessCount() const {
    return impl_->total_success_.load(std::memory_order_relaxed);
}

int64_t InferenceWorker::GetFailureCount() const {
    return impl_->total_failures_.load(std::memory_order_relaxed);
}

}  // namespace core
}  // namespace digital_human
