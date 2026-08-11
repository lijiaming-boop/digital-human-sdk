#include "core/inference_worker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <mat.h>
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

bool WriteFloat32Npy(const std::filesystem::path& path, const ncnn::Mat& mat) {
    if (mat.empty() || mat.elemsize != 4u) return false;
    const int channels = std::max(1, mat.c);
    const int height = std::max(1, mat.h);
    const int width = std::max(1, mat.w);
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        + std::to_string(channels) + ", " + std::to_string(height) + ", "
        + std::to_string(width) + "), }";
    const size_t prefix_size = 10;
    const size_t padding = (16 - ((prefix_size + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > 65535) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    const uint16_t header_size = static_cast<uint16_t>(header.size());
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(mat.data),
              static_cast<std::streamsize>(mat.total() * mat.elemsize));
    return static_cast<bool>(out);
}

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

    // ---- 重试队列（用于失败重试，deque 保证前端 O(1) 删除） ----
    std::deque<InferenceTask> retry_queue_;

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
    std::atomic<bool> input_eos_{false};

    // ---- 积压检测 ----
    std::atomic<bool> backlogged_{false};
    int64_t last_backlog_check_ = 0;

    // ---- 最近延迟滑动窗口 ----
    std::vector<double> recent_latencies_;
    size_t recent_max_samples_ = 200;

    // ---- 人脸张量缓存（拟合场景：同一对齐人脸反复推理，仅在 worker 线程访问） ----
    mutable ncnn::Mat     cached_face_tensor_;
    mutable const uchar*  cached_face_ptr_ = nullptr;
    mutable int           cached_face_w_   = 0;
    mutable int           cached_face_h_   = 0;
    std::filesystem::path calibration_dump_dir_;
    size_t calibration_dump_limit_ = 0;
    size_t calibration_dump_count_ = 0;

    // ========================================================================
    // 张量转换
    // ========================================================================

    /**
     * @brief Mel 频谱 cv::Mat → ncnn::Mat
     *
     * 模型输入布局（与 ModelInferencer warmup 一致）：
     *   ncnn::Mat(w=时间帧数, h=mel_bins, c=1)，row(bin)[t] = mel(t, bin)
     *
     * 旧实现产出 (w=mel_bins, h=T)，与模型期望布局转置相反 ——
     * 时序卷积在 mel bin 维度上滑动，输入退化，口型驱动失效。
     */
    ncnn::Mat MelToNCNN(const cv::Mat& mel) const {
        if (mel.empty()) return ncnn::Mat();

        int T        = mel.rows;   ///< 时间帧数
        int mel_bins = mel.cols;   ///< mel bins

        ncnn::Mat out(T, mel_bins, 1);
        for (int b = 0; b < mel_bins; ++b) {
            float* dst = out.channel(0).row(b);
            for (int t = 0; t < T; ++t) {
                dst[t] = mel.at<float>(t, b);
            }
        }
        return out;
    }

    /**
     * @brief 对齐人脸 cv::Mat → ncnn::Mat (6通道)，Wav2Lip 标准格式
     *
     *   ch0-2: 下半脸遮罩人脸（y >= h/2 置零）—— 强制模型依据音频
     *          重建嘴部区域。旧实现将完整人脸复制两份，模型可直接
     *          复制输入嘴部（走捷径），口型不随音频变化。
     *   ch3-5: 完整人脸
     *
     * 向量化实现（convertTo + split + memcpy），替代逐像素 at<> 循环。
     * 同一 aligned_face 数据指针命中缓存时直接复用
     * （静态图片拟合场景 VideoProcessor 缓存使指针恒定，~100% 命中）。
     */
    ncnn::Mat FaceToNCNN(const cv::Mat& face_in) const {
        if (face_in.empty()) return ncnn::Mat();

        // 规范化为 3 通道 BGR
        cv::Mat face;
        if (face_in.channels() == 3) {
            face = face_in;
        } else if (face_in.channels() == 1) {
            cv::cvtColor(face_in, face, cv::COLOR_GRAY2BGR);
        } else if (face_in.channels() == 4) {
            cv::cvtColor(face_in, face, cv::COLOR_BGRA2BGR);
        } else {
            return ncnn::Mat();
        }

        const int w = face.cols;
        const int h = face.rows;

        // 缓存命中：同一图像数据（VideoProcessor 人脸缓存复用同一 Mat）
        if (cached_face_ptr_ == face_in.data
            && cached_face_w_ == w && cached_face_h_ == h
            && !cached_face_tensor_.empty()) {
            return cached_face_tensor_;
        }

        // 转 float [0,1] 并拆通道
        cv::Mat f32;
        face.convertTo(f32, CV_32FC3, 1.0 / 255.0);
        cv::Mat bgr[3];
        cv::split(f32, bgr);   // bgr[0]=B, [1]=G, [2]=R

        ncnn::Mat out(w, h, 6);
        const int mask_from = h / 2;   // 下半脸起点（y >= 48 置零）
        const size_t row_bytes = static_cast<size_t>(w) * sizeof(float);

        for (int c = 0; c < 3; ++c) {
            const float* src = bgr[2 - c].ptr<float>();   // c=0→R, 1→G, 2→B
            float* dst_masked = out.channel(c).row(0);
            float* dst_full   = out.channel(c + 3).row(0);
            // 上半脸：遮罩通道拷入真实像素
            std::memcpy(dst_masked, src, row_bytes * mask_from);
            // 下半脸：遮罩通道置零（ncnn::Mat 非零初始化，必须显式清零）
            std::memset(dst_masked + static_cast<size_t>(mask_from) * w,
                        0, row_bytes * (h - mask_from));
            // 完整人脸：全图拷贝
            std::memcpy(dst_full, src, row_bytes * h);
        }

        // 写缓存
        cached_face_tensor_ = out;
        cached_face_ptr_    = face_in.data;
        cached_face_w_      = w;
        cached_face_h_      = h;
        return out;
    }

    void DumpCalibrationInputs(const ncnn::Mat& audio, const ncnn::Mat& face) {
        if (calibration_dump_dir_.empty()
            || calibration_dump_count_ >= calibration_dump_limit_) {
            return;
        }
        const size_t index = calibration_dump_count_;
        const auto audio_path = calibration_dump_dir_ / "audio"
                              / (std::to_string(index) + ".npy");
        const auto face_path = calibration_dump_dir_ / "face"
                             / (std::to_string(index) + ".npy");
        if (!WriteFloat32Npy(audio_path, audio) || !WriteFloat32Npy(face_path, face)) {
            std::cerr << "[InferenceWorker] failed to dump INT8 calibration sample "
                      << index << std::endl;
            return;
        }
        std::ofstream audio_list(calibration_dump_dir_ / "audio.list", std::ios::app);
        std::ofstream face_list(calibration_dump_dir_ / "face.list", std::ios::app);
        if (!audio_list || !face_list) {
            std::cerr << "[InferenceWorker] failed to update calibration list files" << std::endl;
            return;
        }
        audio_list << std::filesystem::absolute(audio_path).string() << '\n';
        face_list << std::filesystem::absolute(face_path).string() << '\n';
        ++calibration_dump_count_;
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

        DumpCalibrationInputs(audio_ncnn, face_ncnn);

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

void InferenceWorker::SetCalibrationDumpDirectory(const std::string& directory,
                                                  size_t max_samples) {
    impl_->calibration_dump_dir_.clear();
    impl_->calibration_dump_limit_ = 0;
    impl_->calibration_dump_count_ = 0;
    if (directory.empty() || max_samples == 0) return;

    const std::filesystem::path root(directory);
    std::error_code ec;
    std::filesystem::create_directories(root / "audio", ec);
    std::filesystem::create_directories(root / "face", ec);
    if (ec) {
        std::cerr << "[InferenceWorker] cannot create calibration directory: "
                  << root << " (" << ec.message() << ")" << std::endl;
        return;
    }
    std::ofstream(root / "audio.list", std::ios::trunc).close();
    std::ofstream(root / "face.list", std::ios::trunc).close();
    impl_->calibration_dump_dir_ = root;
    impl_->calibration_dump_limit_ = max_samples;
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
            impl_->retry_queue_.pop_front();

            ncnn::Mat output;
            double latency_ms = 0.0;
            bool ok = impl_->DoInfer(task, output, latency_ms);

            impl_->UpdateStats(latency_ms, ok);

            if (ok) {
                // 成功: 推送到输出队列（携带人脸数据用于下游融合）
                InferenceOutputPacket pkt;
                pkt.InheritHeader(task.mel.header);
                pkt.payload.model_output = output;
                pkt.payload.face_data    = task.face.payload;
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
            // 成功（携带人脸数据用于下游融合流水线）
            InferenceOutputPacket pkt;
            pkt.InheritHeader(task.mel.header);
            pkt.payload.model_output = output;
            pkt.payload.face_data    = task.face.payload;
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
    impl_->input_eos_.store(true, std::memory_order_release);
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
