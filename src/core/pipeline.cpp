#include "core/pipeline.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "core/audio_processor.h"
#include "core/video_processor.h"
#include "core/inference_worker.h"
#include "core/render_thread.h"
#include "core/worker_registry.h"
#include "model/model_inferencer.h"
#include "model/output_processor.h"

namespace digital_human {
namespace core {

// ============================================================================
// PipelineMetrics 实现
// ============================================================================

std::string PipelineMetrics::ToString() const {
    std::ostringstream oss;
    oss << "PipelineMetrics {"
        << " frames_in=" << total_frames_in
        << " frames_out=" << total_frames_out
        << " dropped=" << frames_dropped
        << " skipped=" << frames_skipped
        << " audio_in=" << audio_packets_in
        << " video_in=" << video_packets_in
        << " inference=" << inference_count
        << " fps=" << actual_fps
        << " shutdown_ms=" << last_shutdown_ms
        << " shutdown_timeouts=" << shutdown_timeout_count
        << " av_match_misses=" << av_match_miss_count
        << " output_queue=" << queue_depths.output_frames
        << "/" << queue_peak_depths.output_frames
        << " }";
    return oss.str();
}

// ============================================================================
// Pipeline::Impl — 内部实现（使用独立类实例）
// ============================================================================

struct Pipeline::Impl {
    // ---- 配置 ----
    PipelineConfig config;
    std::atomic<bool> initialized{false};
    std::atomic<bool> running{false};
    std::atomic<bool> paused{false};
    /// 标记 Pipeline 是否已被 Stop() 终止。
    /// 终止后 Start() 会被拒绝（一次性对象语义）：
    /// 队列已 Stop 不可恢复、worker 已退出无法重启。
    std::atomic<bool> terminated{false};
    // Stop is terminal even when a finite wait times out.
    std::atomic<bool> stop_requested{false};
    mutable std::mutex lifecycle_mutex;

    // ---- 队列 ----
    ThreadSafeQueue<AudioRawPacket>       audio_raw_queue;
    ThreadSafeQueue<MelFeaturePacket>     mel_feature_queue;
    ThreadSafeQueue<VideoFramePacket>     video_raw_queue;
    ThreadSafeQueue<ProcessedFacePacket>  processed_face_queue;
    ThreadSafeQueue<InferenceTask>        inference_task_queue;
    ThreadSafeQueue<InferenceOutputPacket> inference_output_queue;
    ThreadSafeQueue<OutputFramePacket>     output_frame_queue;

    // ---- 独立处理线程实例 ----
    std::unique_ptr<AudioProcessor>      audio_processor;
    std::unique_ptr<VideoProcessor>      video_processor;
    std::unique_ptr<InferenceWorker>     inference_worker;
    std::unique_ptr<RenderThread>        render_thread;

    // ---- 音视频匹配线程（读取 mel + face，生成 InferenceTask） ----
    // 该线程负责 PTS 匹配 + 人脸缓存，是必要组件，不重复任何独立类
    std::unique_ptr<ThreadBase> matcher_thread;

    // Non-owning lifecycle registry. Worker unique_ptrs remain the owners.
    WorkerRegistry worker_registry;

    // ---- 外部模块（由 Pipeline 管理生命周期） ----
    model::ModelInferencer   model_inferencer;
    model::OutputProcessor   output_processor;

    // ---- 同步模块（用于指标查询，RenderThread 使用内置 FrameScheduler） ----
    AVSync         av_sync;
    FrameScheduler frame_scheduler;

    // ---- 指标统计 ----
    // 输入计数由 Pipeline 本对象维护（PushAudio/PushVideo 写入）。
    // 输出/丢弃/推理计数从各 worker 的 GetMetrics() 实时聚合，避免重复维护
    // 一套未接线的原子计数（旧实现的 bug：始终为 0）。
    std::atomic<int64_t> total_frames_in{0};
    std::atomic<int64_t> audio_packets_in{0};
    std::atomic<int64_t> video_packets_in{0};

    // ---- 性能计数（由 MatcherThread 在消费 mel/face 包时累加 cost_ms） ----
    // 使用 int64_t 微秒避免 atomic<double> 不支持 fetch_add 的问题。
    std::atomic<int64_t> total_audio_process_us{0};
    std::atomic<int64_t> audio_process_count{0};
    std::atomic<int64_t> total_video_process_us{0};
    std::atomic<int64_t> video_process_count{0};

    std::atomic<int64_t> lifecycle_transition_count{0};
    std::atomic<int64_t> shutdown_attempt_count{0};
    std::atomic<int64_t> shutdown_timeout_count{0};
    std::atomic<int64_t> last_shutdown_us{0};
    std::atomic<int64_t> max_shutdown_us{0};

    std::atomic<int64_t> av_match_count{0};
    std::atomic<int64_t> av_match_miss_count{0};
    std::atomic<int64_t> total_av_match_error_us{0};
    std::atomic<int64_t> max_av_match_error_us{0};

    // ---- 输入标记 ----
    std::atomic<bool> audio_eos{false};
    std::atomic<bool> video_eos{false};

    // ---- 累计已处理的音频样本数（用于 GetAudioClockMs） ----
    std::atomic<int64_t> total_audio_samples{0};

    // ---- 启动时间 ----
    std::chrono::steady_clock::time_point start_time;

    // ========================================================================
    // 构造函数：队列先以无界占位，Init() 中按 config 重建
    // ========================================================================

    Impl() = default;

    static void UpdateAtomicMax(std::atomic<int64_t>& target, int64_t value) {
        int64_t current = target.load(std::memory_order_relaxed);
        while (current < value
               && !target.compare_exchange_weak(
                   current, value, std::memory_order_relaxed)) {
        }
    }

    void RecordLifecycleTransition() {
        lifecycle_transition_count.fetch_add(1, std::memory_order_relaxed);
    }

    // ========================================================================
    // 按 config 重建所有队列（容量来自 PipelineConfig）
    // ========================================================================

    void InitQueuesFromConfig() {
        // ThreadSafeQueue 支持 move-assignment，这里以 config 中指定的容量
        // 重新构造队列并移动赋值给成员。原队列（无界占位）被丢弃。
        // 0 容量表示无界，保留向后兼容；非 0 容量则有界，可防止推理慢或
        // 输入过快时吃光内存。
        audio_raw_queue = ThreadSafeQueue<AudioRawPacket>(
            static_cast<size_t>(std::max(0, config.audio_raw_queue_size)),
            "audio_raw_queue");
        mel_feature_queue = ThreadSafeQueue<MelFeaturePacket>(
            static_cast<size_t>(std::max(0, config.mel_queue_size)),
            "mel_feature_queue");
        video_raw_queue = ThreadSafeQueue<VideoFramePacket>(
            static_cast<size_t>(std::max(0, config.video_raw_queue_size)),
            "video_raw_queue");
        processed_face_queue = ThreadSafeQueue<ProcessedFacePacket>(
            static_cast<size_t>(std::max(0, config.face_queue_size)),
            "processed_face_queue");
        // 推理任务队列复用 infer_queue_size 配置
        inference_task_queue = ThreadSafeQueue<InferenceTask>(
            static_cast<size_t>(std::max(0, config.infer_queue_size)),
            "inference_task_queue");
        inference_output_queue = ThreadSafeQueue<InferenceOutputPacket>(
            static_cast<size_t>(std::max(0, config.infer_queue_size)),
            "inference_output_queue");
        output_frame_queue = ThreadSafeQueue<OutputFramePacket>(
            static_cast<size_t>(std::max(0, config.output_queue_size)),
            "output_frame_queue");
    }

    // ========================================================================
    // 初始化核心模块
    // ========================================================================

    bool InitCoreModules() {
        SyncConfig sync_cfg;
        sync_cfg.audio_sample_rate = config.audio_sample_rate;
        sync_cfg.sync_threshold_ms = config.sync_threshold_ms;
        sync_cfg.max_drift_ms      = config.max_drift_ms;
        av_sync.Init(sync_cfg);

        SchedulerConfig sched_cfg;
        sched_cfg.target_fps       = config.target_fps;
        sched_cfg.smoothing_factor = 0.5;
        sched_cfg.enable_smoothing = true;
        frame_scheduler.Init(sched_cfg);

        return true;
    }

    // ========================================================================
    // 创建独立工作线程
    // ========================================================================

    void CreateWorkers() {

        // ---- AudioProcessor (队列模式) ----
        audio_processor = std::make_unique<AudioProcessor>("AudioProcessor");
        AudioProcessorConfig ap_cfg;
        ap_cfg.sample_rate = config.audio_sample_rate;
        ap_cfg.frame_size  = config.audio_frame_size;
        ap_cfg.hop_size    = config.audio_hop_size;
        ap_cfg.channels    = config.audio_channels;
        audio_processor->SetConfig(ap_cfg);
        audio_processor->SetInputQueue(&audio_raw_queue);
        audio_processor->SetOutputQueue(&mel_feature_queue);

        // ---- VideoProcessor ----
        video_processor = std::make_unique<VideoProcessor>("VideoProcessor");
        VideoProcessorConfig vp_cfg;
        vp_cfg.face_size = config.face_size;
        video_processor->SetConfig(vp_cfg);
        video_processor->SetInputQueue(&video_raw_queue);
        video_processor->SetOutputQueue(&processed_face_queue);

        // ---- 音视频匹配线程（mel ↔ face PTS 匹配 + 缓存） ----
        matcher_thread = CreateMatcherThread();

        // ---- InferenceWorker ----
        inference_worker = std::make_unique<InferenceWorker>("InferenceWorker");
        InferenceWorkerConfig iw_cfg;
        iw_cfg.pop_timeout_ms = config.pop_timeout_ms;
        inference_worker->SetConfig(iw_cfg);
        inference_worker->SetModelInferencer(&model_inferencer);
        inference_worker->SetInputQueue(&inference_task_queue);
        inference_worker->SetOutputQueue(&inference_output_queue);

        // ---- RenderThread（融合 + 同步 + 调度 + 输出） ----
        render_thread = std::make_unique<RenderThread>("RenderThread");
        RenderConfig rt_cfg;
        rt_cfg.target_fps         = config.target_fps;
        rt_cfg.sync_threshold_ms  = config.sync_threshold_ms;
        rt_cfg.max_drift_ms       = config.max_drift_ms;
        rt_cfg.pop_timeout_ms     = config.pop_timeout_ms;
        rt_cfg.drain_max_frames   = 30;
        rt_cfg.enable_frame_pacing = config.enable_frame_pacing;
        rt_cfg.enable_audio_sync  = true;
        render_thread->SetConfig(rt_cfg);
        render_thread->SetOutputProcessor(&output_processor);
        render_thread->SetInputQueue(&inference_output_queue);
        render_thread->SetOutputQueue(&output_frame_queue);

        worker_registry.Clear();
        worker_registry.Add("RenderThread", render_thread.get());
        worker_registry.Add("InferenceWorker", inference_worker.get());
        worker_registry.Add("AVMatcher", matcher_thread.get());
        worker_registry.Add("VideoProcessor", video_processor.get());
        worker_registry.Add("AudioProcessor", audio_processor.get());
    }

    // ========================================================================
    // 音视频匹配线程（face 驱动 + mel 时序窗装配）
    // ========================================================================

    /// @brief 匹配线程：每个人脸帧装配一个 80×N mel 时序窗，生成推理任务
    ///
    /// Wav2Lip 要求每个输出视频帧对应一个 (mel_bins × mel_step_size) 的
    /// mel 时序窗（默认 80×16 = 160ms 上下文），窗口起点 = face_pts / hop_ms。
    ///
    /// 旧实现为 mel 驱动（逐 10ms 包匹配人脸），存在两个致命问题：
    ///   1) 帧率：每个 mel 包触发一次推理 → 任务量放大 ~4×，推理积压丢帧；
    ///   2) 口型：单帧特征喂入时序卷积模型 → 口型驱动退化。
    /// 此处改为视频帧驱动 + 滑动窗口装配 + 滚动上下文归一化，
    /// 与离线参考实现（video_output_test）行为一致。
    struct MatcherThread : public ThreadBase {
        Impl& ctx;

        // ---- mel 行滚动缓冲（dB 域 log-mel，每行 1×mel_bins） ----
        std::deque<cv::Mat> mel_rows_;          ///< 滚动缓冲（含过去上下文）
        int64_t mel_first_seq_ = 0;             ///< 缓冲首行的全局 mel 帧序号
        int64_t mel_next_seq_  = 0;             ///< 下一个待入缓冲的序号
        bool    audio_eos_     = false;         ///< 音频流已结束

        // ---- 配置缓存 ----
        double hop_ms_   = 10.0;                ///< mel 帧间隔（毫秒）
        int    window_   = 16;                  ///< mel 时序窗长度（帧）
        int    context_  = 300;                 ///< 归一化滚动上下文长度（帧）
        int    mel_bins_ = 80;                  ///< mel 滤波器组数

        explicit MatcherThread(Impl& ctx)
            : ThreadBase("AVMatcher"), ctx(ctx) {}

        void Run() override {
            LogInfo("启动 (face-driven)");

            hop_ms_  = 1000.0 * ctx.config.audio_hop_size
                     / ctx.config.audio_sample_rate;
            window_  = std::max(1, ctx.config.mel_window_frames);
            context_ = std::max(window_, ctx.config.mel_context_frames);
            // mel_bins 与 AudioProcessor 默认配置一致（80 滤波器组）

            const int kPopTimeoutMs = 100;

            while (!IsStopping()) {
                // ---- 1. 取人脸帧（视频帧 = 输出节拍） ----
                ProcessedFacePacket face_pkt;
                if (!ctx.processed_face_queue.WaitAndPop(
                        face_pkt, kPopTimeoutMs)) {
                    // MarkVideoEOS only means that no more raw frames will be
                    // submitted. VideoProcessor may still be converting queued
                    // BGR frames, so an empty processed queue is not terminal.
                    // Wait for the explicit ProcessedFacePacket::EOS emitted
                    // after VideoProcessor has drained its input queue.
                    continue;
                }

                if (face_pkt.header.IsEOS()) {
                    LogInfo("人脸流结束");
                    ctx.inference_task_queue.Push(InferenceTask::EOS());
                    break;
                }
                if (face_pkt.header.IsFatal()) {
                    ctx.inference_task_queue.Push(InferenceTask::Fatal());
                    break;
                }
                if (face_pkt.header.IsSkip()) {
                    continue;
                }

                // 累加视频处理耗时（face 包的 cost_ms 由 VideoProcessor 写入）
                if (face_pkt.header.cost_ms > 0.0) {
                    ctx.total_video_process_us.fetch_add(
                        static_cast<int64_t>(face_pkt.header.cost_ms * 1000.0),
                        std::memory_order_relaxed);
                    ctx.video_process_count.fetch_add(
                        1, std::memory_order_relaxed);
                }

                // ---- 2. 该视频帧对应的 mel 窗口起始序号 ----
                int64_t mel_start = static_cast<int64_t>(std::llround(
                    static_cast<double>(face_pkt.header.pts_ms) / hop_ms_));

                // ---- 3. 填充缓冲直到覆盖 [mel_start, mel_start+window) ----
                FillMelBuffer(mel_start);

                // 音频耗尽且缓冲已空 → 无音频可配 → 结束
                if (mel_rows_.empty()) {
                    if (audio_eos_) {
                        LogInfo("音频耗尽，提前结束匹配");
                        ctx.inference_task_queue.Push(InferenceTask::EOS());
                        break;
                    }
                    continue;
                }

                // ---- 4. 在可用 Mel 范围内选择最近窗口并校验 PTS 容差 ----
                const int64_t available_last_seq = mel_first_seq_
                    + static_cast<int64_t>(mel_rows_.size()) - 1;
                const int64_t available_last_start = std::max(
                    mel_first_seq_, available_last_seq - window_ + 1);
                const int64_t matched_start = std::clamp(
                    mel_start, mel_first_seq_, available_last_start);
                const double matched_pts_ms = matched_start * hop_ms_;
                const double match_error_ms = std::abs(
                    static_cast<double>(face_pkt.header.pts_ms) - matched_pts_ms);
                const int64_t match_error_us = static_cast<int64_t>(
                    std::llround(match_error_ms * 1000.0));

                ctx.av_match_count.fetch_add(1, std::memory_order_relaxed);
                ctx.total_av_match_error_us.fetch_add(
                    match_error_us, std::memory_order_relaxed);
                Impl::UpdateAtomicMax(ctx.max_av_match_error_us, match_error_us);

                if (match_error_ms > ctx.config.av_match_threshold_ms) {
                    ctx.av_match_miss_count.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                mel_start = matched_start;

                // ---- 5. 装配 window×80 窗口（边界 clamp，与参考实现一致） ----
                cv::Mat win(window_, mel_bins_, CV_32F);
                for (int f = 0; f < window_; ++f) {
                    int64_t idx = mel_start + f;
                    const cv::Mat* row = nullptr;
                    if (idx <= mel_first_seq_) {
                        row = &mel_rows_.front();   // 人脸滞后：钳到最早可用帧
                    } else if (idx - mel_first_seq_
                               < static_cast<int64_t>(mel_rows_.size())) {
                        row = &mel_rows_[idx - mel_first_seq_];
                    } else {
                        row = &mel_rows_.back();    // 音频末尾：钳到最后一帧
                    }
                    row->copyTo(win.row(f));
                }
                // ---- 6. Mel 已由提取器按 Wav2Lip 规范归一化到 [-4, 4] ----

                // ---- 7. 丢弃窗口之前的旧行 + 裁剪上下文容量 ----
                while (mel_first_seq_ < mel_start && !mel_rows_.empty()) {
                    mel_rows_.pop_front();
                    ++mel_first_seq_;
                }
                while (static_cast<int>(mel_rows_.size()) > context_) {
                    mel_rows_.pop_front();
                    ++mel_first_seq_;
                }

                // ---- 8. 生成推理任务（PTS 以视频帧为准 —— 输出时间轴） ----
                MelFeaturePacket mel_pkt;
                mel_pkt.header.pts_ms = face_pkt.header.pts_ms;
                mel_pkt.header.seq_id = face_pkt.header.seq_id;
                mel_pkt.header.status = StatusCode::OK;
                mel_pkt.payload = std::move(win);

                InferenceTask task;
                task.mel  = std::move(mel_pkt);
                task.face = std::move(face_pkt);
                ctx.inference_task_queue.Push(std::move(task));
            }

            LogInfo("退出");
        }

        /// @brief 从 mel 队列拉取特征行，直到缓冲覆盖目标窗口或音频结束
        void FillMelBuffer(int64_t mel_start) {
            const int kPopTimeoutMs = 100;
            while (!IsStopping() && !audio_eos_
                   && mel_next_seq_ < mel_start + window_) {
                MelFeaturePacket mel_pkt;
                if (!ctx.mel_feature_queue.WaitAndPop(
                        mel_pkt, kPopTimeoutMs)) {
                    // 上游已标记 EOS 且队列空 → 音频结束
                    if (ctx.audio_eos.load(std::memory_order_acquire)
                        && ctx.mel_feature_queue.Empty()) {
                        audio_eos_ = true;
                        break;
                    }
                    continue;
                }
                if (mel_pkt.header.IsEOS() || mel_pkt.header.IsFatal()) {
                    audio_eos_ = true;
                    break;
                }
                if (mel_pkt.header.IsSkip() || mel_pkt.payload.empty()) {
                    continue;
                }

                // 累加音频处理耗时
                if (mel_pkt.header.cost_ms > 0.0) {
                    ctx.total_audio_process_us.fetch_add(
                        static_cast<int64_t>(mel_pkt.header.cost_ms * 1000.0),
                        std::memory_order_relaxed);
                    ctx.audio_process_count.fetch_add(
                        1, std::memory_order_relaxed);
                }

                AppendMelPacket(mel_pkt);
            }
        }

        /// @brief 将 mel 包的各行按全局序号追加到滚动缓冲（缺口自动填补）
        void AppendMelPacket(const MelFeaturePacket& mel_pkt) {
            const cv::Mat& m = mel_pkt.payload;
            if (m.empty()) return;

            // 行优先约定：rows=时间帧, cols=mel_bins
            cv::Mat rows_mat;
            if (m.cols == mel_bins_) {
                rows_mat = m;
            } else if (m.rows == mel_bins_) {
                rows_mat = m.t();   // 防御：转置输入自动纠正
            } else {
                LogInfo("mel 包尺寸异常，丢弃");
                return;
            }

            // 以包 seq_id 重同步（AudioProcessor 逐 hop 连续发射）
            int64_t pkt_seq = mel_pkt.header.seq_id;
            if (pkt_seq < mel_next_seq_) {
                pkt_seq = mel_next_seq_;    // 过期包：按当前位置追加
            }
            // 缺口填补（SKIP/丢帧）：重复最后一行保持时间轴连续
            while (mel_next_seq_ < pkt_seq) {
                if (mel_rows_.empty()) {
                    mel_rows_.emplace_back(
                        cv::Mat::zeros(1, mel_bins_, CV_32F));
                } else {
                    mel_rows_.push_back(mel_rows_.back());
                }
                ++mel_next_seq_;
            }
            for (int r = 0; r < rows_mat.rows; ++r) {
                mel_rows_.push_back(rows_mat.row(r).clone());
                ++mel_next_seq_;
            }
        }
    };

    /// @brief 创建匹配线程实例
    std::unique_ptr<ThreadBase> CreateMatcherThread() {
        return std::make_unique<MatcherThread>(*this);
    }
};

// ============================================================================
// Pipeline 公有接口实现
// ============================================================================

Pipeline::Pipeline()
    : impl_(std::make_unique<Impl>()) {}

Pipeline::~Pipeline() {
    if (!Stop()) {
        // The finite API timeout has expired. Destruction still owns every
        // worker and must join them before worker-specific members disappear.
        impl_->worker_registry.WaitAll();
        impl_->terminated.store(true, std::memory_order_release);
        impl_->RecordLifecycleTransition();
    }
}

bool Pipeline::Init(const PipelineConfig& config) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);

    if (impl_->initialized.load(std::memory_order_acquire)) {
        std::cerr << "[Pipeline] Init: duplicate initialization" << std::endl;
        return false;
    }
    if (impl_->terminated.load(std::memory_order_acquire)
        || impl_->stop_requested.load(std::memory_order_acquire)) {
        std::cerr << "[Pipeline] Init: Pipeline is terminal and cannot be reused"
                  << std::endl;
        return false;
    }

    // Reject values that would cause division by zero, invalid queue sizes,
    // or ambiguous shutdown semantics.
    if (config.audio_sample_rate <= 0 || config.audio_channels <= 0
        || config.audio_frame_size <= 0 || config.audio_hop_size <= 0
        || config.audio_hop_size > config.audio_frame_size
        || config.face_size <= 0 || config.target_fps <= 0.0
        || config.sync_threshold_ms < 0.0 || config.max_drift_ms < 0.0
        || config.max_drift_ms < config.sync_threshold_ms
        || config.av_match_threshold_ms < 0.0
        || config.mel_window_frames <= 0
        || config.mel_context_frames < config.mel_window_frames
        || config.audio_raw_queue_size < 0 || config.mel_queue_size < 0
        || config.video_raw_queue_size < 0 || config.face_queue_size < 0
        || config.infer_queue_size < 0 || config.output_queue_size < 0
        || config.pop_timeout_ms < 0 || config.shutdown_timeout_ms < 0) {
        std::cerr << "[Pipeline] Init: invalid configuration" << std::endl;
        return false;
    }

    impl_->config = config;

    // 限制 OpenCV 并行线程数：本 SDK 的 OpenCV 操作均为小图（96×96~ROI 几百像素），
    // 线程过多同步开销大于收益，且会与 ncnn 推理线程争抢 CPU 核
    if (config.opencv_num_threads > 0) {
        cv::setNumThreads(config.opencv_num_threads);
    }

    // 按 config 重建队列容量（替代旧的硬编码 0/60/30/10）
    impl_->InitQueuesFromConfig();

    if (!impl_->InitCoreModules()) {
        return false;
    }

    impl_->CreateWorkers();
    impl_->initialized.store(true, std::memory_order_release);
    impl_->RecordLifecycleTransition();

    std::cout << "[Pipeline] 初始化成功: "
              << config.audio_sample_rate << "Hz, "
              << config.target_fps << "fps" << std::endl;
    return true;
}

bool Pipeline::Start() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);

    if (!impl_->initialized.load(std::memory_order_acquire)) {
        std::cerr << "[Pipeline] Start: not initialized" << std::endl;
        return false;
    }
    if (impl_->terminated.load(std::memory_order_acquire)
        || impl_->stop_requested.load(std::memory_order_acquire)) {
        std::cerr << "[Pipeline] Start: Pipeline is terminal, restart rejected"
                  << std::endl;
        return false;
    }
    if (impl_->running.load(std::memory_order_acquire)) {
        return true;
    }

    impl_->start_time = std::chrono::steady_clock::now();

    if (!impl_->worker_registry.StartAll()) {
        impl_->running.store(false, std::memory_order_release);
        impl_->stop_requested.store(true, std::memory_order_release);
        impl_->terminated.store(true, std::memory_order_release);
        impl_->RecordLifecycleTransition();
        std::cerr << "[Pipeline] Start: worker startup failed; Pipeline terminated"
                  << std::endl;
        return false;
    }

    impl_->running.store(true, std::memory_order_release);
    impl_->RecordLifecycleTransition();
    std::cout << "[Pipeline] started (" << impl_->worker_registry.Size()
              << " workers)" << std::endl;
    return true;
}

bool Pipeline::Stop() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);

    if (!impl_->initialized.load(std::memory_order_acquire)) {
        return true;
    }
    if (impl_->terminated.load(std::memory_order_acquire)) {
        return true;
    }

    const auto shutdown_start = std::chrono::steady_clock::now();
    impl_->shutdown_attempt_count.fetch_add(1, std::memory_order_relaxed);

    const bool first_request = !impl_->stop_requested.exchange(
        true, std::memory_order_acq_rel);
    if (first_request) {
        impl_->RecordLifecycleTransition();
    }
    std::cout << "[Pipeline] stopping..." << std::endl;
    impl_->running.store(false, std::memory_order_release);

    if (impl_->audio_processor) {
        impl_->audio_processor->MarkEOS();
    }
    if (impl_->video_processor) {
        impl_->video_processor->MarkInputEOS();
    }

    impl_->audio_raw_queue.Stop();
    impl_->mel_feature_queue.Stop();
    impl_->video_raw_queue.Stop();
    impl_->processed_face_queue.Stop();
    impl_->inference_task_queue.Stop();
    impl_->inference_output_queue.Stop();
    impl_->output_frame_queue.Stop();

    impl_->worker_registry.RequestStopAll();
    const auto report = impl_->worker_registry.WaitAllFor(
        impl_->config.shutdown_timeout_ms);
    const int64_t elapsed_us = static_cast<int64_t>(std::llround(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - shutdown_start).count()));
    impl_->last_shutdown_us.store(elapsed_us, std::memory_order_relaxed);
    Impl::UpdateAtomicMax(impl_->max_shutdown_us, elapsed_us);

    if (report.all_stopped) {
        impl_->terminated.store(true, std::memory_order_release);
        impl_->RecordLifecycleTransition();
        std::cout << "[Pipeline] stopped in "
                  << static_cast<double>(elapsed_us) / 1000.0 << "ms" << std::endl;
        return true;
    }

    impl_->shutdown_timeout_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "[Pipeline] Stop: shutdown timeout after "
              << static_cast<double>(elapsed_us) / 1000.0
              << "ms; workers remain owned" << std::endl;
    for (const auto& worker : report.workers) {
        if (!worker.stopped) {
            std::cerr << "[Pipeline] Stop: worker '" << worker.name
                      << "' did not stop within the shared deadline"
                      << std::endl;
        }
    }
    return false;
}

bool Pipeline::IsRunning() const {
    return impl_->running.load(std::memory_order_acquire);
}

// ========================================================================
// 数据输入
// ========================================================================

bool Pipeline::PushAudio(const std::vector<float>& pcm_data, int64_t pts_ms) {
    if (!impl_->initialized.load(std::memory_order_acquire)
        || impl_->stop_requested.load(std::memory_order_acquire)
        || !impl_->running.load(std::memory_order_acquire)) {
        return false;
    }
    impl_->audio_packets_in.fetch_add(1, std::memory_order_relaxed);
    impl_->total_audio_samples.fetch_add(pcm_data.size(),
        std::memory_order_relaxed);
    auto pkt = AudioRawPacket::Make(pcm_data, pts_ms,
                                    impl_->audio_packets_in.load());
    return impl_->audio_raw_queue.Push(std::move(pkt));
}

bool Pipeline::PushVideo(const cv::Mat& frame, int64_t pts_ms) {
    if (!impl_->initialized.load(std::memory_order_acquire)
        || impl_->stop_requested.load(std::memory_order_acquire)
        || !impl_->running.load(std::memory_order_acquire)) {
        return false;
    }
    impl_->video_packets_in.fetch_add(1, std::memory_order_relaxed);
    impl_->total_frames_in.fetch_add(1, std::memory_order_relaxed);
    auto pkt = VideoFramePacket::Make(frame.clone(), pts_ms,
                                      impl_->video_packets_in.load());
    return impl_->video_raw_queue.Push(std::move(pkt));
}

void Pipeline::MarkAudioEOS() {
    if (!impl_->initialized.load(std::memory_order_acquire) || !impl_->audio_processor) {
        return;
    }
    impl_->audio_eos.store(true, std::memory_order_release);
    impl_->audio_processor->MarkEOS();
}

void Pipeline::MarkVideoEOS() {
    if (!impl_->initialized.load(std::memory_order_acquire) || !impl_->video_processor) {
        return;
    }
    impl_->video_eos.store(true, std::memory_order_release);
    impl_->video_processor->MarkInputEOS();
}

// ========================================================================
// 数据输出
// ========================================================================

bool Pipeline::GetOutputFrame(OutputFramePacket& frame, int timeout_ms) {
    return impl_->output_frame_queue.WaitAndPop(frame, timeout_ms);
}

// ========================================================================
// 控制
// ========================================================================

void Pipeline::Pause() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    if (!impl_->running.load(std::memory_order_acquire)) {
        return;
    }
    if (!impl_->paused.exchange(true, std::memory_order_acq_rel)) {
        impl_->RecordLifecycleTransition();
        std::cout << "[Pipeline] paused" << std::endl;
    }
}

void Pipeline::Resume() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    if (!impl_->running.load(std::memory_order_acquire)) {
        return;
    }
    if (impl_->paused.exchange(false, std::memory_order_acq_rel)) {
        impl_->RecordLifecycleTransition();
        std::cout << "[Pipeline] resumed" << std::endl;
    }
}

bool Pipeline::IsPaused() const {
    return impl_->paused.load(std::memory_order_acquire);
}

// ========================================================================
// 查询
// ========================================================================

PipelineMetrics Pipeline::GetMetrics() const {
    PipelineMetrics metrics;
    metrics.total_frames_in = impl_->total_frames_in.load(std::memory_order_relaxed);
    metrics.audio_packets_in = impl_->audio_packets_in.load(std::memory_order_relaxed);
    metrics.video_packets_in = impl_->video_packets_in.load(std::memory_order_relaxed);

    const int64_t audio_count = impl_->audio_process_count.load(
        std::memory_order_relaxed);
    if (audio_count > 0) {
        metrics.avg_audio_process_ms = static_cast<double>(
            impl_->total_audio_process_us.load(std::memory_order_relaxed))
            / audio_count / 1000.0;
    }
    const int64_t video_count = impl_->video_process_count.load(
        std::memory_order_relaxed);
    if (video_count > 0) {
        metrics.avg_video_process_ms = static_cast<double>(
            impl_->total_video_process_us.load(std::memory_order_relaxed))
            / video_count / 1000.0;
    }

    if (impl_->inference_worker) {
        const auto inference_metrics = impl_->inference_worker->GetMetrics();
        metrics.inference_count = inference_metrics.total_inferences;
        metrics.frames_skipped = inference_metrics.total_failures
                               + inference_metrics.skipped_due_to_backlog;
        metrics.avg_inference_ms = inference_metrics.avg_latency_ms;
    }

    if (impl_->render_thread) {
        const auto render_metrics = impl_->render_thread->GetMetrics();
        metrics.total_frames_out = render_metrics.frames_displayed;
        metrics.frames_dropped = render_metrics.frames_dropped;
        metrics.avg_output_ms = render_metrics.avg_render_ms;
        metrics.actual_fps = impl_->render_thread->GetFrameStats().actual_fps;
    }

    metrics.lifecycle_transition_count = impl_->lifecycle_transition_count.load(
        std::memory_order_relaxed);
    metrics.shutdown_attempt_count = impl_->shutdown_attempt_count.load(
        std::memory_order_relaxed);
    metrics.shutdown_timeout_count = impl_->shutdown_timeout_count.load(
        std::memory_order_relaxed);
    metrics.last_shutdown_ms = static_cast<double>(
        impl_->last_shutdown_us.load(std::memory_order_relaxed)) / 1000.0;
    metrics.max_shutdown_ms = static_cast<double>(
        impl_->max_shutdown_us.load(std::memory_order_relaxed)) / 1000.0;

    metrics.av_match_count = impl_->av_match_count.load(std::memory_order_relaxed);
    metrics.av_match_miss_count = impl_->av_match_miss_count.load(
        std::memory_order_relaxed);
    if (metrics.av_match_count > 0) {
        metrics.avg_av_match_error_ms = static_cast<double>(
            impl_->total_av_match_error_us.load(std::memory_order_relaxed))
            / metrics.av_match_count / 1000.0;
    }
    metrics.max_av_match_error_ms = static_cast<double>(
        impl_->max_av_match_error_us.load(std::memory_order_relaxed)) / 1000.0;

    const auto audio_raw = impl_->audio_raw_queue.GetMetrics();
    const auto mel_features = impl_->mel_feature_queue.GetMetrics();
    const auto video_raw = impl_->video_raw_queue.GetMetrics();
    const auto processed_faces = impl_->processed_face_queue.GetMetrics();
    const auto inference_tasks = impl_->inference_task_queue.GetMetrics();
    const auto inference_output = impl_->inference_output_queue.GetMetrics();
    const auto output_frames = impl_->output_frame_queue.GetMetrics();

    metrics.queue_depths.audio_raw = static_cast<int64_t>(audio_raw.current_size);
    metrics.queue_depths.mel_features = static_cast<int64_t>(
        mel_features.current_size);
    metrics.queue_depths.video_raw = static_cast<int64_t>(video_raw.current_size);
    metrics.queue_depths.processed_faces = static_cast<int64_t>(
        processed_faces.current_size);
    metrics.queue_depths.inference_tasks = static_cast<int64_t>(
        inference_tasks.current_size);
    metrics.queue_depths.inference_output = static_cast<int64_t>(
        inference_output.current_size);
    metrics.queue_depths.output_frames = static_cast<int64_t>(
        output_frames.current_size);

    metrics.queue_peak_depths.audio_raw = static_cast<int64_t>(audio_raw.peak_size);
    metrics.queue_peak_depths.mel_features = static_cast<int64_t>(
        mel_features.peak_size);
    metrics.queue_peak_depths.video_raw = static_cast<int64_t>(video_raw.peak_size);
    metrics.queue_peak_depths.processed_faces = static_cast<int64_t>(
        processed_faces.peak_size);
    metrics.queue_peak_depths.inference_tasks = static_cast<int64_t>(
        inference_tasks.peak_size);
    metrics.queue_peak_depths.inference_output = static_cast<int64_t>(
        inference_output.peak_size);
    metrics.queue_peak_depths.output_frames = static_cast<int64_t>(
        output_frames.peak_size);

    return metrics;
}

FrameStats Pipeline::GetFrameStats() const {
    if (impl_->render_thread) {
        return impl_->render_thread->GetFrameStats();
    }
    return FrameStats();
}

SyncStatus Pipeline::GetSyncStatus() const {
    // 从 RenderThread 的 FrameScheduler 推断同步状态
    double drift = GetDriftMs();
    if (std::abs(drift) >= impl_->config.max_drift_ms) {
        return SyncStatus::SEVERE_OFFSET;
    }
    if (drift > impl_->config.sync_threshold_ms) {
        return SyncStatus::VIDEO_AHEAD;
    }
    if (drift < -impl_->config.sync_threshold_ms) {
        return SyncStatus::VIDEO_BEHIND;
    }
    return SyncStatus::SYNCED;
}

double Pipeline::GetDriftMs() const {
    if (impl_->render_thread) {
        return impl_->render_thread->GetDriftMs();
    }
    return 0.0;
}

double Pipeline::GetAudioClockMs() const {
    // 基于累计输入音频样本数估算音频时钟
    if (impl_->config.audio_sample_rate <= 0) return 0.0;
    int64_t samples = impl_->total_audio_samples.load(std::memory_order_relaxed);
    return static_cast<double>(samples) / impl_->config.audio_sample_rate * 1000.0;
}

void Pipeline::SetLandmarkModelPath(const std::string& path) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    if (!impl_->running.load(std::memory_order_acquire)
        && !impl_->stop_requested.load(std::memory_order_acquire)
        && impl_->video_processor) {
        impl_->video_processor->SetLandmarkModelPath(path);
    }
}

bool Pipeline::LoadFaceModel(const std::string& path) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return !impl_->running.load(std::memory_order_acquire)
        && !impl_->stop_requested.load(std::memory_order_acquire)
        && impl_->video_processor
        && impl_->video_processor->LoadFaceModel(path);
}

bool Pipeline::IsFaceModelLoaded() const {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return impl_->video_processor
        && impl_->video_processor->IsFaceModelLoaded();
}

bool Pipeline::InitModelInferencer(const std::string& model_dir) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return !impl_->running.load(std::memory_order_acquire)
        && !impl_->stop_requested.load(std::memory_order_acquire)
        && impl_->model_inferencer.Init(model_dir);
}

bool Pipeline::InitModelInferencer(const std::string& param_path,
                                   const std::string& bin_path) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return !impl_->running.load(std::memory_order_acquire)
        && !impl_->stop_requested.load(std::memory_order_acquire)
        && impl_->model_inferencer.Init(param_path, bin_path);
}

bool Pipeline::EnableGPU(bool enable) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return !impl_->running.load(std::memory_order_acquire)
        && !impl_->stop_requested.load(std::memory_order_acquire)
        && impl_->model_inferencer.EnableGPU(enable);
}

bool Pipeline::IsGPUEnabled() const {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return impl_->model_inferencer.IsGPUEnabled();
}

void Pipeline::SetCalibrationDumpDirectory(const std::string& directory,
                                           size_t max_samples) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    if (!impl_->running.load(std::memory_order_acquire)
        && !impl_->stop_requested.load(std::memory_order_acquire)
        && impl_->inference_worker) {
        impl_->inference_worker->SetCalibrationDumpDirectory(directory, max_samples);
    }
}

void Pipeline::SetInferenceThreads(int n) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    if (n > 0 && !impl_->running.load(std::memory_order_acquire)
        && !impl_->stop_requested.load(std::memory_order_acquire)) {
        impl_->model_inferencer.SetThreadCount(n);
    }
}

}  // namespace core
}  // namespace digital_human
