#include "core/render_thread.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

#include <opencv2/core.hpp>
#include <mat.h>

#include "audio/audio_player.h"
#include "model/output_processor.h"

namespace digital_human {
namespace core {

using audio::AudioPlayer;
using model::OutputProcessor;

// ============================================================================
// RenderMetrics 实现
// ============================================================================

std::string RenderMetrics::ToString() const {
    std::ostringstream oss;
    oss << "RenderMetrics {"
        << " rendered=" << frames_rendered
        << " displayed=" << frames_displayed
        << " dropped=" << frames_dropped
        << " duplicated=" << frames_duplicated
        << " fps=" << actual_fps
        << " render=" << avg_render_ms << "ms"
        << " sync_wait=" << avg_sync_wait_ms << "ms"
        << " queue=" << output_queue_depth
        << " }";
    return oss.str();
}

// ============================================================================
// Impl 结构体
// ============================================================================

struct RenderThread::Impl {
    // ---- 配置 ----
    RenderConfig config;

    // ---- 外部模块 ----
    OutputProcessor*               output_processor_ = nullptr;
    AudioPlayer*                   audio_player_     = nullptr;

    // ---- 队列 ----
    ThreadSafeQueue<InferenceOutputPacket>* input_queue_  = nullptr;
    ThreadSafeQueue<OutputFramePacket>*     output_queue_ = nullptr;

    // ---- 帧调度 ----
    FrameScheduler frame_scheduler_;

    // ---- 帧回调 ----
    FrameCallback frame_callback_;

    // ---- 状态 ----
    int64_t frame_id_       = 0;
    int64_t frames_rendered_ = 0;
    int64_t frames_displayed_ = 0;
    int64_t frames_dropped_   = 0;
    int64_t frames_duplicated_ = 0;
    std::atomic<bool> input_eos_{false};

    // ---- 上一帧缓存（用于 DUPLICATE） ----
    cv::Mat last_frame_;

    // ---- 帧间隔调节 ----
    std::chrono::steady_clock::time_point last_frame_time_;
    bool   has_last_frame_time_ = false;
    double target_interval_ms_  = 33.333333;  // 30fps 默认值，SetConfig 会按 target_fps 重算

    // ---- 渲染计时 ----
    double total_render_ms_     = 0.0;
    double total_sync_wait_ms_  = 0.0;
    int64_t render_count_       = 0;

    // ---- 漂移追踪（供 Pipeline 查询） ----
    std::atomic<double> last_drift_ms_{0.0};

    // ========================================================================
    // 核心渲染方法
    // ========================================================================

    /**
     * @brief 从 InferenceOutputPacket 构建 RenderTaskData
     */
    RenderTaskData MakeRenderData(const InferenceOutputData& data) {
        RenderTaskData rd;
        rd.model_output  = data.model_output;
        rd.original_face = data.face_data.original_face;
        rd.M_inv         = data.face_data.M_inv;
        rd.face_mask     = data.face_data.face_mask;
        rd.face_rect     = data.face_data.face_rect;
        return rd;
    }

    /**
     * @brief 渲染一帧：融合模型输出到原始背景
     *
     * 使用 ROI 加速路径（ProcessROI）：逆变换/融合/锐化仅在人脸区域
     * 内执行，大图下的渲染耗时降低一个数量级。
     */
    cv::Mat RenderFrame(const RenderTaskData& data) {
        if (!output_processor_) {
            return data.original_face.clone();
        }

        return output_processor_->ProcessROI(
            data.model_output,
            data.original_face,
            data.face_mask,
            data.M_inv,
            data.face_rect);
    }

    /**
     * @brief 综合同步判定：FrameScheduler + 音频漂移检查
     *
     * 使用 FrameScheduler 做基于 PTS 的帧率调度（含 EMA 平滑），
     * 叠加音频时钟驱动漂移修正（audio-aware DROP/DUPLICATE）。
     *
     * @return FrameAction 调度决策
     */
    FrameAction GetSyncAction(int64_t pts_ms, int frame_id) {
        // 1. 基础调度：FrameScheduler 基于 PTS 做帧率控制
        ScheduleResult sched = frame_scheduler_.ScheduleFrame(
            frame_id, static_cast<double>(pts_ms));

        // 2. 如果未启用音频同步，直接返回 FrameScheduler 的决策
        if (!config.enable_audio_sync || !audio_player_) {
            return sched.action;
        }

        // 3. 音频漂移修正
        double audio_clock_ms = audio_player_->GetPlaybackPositionMs();
        double video_pts_ms   = static_cast<double>(pts_ms);
        double drift          = video_pts_ms - audio_clock_ms;
        last_drift_ms_.store(drift, std::memory_order_relaxed);

        if (std::abs(drift) >= config.max_drift_ms) {
            std::cout << "[RenderThread] 严重偏移: drift="
                      << drift << "ms, DROP" << std::endl;
            return FrameAction::DROP;
        }
        if (drift > config.sync_threshold_ms) {
            std::cout << "[RenderThread] 视频超前: drift="
                      << drift << "ms, DUPLICATE" << std::endl;
            return FrameAction::DUPLICATE;
        }
        if (drift < -config.sync_threshold_ms) {
            std::cout << "[RenderThread] 视频滞后: drift="
                      << drift << "ms, DROP" << std::endl;
            return FrameAction::DROP;
        }

        return sched.action;
    }

    /**
     * @brief 帧间隔调节：等待到下一帧的理想显示时间
     *
     * 高精度实现：
     *   - 以 steady_clock 绝对时间点为基准，避免 sleep_for 截断误差累积
     *   - 等待拆为两段：先 sleep_until 到目标前 1ms，末段 yield 忙等
     *     以规避 Windows 默认 ~15ms 调度 tick 的过睡问题
     *   - 目标间隔用微秒表达，无整除损失
     */
    void PaceFrame() {
        const auto now = std::chrono::steady_clock::now();

        if (!config.enable_frame_pacing || !has_last_frame_time_) {
            last_frame_time_ = now;
            has_last_frame_time_ = true;
            return;
        }

        // 计算下一帧的目标呈现时间点（基于上一帧时间 + 目标间隔）
        const auto interval_us = std::chrono::microseconds(
            static_cast<int64_t>(target_interval_ms_ * 1000.0));
        const auto target_time = last_frame_time_ + interval_us;

        if (now < target_time) {
            auto wait_start = std::chrono::steady_clock::now();

            // 末段 1ms 切换为忙等，规避 OS 调度 tick 误差
            const auto busy_wait_threshold = target_time -
                std::chrono::milliseconds(1);

            if (now < busy_wait_threshold) {
                // 粗等到目标前 1ms
                std::this_thread::sleep_until(busy_wait_threshold);
            }
            // 末段忙等：yield 让出 CPU 但快速响应
            while (std::chrono::steady_clock::now() < target_time) {
                std::this_thread::yield();
            }

            total_sync_wait_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wait_start).count();
        }

        // 更新基准时间。若已落后（now > target_time），从当前时间重新计起，
        // 避免追赶式连发破坏下游节奏。
        last_frame_time_ = std::chrono::steady_clock::now();
    }
};

// ============================================================================
// 构造 / 析构
// ============================================================================

RenderThread::RenderThread(const std::string& name)
    : ThreadBase(name)
    , impl_(std::make_unique<Impl>()) {}

RenderThread::~RenderThread() {
    Shutdown();
}

// ============================================================================
// 配置
// ============================================================================

void RenderThread::SetConfig(const RenderConfig& config) {
    impl_->config = config;
    impl_->target_interval_ms_ = 1000.0 / config.target_fps;

    SchedulerConfig sc;
    sc.target_fps        = config.target_fps;
    sc.smoothing_factor  = 0.5;
    sc.enable_smoothing  = true;
    impl_->frame_scheduler_.Init(sc);
}

const RenderConfig& RenderThread::GetConfig() const {
    return impl_->config;
}

// ============================================================================
// 模块注入
// ============================================================================

void RenderThread::SetOutputProcessor(OutputProcessor* processor) {
    impl_->output_processor_ = processor;
}

void RenderThread::SetAudioPlayer(AudioPlayer* player) {
    impl_->audio_player_ = player;
}

// ============================================================================
// 队列
// ============================================================================

void RenderThread::SetInputQueue(ThreadSafeQueue<InferenceOutputPacket>* queue) {
    impl_->input_queue_ = queue;
}

void RenderThread::SetOutputQueue(
    ThreadSafeQueue<OutputFramePacket>* queue) {
    impl_->output_queue_ = queue;
}

// ============================================================================
// 回调
// ============================================================================

void RenderThread::SetFrameCallback(FrameCallback callback) {
    impl_->frame_callback_ = std::move(callback);
}

// ============================================================================
// 线程主循环
// ============================================================================

void RenderThread::Run() {
    LogInfo("[RenderThread] 启动");

    if (!impl_->input_queue_) {
        LogError("输入队列未设置");
        return;
    }

    while (!IsStopping()) {
        // ---- 取帧 ----
        InferenceOutputPacket pkt;
        if (!impl_->input_queue_->WaitAndPop(
                pkt, impl_->config.pop_timeout_ms)) {
            if (IsStopping()) break;
            continue;
        }

        // ---- 终止信号 ----
        if (pkt.header.IsEOS()) {
            LogInfo("[RenderThread] 收到 EOS");
            // 排空输入队列中剩余的帧
            int drained = 0;
            InferenceOutputPacket remaining;
            while (impl_->input_queue_->TryPop(remaining)) {
                if (remaining.header.IsSkip() ||
                    !remaining.payload.IsValid()) {
                    continue;
                }
                auto t0 = std::chrono::steady_clock::now();
                auto rdata = impl_->MakeRenderData(remaining.payload);
                cv::Mat result = impl_->RenderFrame(rdata);
                impl_->frames_rendered_++;
                // 浅拷贝共享：result 即将被回调消费，无后续写入者
                impl_->last_frame_ = result;
                if (impl_->frame_callback_) {
                    impl_->frame_callback_(
                        result, remaining.header.pts_ms, impl_->frame_id_);
                }
                impl_->frames_displayed_++;
                impl_->frame_id_++;
                drained++;
                if (drained >= impl_->config.drain_max_frames) break;
            }
            if (drained > 0) {
                LogInfo("[RenderThread] 排空 " + std::to_string(drained) + " 帧");
            }
            if (impl_->output_queue_) {
                impl_->output_queue_->Push(OutputFramePacket::EOS());
            }
            break;
        }
        if (pkt.header.IsFatal()) {
            LogError("收到致命错误信号");
            if (impl_->output_queue_) {
                impl_->output_queue_->Push(OutputFramePacket::Fatal());
            }
            break;
        }
        if (pkt.header.IsSkip() || !pkt.payload.IsValid()) {
            continue;
        }

        // ---- 转换为 RenderTaskData ----
        auto render_data = impl_->MakeRenderData(pkt.payload);

        // ---- 综合同步判定（FrameScheduler + 音频漂移） ----
        FrameAction action = impl_->GetSyncAction(
            pkt.header.pts_ms, impl_->frame_id_);

        // ---- 执行帧决策 ----
        switch (action) {
            case FrameAction::DISPLAY: {
                auto t0 = std::chrono::steady_clock::now();

                // 渲染融合（完整流水线：逆变换 → 融合 → 锐化 → 色彩混合）
                cv::Mat result = impl_->RenderFrame(render_data);
                impl_->frames_rendered_++;
                impl_->render_count_++;

                double render_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                impl_->total_render_ms_ += render_ms;

                // 通知 FrameScheduler 帧已显示（更新 EMA）
                impl_->frame_scheduler_.OnFrameDisplayed(
                    static_cast<double>(pkt.header.pts_ms));

                // 帧间隔调节
                impl_->PaceFrame();

                // 缓存上一帧用于 DUPLICATE：浅拷贝共享数据
                // cv::Mat 引用计数为原子操作，COW 在写时自动触发；
                // 回调签名是 const cv::Mat& 不会修改，
                // 输出队列消费者修改时 COW 会自动分离内存。
                impl_->last_frame_ = result;

                // 先回调（此时 result 仍有效）
                if (impl_->frame_callback_) {
                    impl_->frame_callback_(
                        result, pkt.header.pts_ms, impl_->frame_id_);
                }

                // 推送输出（move 语义，O(1) 引用计数交换）
                if (impl_->output_queue_) {
                    OutputFramePacket out;
                    out.InheritHeader(pkt.header);
                    out.payload = result;  // move 语义，无深度拷贝
                    out.header.cost_ms = render_ms;
                    impl_->output_queue_->Push(std::move(out));
                }

                impl_->frames_displayed_++;
                impl_->frame_id_++;
                break;
            }

            case FrameAction::DROP:
                impl_->frames_dropped_++;
                break;

            case FrameAction::DUPLICATE:
                if (!impl_->last_frame_.empty()) {
                    if (impl_->frame_callback_) {
                        impl_->frame_callback_(
                            impl_->last_frame_,
                            pkt.header.pts_ms,
                            impl_->frame_id_);
                    }
                    impl_->frames_displayed_++;
                    impl_->frames_duplicated_++;
                }
                impl_->frame_id_++;
                break;

            case FrameAction::WAIT:
            {
                // 等待后重试：把数据放回队列（模拟重入）
                int wait_ms = static_cast<int>(
                    impl_->frame_scheduler_.GetFrameIntervalMs());
                if (wait_ms > 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(
                            std::min(wait_ms, 50)));
                }
                // 包放回输入队列等待下轮处理
                impl_->input_queue_->Push(std::move(pkt));
                continue;
            }
        }
    }

    LogInfo("[RenderThread] 退出 (渲染="
            + std::to_string(impl_->frames_rendered_)
            + " 显示=" + std::to_string(impl_->frames_displayed_)
            + " 丢弃=" + std::to_string(impl_->frames_dropped_)
            + " 重复=" + std::to_string(impl_->frames_duplicated_)
            + ")");
}

// ============================================================================
// 控制
// ============================================================================

void RenderThread::MarkInputEOS() {
    impl_->input_eos_.store(true, std::memory_order_release);
}

void RenderThread::Reset() {
    impl_->frame_id_         = 0;
    impl_->frames_rendered_  = 0;
    impl_->frames_displayed_ = 0;
    impl_->frames_dropped_   = 0;
    impl_->frames_duplicated_ = 0;
    impl_->render_count_     = 0;
    impl_->total_render_ms_  = 0.0;
    impl_->total_sync_wait_ms_ = 0.0;
    impl_->last_frame_       = cv::Mat();
    impl_->has_last_frame_time_ = false;
    impl_->frame_scheduler_.Reset();
}

// ============================================================================
// 查询
// ============================================================================

RenderMetrics RenderThread::GetMetrics() const {
    RenderMetrics m;
    m.frames_rendered    = impl_->frames_rendered_;
    m.frames_displayed   = impl_->frames_displayed_;
    m.frames_dropped     = impl_->frames_dropped_;
    m.frames_duplicated  = impl_->frames_duplicated_;
    m.frame_pacing_active = impl_->config.enable_frame_pacing;

    if (impl_->render_count_ > 0) {
        m.avg_render_ms = impl_->total_render_ms_
                         / static_cast<double>(impl_->render_count_);
        m.avg_sync_wait_ms = impl_->total_sync_wait_ms_
                            / static_cast<double>(impl_->render_count_);
    }

    // 帧率从 FrameScheduler 获取
    auto stats = impl_->frame_scheduler_.GetStats();
    m.actual_fps = stats.actual_fps;

    if (impl_->input_queue_) {
        m.output_queue_depth = static_cast<int>(impl_->input_queue_->Size());
    }

    return m;
}

FrameStats RenderThread::GetFrameStats() const {
    return impl_->frame_scheduler_.GetStats();
}

int64_t RenderThread::GetRenderedCount() const {
    return impl_->frames_rendered_;
}

double RenderThread::GetDriftMs() const {
    return impl_->last_drift_ms_.load(std::memory_order_acquire);
}

}  // namespace core
}  // namespace digital_human
