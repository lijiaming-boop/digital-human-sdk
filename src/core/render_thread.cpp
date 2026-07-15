#include "core/render_thread.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

#include <opencv2/core.hpp>
#include <ncnn/mat.h>

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
    ThreadSafeQueue<RenderPacket>*       input_queue_  = nullptr;
    ThreadSafeQueue<OutputFramePacket>*  output_queue_ = nullptr;

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
    bool    input_eos_      = false;

    // ---- 上一帧缓存（用于 DUPLICATE） ----
    cv::Mat last_frame_;

    // ---- 帧间隔调节 ----
    std::chrono::steady_clock::time_point last_frame_time_;
    bool   has_last_frame_time_ = false;
    double target_interval_ms_  = 40.0;  // 25fps

    // ---- 渲染计时 ----
    double total_render_ms_     = 0.0;
    double total_sync_wait_ms_  = 0.0;
    int64_t render_count_       = 0;

    // ========================================================================
    // 核心渲染方法
    // ========================================================================

    /**
     * @brief 渲染一帧：融合模型输出到原始背景
     */
    cv::Mat RenderFrame(const RenderTaskData& data) {
        if (!output_processor_) {
            // 无 OutputProcessor 时直接返回原始人脸
            return data.original_face.clone();
        }

        return output_processor_->Process(
            data.model_output,
            data.original_face,
            data.face_mask,
            data.M_inv);
    }

    /**
     * @brief 音频同步判定
     *
     * @return FrameAction 调度决策
     */
    FrameAction GetSyncAction(int64_t pts_ms) {
        if (!config.enable_audio_sync || !audio_player_) {
            return FrameAction::DISPLAY;
        }

        double audio_clock_ms = audio_player_->GetPlaybackPositionMs();
        double video_pts_ms   = static_cast<double>(pts_ms);
        double drift          = video_pts_ms - audio_clock_ms;
        double threshold      = config.sync_threshold_ms;
        double max_drift      = config.max_drift_ms;

        if (std::abs(drift) >= max_drift) {
            std::cout << "[RenderThread] 严重偏移: drift="
                      << drift << "ms, DROP" << std::endl;
            return FrameAction::DROP;
        }
        if (drift > threshold) {
            std::cout << "[RenderThread] 视频超前: drift="
                      << drift << "ms, DUPLICATE" << std::endl;
            return FrameAction::DUPLICATE;
        }
        if (drift < -threshold) {
            std::cout << "[RenderThread] 视频滞后: drift="
                      << drift << "ms, DROP" << std::endl;
            return FrameAction::DROP;
        }

        return FrameAction::DISPLAY;
    }

    /**
     * @brief 帧间隔调节：等待到下一帧的理想显示时间
     */
    void PaceFrame() {
        if (!config.enable_frame_pacing || !has_last_frame_time_) {
            last_frame_time_ = std::chrono::steady_clock::now();
            has_last_frame_time_ = true;
            return;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(
            now - last_frame_time_).count();
        double wait_ms = target_interval_ms_ - elapsed;

        if (wait_ms > 0) {
            auto start = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(wait_ms)));
            total_sync_wait_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        }

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
    Stop();
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

void RenderThread::SetInputQueue(ThreadSafeQueue<RenderPacket>* queue) {
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
        RenderPacket pkt;
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
            RenderPacket remaining;
            while (impl_->input_queue_->TryPop(remaining)) {
                if (remaining.header.IsSkip() || !remaining.payload.IsValid()) {
                    continue;
                }
                auto t0 = std::chrono::steady_clock::now();
                cv::Mat result = impl_->RenderFrame(remaining.payload);
                impl_->frames_rendered_++;
                impl_->last_frame_ = result.clone();
                if (impl_->frame_callback_) {
                    impl_->frame_callback_(result, remaining.header.pts_ms,
                                           impl_->frame_id_);
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

        // ---- 音频同步 ----
        FrameAction action = impl_->GetSyncAction(pkt.header.pts_ms);

        // ---- 执行帧决策 ----
        switch (action) {
            case FrameAction::DISPLAY: {
                auto t0 = std::chrono::steady_clock::now();

                // 渲染融合
                cv::Mat result = impl_->RenderFrame(pkt.payload);
                impl_->frames_rendered_++;
                impl_->render_count_++;

                double render_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                impl_->total_render_ms_ += render_ms;

                // 帧间隔调节
                impl_->PaceFrame();

                // 缓存上一帧
                impl_->last_frame_ = result.clone();

                // 推送输出
                if (impl_->output_queue_) {
                    OutputFramePacket out;
                    out.InheritHeader(pkt.header);
                    out.payload = result.clone();
                    out.header.cost_ms = render_ms;
                    impl_->output_queue_->Push(std::move(out));
                }

                // 回调
                if (impl_->frame_callback_) {
                    impl_->frame_callback_(
                        result, pkt.header.pts_ms, impl_->frame_id_);
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
                // WAIT 在 GetSyncAction 中不会产生
                break;
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
    impl_->input_eos_ = true;
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

}  // namespace core
}  // namespace digital_human
