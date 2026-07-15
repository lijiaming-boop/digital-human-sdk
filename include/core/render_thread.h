#pragma once

#include <memory>
#include <atomic>
#include <cstdint>
#include <functional>

#include "core/thread_base.h"
#include "core/thread_safe_queue.h"
#include "core/packet.h"
#include "core/frame_scheduler.h"

namespace digital_human {
namespace audio {
class AudioPlayer;
}
namespace model {
class OutputProcessor;
}

namespace core {

// ============================================================================
// 渲染线程配置
// ============================================================================

/// @brief 渲染线程配置
struct RenderConfig {
    double target_fps              = 25.0;   ///< 目标帧率
    double sync_threshold_ms       = 30.0;   ///< 同步阈值（毫秒）
    double max_drift_ms            = 100.0;  ///< 最大允许漂移（毫秒）
    int    render_queue_warn       = 5;      ///< 渲染队列深度警告阈值
    int    drain_max_frames        = 30;     ///< 退出时最多排空帧数
    int    pop_timeout_ms          = 100;    ///< 队列弹出超时（毫秒）
    bool   enable_frame_pacing     = true;   ///< 是否启用帧间隔调节
    bool   enable_audio_sync       = true;   ///< 是否启用音频同步
    bool   enable_display          = true;   ///< 是否显示窗口
};

// ============================================================================
// 渲染统计
// ============================================================================

/// @brief 渲染线程运行时指标
struct RenderMetrics {
    int64_t frames_rendered     = 0;     ///< 渲染帧数
    int64_t frames_displayed    = 0;     ///< 实际显示帧数
    int64_t frames_dropped      = 0;     ///< 丢弃帧数
    int64_t frames_duplicated   = 0;     ///< 重复帧数
    double  actual_fps          = 0.0;   ///< 实际帧率
    double  avg_render_ms       = 0.0;   ///< 平均渲染耗时（毫秒）
    double  avg_sync_wait_ms    = 0.0;   ///< 平均同步等待耗时
    int     output_queue_depth  = 0;     ///< 渲染队列深度
    bool    frame_pacing_active = false; ///< 帧间隔调节是否生效

    std::string ToString() const;
};

// ============================================================================
// 帧输出回调
// ============================================================================

/// @brief 帧输出回调，用户自定义帧处理（显示/编码/网络发送等）
using FrameCallback = std::function<void(const cv::Mat& frame,
                                         int64_t pts_ms,
                                         int64_t frame_id)>;

// ============================================================================
// 渲染线程
// ============================================================================

/**
 * @brief 渲染线程
 *
 * 位于流水线末端，负责：
 * - 从渲染队列获取 RenderPacket
 * - 通过 OutputProcessor 执行图像融合（逆变换→融合→锐化→色彩混合）
 * - 通过 AudioPlayer 获取音频时钟进行音视频同步
 * - 通过 FrameScheduler 决策 DROP/DUPLICATE/DISPLAY
 * - 帧间隔调节稳定输出帧率
 * - 帧输出回调（可接入显示窗口或编码器）
 */
class RenderThread : public ThreadBase {
public:
    explicit RenderThread(const std::string& name = "RenderThread");
    ~RenderThread() override;

    RenderThread(const RenderThread&) = delete;
    RenderThread& operator=(const RenderThread&) = delete;
    RenderThread(RenderThread&&) = delete;
    RenderThread& operator=(RenderThread&&) = delete;

    // ========================================================================
    // 配置
    // ========================================================================

    void SetConfig(const RenderConfig& config);
    const RenderConfig& GetConfig() const;

    // ========================================================================
    // 模块注入
    // ========================================================================

    /// @brief 设置 OutputProcessor（外部管理生命周期）
    void SetOutputProcessor(model::OutputProcessor* processor);

    /// @brief 设置 AudioPlayer（用于音频同步时钟，可选）
    void SetAudioPlayer(audio::AudioPlayer* player);

    // ========================================================================
    // 队列
    // ========================================================================

    /// @brief 设置渲染输入队列
    void SetInputQueue(ThreadSafeQueue<RenderPacket>* queue);

    /// @brief 设置输出帧队列（可选，用于下游消费）
    void SetOutputQueue(ThreadSafeQueue<OutputFramePacket>* queue);

    // ========================================================================
    // 回调
    // ========================================================================

    /// @brief 设置帧输出回调
    void SetFrameCallback(FrameCallback callback);

    // ========================================================================
    // 线程主循环
    // ========================================================================

    void Run() override;

    // ========================================================================
    // 控制
    // ========================================================================

    /// @brief 标记输入结束
    void MarkInputEOS();

    /// @brief 重置所有状态
    void Reset();

    // ========================================================================
    // 查询
    // ========================================================================

    RenderMetrics GetMetrics() const;
    FrameStats    GetFrameStats() const;
    int64_t       GetRenderedCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
