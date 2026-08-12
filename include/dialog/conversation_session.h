#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "dialog/text_generation_client.h"
#include "tts/tts_client.h"

namespace digital_human {

class DigitalHumanSDK;

namespace dialog {

class IDigitalHumanSink {
public:
    virtual ~IDigitalHumanSink() = default;
    virtual bool PushAudio(const std::vector<float>& samples,
                           int64_t pts_ms,
                           std::string& error) = 0;
    virtual bool PushVideo(const cv::Mat& frame,
                           int64_t pts_ms,
                           std::string& error) = 0;
    virtual void Finish() = 0;
};

class SDKDigitalHumanSink final : public IDigitalHumanSink {
public:
    explicit SDKDigitalHumanSink(DigitalHumanSDK& sdk);

    bool PushAudio(const std::vector<float>& samples,
                   int64_t pts_ms,
                   std::string& error) override;
    bool PushVideo(const cv::Mat& frame,
                   int64_t pts_ms,
                   std::string& error) override;
    void Finish() override;

private:
    DigitalHumanSDK& sdk_;
};

/// 头像热更新画布策略（P0）：UpdateAvatar 收到与初始画布尺寸不同的头像时，
/// 按 policy 决定如何适配到固定画布，保证编码流分辨率不中途变更。
enum class AvatarUpdatePolicy {
    Reject,  // 尺寸与画布不一致时直接拒绝
    Fit,     // 等比缩放并居中，多余区域用黑色填充（保持画布尺寸）
    Cover,   // 等比缩放填满画布并居中裁剪（保持画布尺寸）
};

struct ConversationConfig {
    std::string session_id;
    std::string system_prompt;
    int audio_sample_rate = 16000;
    int audio_channels = 1;
    double target_fps = 25.0;
    int mel_lookahead_ms = 160;
    int reply_tail_silence_ms = 200;
    size_t min_tts_clause_chars = 8;
    size_t max_pending_audio_chunks = 64;
    /// 头像画布契约（P0 热更新）：Session 启动时固定画布尺寸，后续 UpdateAvatar
    /// 必须服从该画布，避免改变已打开的 H.264 编码流分辨率。
    AvatarUpdatePolicy avatar_update_policy = AvatarUpdatePolicy::Fit;
    int avatar_canvas_width = 0;   // 0: 以初始头像尺寸为画布
    int avatar_canvas_height = 0;
};

struct ConversationCallbacks {
    // Callbacks run on ConversationSession worker threads. They must return
    // quickly and must not call Stop() synchronously from inside the callback.
    std::function<void(uint64_t, const std::string&)> on_text_delta;
    std::function<void(uint64_t, const std::string&)> on_reply_ready;
    std::function<void(uint64_t)> on_turn_complete;
    std::function<void(uint64_t, const std::string&)> on_error;
};

/// 显式会话状态机（P0 错误/取消/停止语义）：
///   IDLE → GENERATING → SYNTHESIZING → PLAYING → IDLE
///              │              │             │
///              └──────────────┴─────────────┤
///                                        INTERRUPTING
///   任意活动状态 → FAILED / STOPPING → STOPPED
/// 任一不可恢复媒体错误只产生一次终态事件（turn_failed 去重）。
enum class SessionState {
    IDLE,
    GENERATING,
    SYNTHESIZING,
    PLAYING,
    INTERRUPTING,
    FAILED,
    STOPPING,
    STOPPED,
};

/// Stop() 的返回结果，使调用者能区分正常退出、超时和失败。
enum class StopResult {
    Stopped,  // 所有工作线程在 deadline 内正常退出
    Timeout,  // drain 阶段超过 deadline，已强制 Interrupt
    Failed,   // 停止过程中出现错误（见 GetLastError）
};

/// First-stage conversation orchestrator:
/// user text -> text generation service -> sentence segmentation -> TTS -> SDK input.
class ConversationSession {
public:
    ConversationSession(ITextGenerationClient& text_client,
                        tts::ITTSClient& tts_client,
                        IDigitalHumanSink& media_sink);
    ~ConversationSession();

    ConversationSession(const ConversationSession&) = delete;
    ConversationSession& operator=(const ConversationSession&) = delete;

    bool Start(const ConversationConfig& config,
               const cv::Mat& avatar_frame,
               ConversationCallbacks callbacks = {});

    /// Replaces the avatar used by subsequently submitted video frames.
    /// The image is normalized to BGR and cloned before this method returns.
    bool UpdateAvatar(const cv::Mat& avatar_frame);

    /// Returns 0 if the session is stopped, busy, or the text is empty.
    uint64_t SubmitUserText(const std::string& text);
    void Interrupt();
    bool WaitUntilIdle(std::chrono::milliseconds timeout);

    /// 停止会话并回收所有工作线程。
    /// - drain=true 时先等待当前 turn 在 timeout 内完成，超时则强制 Interrupt。
    /// - drain=false 时立即丢弃所有待处理任务。
    /// 返回明确的成功/超时/失败结果，调用者可据此决定后续动作。
    StopResult Stop(bool drain = true,
                    std::chrono::milliseconds timeout = std::chrono::seconds(30));
    bool IsBusy() const;
    SessionState State() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dialog
}  // namespace digital_human
