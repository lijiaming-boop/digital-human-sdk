#pragma once

/// @file streaming_vad.h
/// @brief 流式 VAD 接口（P1-3.10b）
///
/// 提供实时语音活动检测，触发 VoiceStart/VoiceEnd 事件。
/// 与离线批处理 VAD (audio_vad.h) 不同，此接口支持实时帧级推送。

#include <functional>
#include <memory>
#include "digital_human/export.h"

namespace digital_human {
namespace audio {

/// 流式 VAD 事件
enum class VADEvent {
    VoiceStart,    ///< 检测到语音开始
    VoiceEnd,      ///< 检测到语音结束
    Silence,       ///< 静音持续（用于超时判断）
};

using VADCallback = std::function<void(VADEvent)>;

/// 抽象流式 VAD 接口
class DH_API IStreamingVAD {
public:
    virtual ~IStreamingVAD() = default;

    /// 开始检测
    virtual bool Start(const VADCallback& on_event) = 0;

    /// 推送音频帧（16kHz mono float32）
    virtual bool PushAudio(const float* samples, size_t sample_count) = 0;

    /// 停止检测
    virtual void Stop() = 0;
};

/// VAD 配置
struct DH_API StreamingVADConfig {
    int   sample_rate = 16000;
    int   frame_ms = 30;               ///< 帧长（ms）
    float energy_threshold = 0.02F;    ///< 能量阈值
    int   hangover_frames = 8;         ///< 语音结束后的 hangover 帧数
    int   min_voice_frames = 3;        ///< 判定 VoiceStart 的最小连续语音帧
    int   silence_timeout_ms = 0;      ///< 静音超时（0=不触发）
};

/// 创建默认流式 VAD（能量+过零率）
DH_API std::unique_ptr<IStreamingVAD> create_streaming_vad(
    const StreamingVADConfig& config);

}  // namespace audio
}  // namespace digital_human
