#pragma once

/// @file asr_client.h
/// @brief ASR 抽象接口（P1-3.10b）
///
/// 支持流式语音识别，提供 partial/final transcript 事件。

#include <functional>
#include <string>
#include "digital_human/export.h"

namespace digital_human {
namespace dialog {

/// ASR 转录结果
struct Transcript {
    std::string text;
    bool is_final = false;       ///< true=最终结果, false=partial
    double confidence = 1.0;
    int64_t start_ms = 0;        ///< 语音起始时间
    int64_t end_ms = 0;          ///< 语音结束时间
};

using TranscriptCallback = std::function<void(const Transcript&)>;
using ASRCancelCheck = std::function<bool()>;

/// 抽象 ASR 接口
/// 实现 1：本地流式 ASR（如 sherpa-onnx、vosk）
/// 实现 2：HTTP 流式 ASR（如 Whisper API streaming）
class DH_API IASRClient {
public:
    virtual ~IASRClient() = default;

    /// 开始识别会话
    virtual bool Start(const TranscriptCallback& on_transcript,
                       const ASRCancelCheck& cancelled,
                       std::string& error) = 0;

    /// 推送音频帧（16kHz mono float32）
    virtual bool PushAudio(const float* samples,
                           size_t sample_count,
                           std::string& error) = 0;

    /// 停止识别，等待最终结果
    virtual bool Stop(std::string& error) = 0;
};

}  // namespace dialog
}  // namespace digital_human
