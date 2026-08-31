#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "digital_human/export.h"

namespace digital_human {
namespace tts {

struct PCMChunk {
    std::vector<float> samples;
    int sample_rate = 16000;
    int channels = 1;
};

using PCMCallback = std::function<bool(PCMChunk)>;
using CancelCheck = std::function<bool()>;

/// 抽象 TTS 接口：会话层只依赖此接口，不传递任何 HTTP/libcurl 依赖。
/// 具体适配器（HttpTTSClient）声明在独立头文件中，由需要网络能力的
/// 调用方按需引入。
class DH_API ITTSClient {
public:
    virtual ~ITTSClient() = default;

    virtual bool Synthesize(const std::string& text,
                            const PCMCallback& on_audio,
                            const CancelCheck& cancelled,
                            std::string& error) = 0;
};

enum class TTSAudioFormat {
    PCM_S16LE,
    PCM_F32LE,
};

}  // namespace tts
}  // namespace digital_human
