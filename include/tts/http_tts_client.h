#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "network/http_client.h"
#include "tts/tts_client.h"

namespace digital_human {
namespace tts {

/// Generic HTTP contract:
/// request JSON: {text, sample_rate, channels, format}
/// response body: raw little-endian PCM in the configured format.
struct HttpTTSConfig {
    std::string endpoint;
    std::string api_key;
    std::vector<std::string> headers;
    TTSAudioFormat response_format = TTSAudioFormat::PCM_S16LE;
    int sample_rate = 16000;
    int channels = 1;
    int chunk_samples = 1600;
    int connect_timeout_ms = 2000;
    int request_timeout_ms = 30000;
    /// 资源限制（P0 流式 TTS）：在 libcurl write callback 中增量校验，
    /// 超过任一上限立即中止传输，避免长文本回复造成无界内存增长。
    int max_response_bytes = 52'428'800;   // 50 MB
    int max_audio_duration_ms = 300'000;   // 5 分钟
};

/// HTTP 适配器：将通用 HTTP TTS 端点适配为 ITTSClient。
/// 此类从 tts_client.h 拆分出来，使抽象接口不再传递 libcurl 依赖。
class DH_API HttpTTSClient final : public ITTSClient {
public:
    explicit HttpTTSClient(HttpTTSConfig config);

    bool Synthesize(const std::string& text,
                    const PCMCallback& on_audio,
                    const CancelCheck& cancelled,
                    std::string& error) override;

private:
    HttpTTSConfig config_;
    network::HttpClient http_;
};

}  // namespace tts
}  // namespace digital_human
