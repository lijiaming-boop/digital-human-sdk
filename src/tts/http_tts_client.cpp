#include "tts/tts_client.h"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include "network/json_utils.h"

namespace digital_human {
namespace tts {
namespace {

std::string FormatName(TTSAudioFormat format) {
    return format == TTSAudioFormat::PCM_F32LE ? "pcm_f32le" : "pcm_s16le";
}

std::string BuildBody(const std::string& text, const HttpTTSConfig& config) {
    return "{\"text\":\"" + network::json::Escape(text)
        + "\",\"sample_rate\":" + std::to_string(config.sample_rate)
        + ",\"channels\":" + std::to_string(config.channels)
        + ",\"format\":\"" + FormatName(config.response_format) + "\"}";
}

/// 将单个 PCM 样本的原始字节解码为 float。
inline float DecodeSample(const uint8_t* p, TTSAudioFormat format) {
    if (format == TTSAudioFormat::PCM_F32LE) {
        uint32_t bits = static_cast<uint32_t>(p[0])
            | (static_cast<uint32_t>(p[1]) << 8)
            | (static_cast<uint32_t>(p[2]) << 16)
            | (static_cast<uint32_t>(p[3]) << 24);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(float));
        return value;
    }
    const uint16_t raw = static_cast<uint16_t>(p[0])
        | (static_cast<uint16_t>(p[1]) << 8);
    return static_cast<float>(static_cast<int16_t>(raw)) / 32768.0f;
}

}  // namespace

HttpTTSClient::HttpTTSClient(HttpTTSConfig config)
    : config_(std::move(config)) {}

bool HttpTTSClient::Synthesize(const std::string& text,
                               const PCMCallback& on_audio,
                               const CancelCheck& cancelled,
                               std::string& error) {
    if (text.empty() || !on_audio) {
        error = "TTS text or audio callback is empty";
        return false;
    }
    if (config_.sample_rate <= 0 || config_.channels <= 0
        || config_.chunk_samples <= 0) {
        error = "invalid TTS audio configuration";
        return false;
    }

    network::HttpRequest request;
    request.url = config_.endpoint;
    request.body = BuildBody(text, config_);
    request.connect_timeout_ms = config_.connect_timeout_ms;
    request.request_timeout_ms = config_.request_timeout_ms;
    // 将 TTS 资源上限下推到传输层，由 libcurl 在 socket 层强制（P0 HTTP 边界）。
    request.max_response_bytes = config_.max_response_bytes > 0
        ? static_cast<int64_t>(config_.max_response_bytes) : 0;
    request.low_speed_limit_bytes_per_s = 1024;   // 1 KB/s 下限
    request.low_speed_time_ms = config_.request_timeout_ms > 0
        ? config_.request_timeout_ms : 30000;
    request.follow_redirects = false;  // TTS 端点不应重定向，收紧 SSRF 边界
    request.headers = config_.headers;
    request.headers.emplace_back("Content-Type: application/json");
    request.headers.emplace_back("Accept: application/octet-stream");
    if (!config_.api_key.empty()) {
        request.headers.emplace_back("Authorization: Bearer " + config_.api_key);
    }

    // 真正流式 PCM（P0）：在 libcurl write callback 中增量解析字节为 float 样本，
    // 凑够一个 chunk 立即回调 on_audio，不再缓存完整响应。首 PCM 延迟仅取决于
    // 服务端返回首个完整 chunk 的时机，而非整段合成完成。
    const size_t bytes_per_sample =
        config_.response_format == TTSAudioFormat::PCM_F32LE ? 4 : 2;
    const size_t chunk_values = static_cast<size_t>(config_.chunk_samples)
                              * static_cast<size_t>(config_.channels);
    const size_t max_samples = config_.max_audio_duration_ms > 0
        ? static_cast<size_t>(config_.sample_rate)
            * static_cast<size_t>(config_.channels)
            * static_cast<size_t>(config_.max_audio_duration_ms) / 1000
        : 0;
    const size_t max_bytes = config_.max_response_bytes > 0
        ? static_cast<size_t>(config_.max_response_bytes) : 0;

    struct StreamState {
        std::vector<uint8_t> residual;   // 不足一个 sample 的尾部字节
        std::vector<float> pending;      // 已解码、待凑齐 chunk 的样本
        size_t total_bytes = 0;
        size_t total_samples = 0;
        bool rejected = false;           // on_audio 返回 false
        bool limit_exceeded = false;
        std::string error;
    } state;

    auto on_data = [&](const uint8_t* data, size_t size) -> bool {
        if (cancelled && cancelled()) return false;
        state.total_bytes += size;
        if (max_bytes > 0 && state.total_bytes > max_bytes) {
            state.limit_exceeded = true;
            state.error = "TTS response exceeded max_response_bytes ("
                        + std::to_string(config_.max_response_bytes) + ")";
            return false;
        }
        // 追加到尾部缓冲，尝试解码完整 sample。
        state.residual.insert(state.residual.end(), data, data + size);
        const size_t available = state.residual.size() / bytes_per_sample;
        if (available == 0) return true;
        for (size_t i = 0; i < available; ++i) {
            state.pending.push_back(DecodeSample(
                state.residual.data() + i * bytes_per_sample,
                config_.response_format));
        }
        const size_t consumed = available * bytes_per_sample;
        state.residual.erase(state.residual.begin(),
                             state.residual.begin() + consumed);
        state.total_samples += available;
        if (max_samples > 0 && state.total_samples > max_samples) {
            state.limit_exceeded = true;
            state.error = "TTS response exceeded max_audio_duration_ms ("
                        + std::to_string(config_.max_audio_duration_ms) + ")";
            return false;
        }
        // 凑够一个 chunk 立即回调，缩短首 PCM 延迟。
        while (state.pending.size() >= chunk_values) {
            PCMChunk chunk;
            chunk.sample_rate = config_.sample_rate;
            chunk.channels = config_.channels;
            chunk.samples.assign(state.pending.begin(),
                                 state.pending.begin() + chunk_values);
            state.pending.erase(state.pending.begin(),
                                state.pending.begin() + chunk_values);
            if (!on_audio(std::move(chunk))) {
                state.rejected = true;
                return false;
            }
        }
        return true;
    };

    network::HttpResponseInfo response;
    const bool posted = http_.Post(
        request, on_data, cancelled, response, error);
    if (!posted) {
        // 优先用流式解析中记录的具体原因覆盖传输层通用错误。
        if (state.limit_exceeded && !state.error.empty()) {
            error = state.error;
        } else if (state.rejected) {
            error = "TTS audio callback rejected data";
        }
        return false;
    }

    // 校验响应 Content-Type（传输完成后才可读取，仅作后置校验）。
    if (!response.content_type.empty()
        && response.content_type.find("audio/") == std::string::npos
        && response.content_type.find("octet-stream") == std::string::npos
        && response.content_type.find("text/plain") == std::string::npos) {
        error = "TTS response has unexpected Content-Type: "
              + response.content_type;
        return false;
    }

    // 残留字节必须是完整 sample，否则视为截断。
    if (!state.residual.empty()) {
        error = "TTS response ended with an incomplete PCM sample";
        return false;
    }

    // flush 尾部不足一个 chunk 的剩余样本。
    if (!state.pending.empty()) {
        if (cancelled && cancelled()) {
            error = "TTS request cancelled";
            return false;
        }
        PCMChunk chunk;
        chunk.sample_rate = config_.sample_rate;
        chunk.channels = config_.channels;
        chunk.samples = std::move(state.pending);
        if (!on_audio(std::move(chunk))) {
            error = "TTS audio callback rejected data";
            return false;
        }
    }
    return true;
}

}  // namespace tts
}  // namespace digital_human
