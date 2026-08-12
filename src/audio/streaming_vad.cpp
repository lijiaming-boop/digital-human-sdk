#include "audio/streaming_vad.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <vector>

namespace digital_human {
namespace audio {

namespace {

class EnergyStreamingVAD : public IStreamingVAD {
public:
    explicit EnergyStreamingVAD(const StreamingVADConfig& config)
        : config_(config)
        , frame_samples_(static_cast<size_t>(
              config.sample_rate * config.frame_ms / 1000))
        , residual_(frame_samples_, 0.0F)
        , residual_count_(0) {
    }

    bool Start(const VADCallback& on_event) override {
        if (!on_event) return false;
        on_event_ = on_event;
        running_ = true;
        in_voice_ = false;
        voice_frame_count_ = 0;
        silence_frame_count_ = 0;
        residual_count_ = 0;
        return true;
    }

    bool PushAudio(const float* samples, size_t sample_count) override {
        if (!running_ || !on_event_) return false;
        if (sample_count > 0 && samples == nullptr) return false;

        size_t consumed = 0;
        while (consumed < sample_count) {
            const size_t needed = frame_samples_ - residual_count_;
            const size_t available = sample_count - consumed;
            const size_t to_copy = std::min(needed, available);

            std::memcpy(residual_.data() + residual_count_,
                        samples + consumed,
                        to_copy * sizeof(float));
            residual_count_ += to_copy;
            consumed += to_copy;

            if (residual_count_ >= frame_samples_) {
                ProcessFrame(residual_.data(), frame_samples_);
                residual_count_ = 0;
            }
        }
        return true;
    }

    void Stop() override {
        running_ = false;
        on_event_ = {};
    }

private:
    void ProcessFrame(const float* frame, size_t count) {
        // 计算能量
        double energy = 0.0;
        for (size_t i = 0; i < count; ++i) {
            energy += static_cast<double>(frame[i]) * frame[i];
        }
        energy /= static_cast<double>(count);
        const double rms = std::sqrt(energy);

        const bool frame_has_voice = rms > config_.energy_threshold;

        if (frame_has_voice) {
            ++voice_frame_count_;
            silence_frame_count_ = 0;

            if (!in_voice_ && voice_frame_count_ >= config_.min_voice_frames) {
                in_voice_ = true;
                on_event_(VADEvent::VoiceStart);
            }
        } else {
            ++silence_frame_count_;
            voice_frame_count_ = 0;

            if (in_voice_ && silence_frame_count_ >= config_.hangover_frames) {
                in_voice_ = false;
                on_event_(VADEvent::VoiceEnd);
            }

            // 静音超时检查
            if (config_.silence_timeout_ms > 0) {
                const int silence_ms = silence_frame_count_ * config_.frame_ms;
                if (silence_ms >= config_.silence_timeout_ms) {
                    on_event_(VADEvent::Silence);
                }
            }
        }
    }

    StreamingVADConfig config_;
    size_t             frame_samples_;
    std::vector<float> residual_;
    size_t             residual_count_;
    VADCallback        on_event_;
    bool               running_ = false;
    bool               in_voice_ = false;
    int                voice_frame_count_ = 0;
    int                silence_frame_count_ = 0;
};

}  // namespace

DH_API std::unique_ptr<IStreamingVAD> create_streaming_vad(
    const StreamingVADConfig& config) {
    if (config.sample_rate <= 0 || config.frame_ms <= 0
        || config.hangover_frames < 0 || config.min_voice_frames <= 0
        || config.silence_timeout_ms < 0
        || !std::isfinite(config.energy_threshold)
        || config.energy_threshold < 0.0F) {
        return nullptr;
    }
    const int64_t frame_samples =
        static_cast<int64_t>(config.sample_rate) * config.frame_ms / 1000;
    if (frame_samples <= 0
        || static_cast<uint64_t>(frame_samples)
            > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return nullptr;
    }
    return std::make_unique<EnergyStreamingVAD>(config);
}

}  // namespace audio
}  // namespace digital_human
