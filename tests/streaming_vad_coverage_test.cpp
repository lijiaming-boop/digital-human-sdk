#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "audio/streaming_vad.h"
#include "test_support.h"

namespace {

using digital_human::audio::StreamingVADConfig;
using digital_human::audio::VADEvent;
using digital_human::audio::create_streaming_vad;

size_t Count(const std::vector<VADEvent>& events, VADEvent wanted) {
    return static_cast<size_t>(std::count(events.begin(), events.end(), wanted));
}

}  // namespace

int main() {
    TestSuite suite;

    StreamingVADConfig invalid;
    invalid.sample_rate = 0;
    suite.Check(!create_streaming_vad(invalid), "reject zero sample rate");
    invalid = {};
    invalid.frame_ms = 0;
    suite.Check(!create_streaming_vad(invalid), "reject zero frame length");
    invalid = {};
    invalid.hangover_frames = -1;
    suite.Check(!create_streaming_vad(invalid), "reject negative hangover");
    invalid = {};
    invalid.min_voice_frames = 0;
    suite.Check(!create_streaming_vad(invalid), "reject zero voice threshold");
    invalid = {};
    invalid.silence_timeout_ms = -1;
    suite.Check(!create_streaming_vad(invalid), "reject negative silence timeout");
    invalid = {};
    invalid.energy_threshold = std::numeric_limits<float>::quiet_NaN();
    suite.Check(!create_streaming_vad(invalid), "reject NaN energy threshold");
    invalid = {};
    invalid.sample_rate = 1;
    invalid.frame_ms = 1;
    suite.Check(!create_streaming_vad(invalid), "reject sub-sample frame");

    StreamingVADConfig config;
    config.sample_rate = 8000;
    config.frame_ms = 20;
    config.energy_threshold = 0.1F;
    config.min_voice_frames = 2;
    config.hangover_frames = 2;
    config.silence_timeout_ms = 60;
    const size_t frame_samples = 160;
    std::vector<float> voice(frame_samples, 0.5F);
    std::vector<float> silence(frame_samples, 0.0F);

    auto vad = create_streaming_vad(config);
    suite.Check(static_cast<bool>(vad), "create valid streaming VAD");
    suite.Check(!vad->Start({}), "reject empty event callback");
    suite.Check(!vad->PushAudio(voice.data(), voice.size()),
                "reject audio before Start");

    std::vector<VADEvent> events;
    suite.Check(vad->Start([&](VADEvent event) { events.push_back(event); }),
                "start streaming VAD");
    suite.Check(vad->PushAudio(nullptr, 0), "accept empty audio chunk");
    suite.Check(!vad->PushAudio(nullptr, 1), "reject null non-empty chunk");
    suite.Check(vad->PushAudio(voice.data(), frame_samples / 2),
                "accept partial frame");
    suite.Check(events.empty(), "partial frame produces no event");
    suite.Check(vad->PushAudio(voice.data() + frame_samples / 2,
                               frame_samples / 2),
                "complete partial frame");
    suite.Check(events.empty(), "one voice frame is below debounce");
    suite.Check(vad->PushAudio(voice.data(), voice.size()),
                "push second voice frame");
    suite.Equal(Count(events, VADEvent::VoiceStart), size_t{1},
                "VoiceStart fires after configured debounce");

    suite.Check(vad->PushAudio(silence.data(), silence.size()),
                "push first hangover frame");
    suite.Equal(Count(events, VADEvent::VoiceEnd), size_t{0},
                "VoiceEnd waits for hangover");
    suite.Check(vad->PushAudio(silence.data(), silence.size()),
                "push second hangover frame");
    suite.Equal(Count(events, VADEvent::VoiceEnd), size_t{1},
                "VoiceEnd fires at hangover boundary");
    suite.Check(vad->PushAudio(silence.data(), silence.size()),
                "push silence timeout frame");
    suite.Check(Count(events, VADEvent::Silence) >= 1,
                "Silence event fires at timeout");

    vad->Stop();
    suite.Check(!vad->PushAudio(voice.data(), voice.size()),
                "Stop rejects further audio");

    // A new Start must not combine residual samples from the previous run.
    events.clear();
    suite.Check(vad->Start([&](VADEvent event) { events.push_back(event); }),
                "restart VAD for residual test");
    suite.Check(vad->PushAudio(voice.data(), frame_samples / 2),
                "seed residual before stop");
    vad->Stop();
    events.clear();
    suite.Check(vad->Start([&](VADEvent event) { events.push_back(event); }),
                "restart clears previous residual");
    std::vector<float> mixed;
    mixed.insert(mixed.end(), frame_samples / 2, 0.0F);
    mixed.insert(mixed.end(), frame_samples, 0.5F);
    suite.Check(vad->PushAudio(mixed.data(), mixed.size()),
                "push mixed chunk after restart");
    suite.Equal(Count(events, VADEvent::VoiceStart), size_t{0},
                "old residual does not affect restarted detector");
    vad->Stop();

    return suite.Finish("streaming_vad_coverage_test");
}
