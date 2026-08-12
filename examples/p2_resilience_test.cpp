#include <cmath>
#include <iostream>
#include <memory>

#include "audio/audio_player.h"
#include "core/audio_sync_scheduler.h"
#include "core/media_clock.h"
#include "digital_human/quality_gate.h"

namespace {

class FakeAudioPlayer final : public digital_human::audio::IAudioPlayer {
public:
    bool Init(int sample_rate, int, int) override {
        sample_rate_ = sample_rate;
        initialized_ = true;
        return true;
    }
    void Destroy() override { initialized_ = false; }
    bool IsInitialized() const override { return initialized_; }
    bool LoadAudio(const float*, int num_samples, int channels) override {
        duration_ms_ = channels > 0
            ? static_cast<double>(num_samples / channels) * 1000.0
                / sample_rate_
            : 0.0;
        return true;
    }
    bool LoadAudio(const std::vector<float>& samples, int channels) override {
        return LoadAudio(samples.data(), static_cast<int>(samples.size()),
                         channels);
    }
    bool Play() override { state_ = digital_human::audio::AudioPlayerState::PLAYING; return true; }
    bool Pause() override { state_ = digital_human::audio::AudioPlayerState::PAUSED; return true; }
    bool Resume() override { return Play(); }
    bool Stop() override { state_ = digital_human::audio::AudioPlayerState::STOPPED; consumed_ = 0; return true; }
    digital_human::audio::AudioPlayerState GetState() const override { return state_; }
    bool IsPlaying() const override { return state_ == digital_human::audio::AudioPlayerState::PLAYING; }
    bool IsPaused() const override { return state_ == digital_human::audio::AudioPlayerState::PAUSED; }
    bool IsStopped() const override { return state_ == digital_human::audio::AudioPlayerState::STOPPED; }
    bool IsFinished() const override { return false; }
    int64_t GetConsumedFrames() const override { return consumed_; }
    double GetPlaybackPositionMs() const override { return static_cast<double>(consumed_) * 1000.0 / sample_rate_; }
    double GetDacTimeMs() const override { return GetPlaybackPositionMs(); }
    double GetTotalDurationMs() const override { return duration_ms_; }
    std::string GetLastErrorMsg() const override { return {}; }

private:
    int sample_rate_ = 48000;
    int64_t consumed_ = 0;
    double duration_ms_ = 0.0;
    bool initialized_ = false;
    digital_human::audio::AudioPlayerState state_ =
        digital_human::audio::AudioPlayerState::IDLE;
};

bool Check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAILED: " << message << '\n';
    return condition;
}

}  // namespace

int main() {
    using digital_human::QualityGate;
    using digital_human::QualityGateSample;
    using digital_human::core::MediaClock;
    using digital_human::core::MediaClockConfig;

    bool ok = true;

    digital_human::core::AudioSyncScheduler scheduler(
        std::make_unique<FakeAudioPlayer>());
    digital_human::core::AudioSyncConfig sync_config;
    ok &= Check(scheduler.Init(sync_config),
                "injected audio backend was not accepted");
    ok &= Check(scheduler.LoadAudio(std::vector<float>(4800), 1),
                "injected audio backend could not load PCM");
    scheduler.Destroy();

    MediaClockConfig config;
    config.audio_sample_rate = 1000;
    config.expected_video_interval_ms = 40.0;
    config.drift_correction_threshold_ms = 50.0;
    config.max_correction_step_ms = 5.0;
    MediaClock clock(config);

    for (int index = 0; index < 100; ++index) {
        clock.AdvanceAudioFrames(40);
        // Simulate a long-session clock that accumulates 1 ms per frame.
        clock.ObserveVideoPts((index + 1) * 41.0);
    }
    const auto snapshot = clock.Snapshot();
    ok &= Check(snapshot.pts_monotonic, "monotonic PTS rejected");
    ok &= Check(snapshot.corrections > 0, "drift correction did not engage");
    ok &= Check(snapshot.jitter_p95_ms <= 1.01, "jitter percentile is wrong");
    ok &= Check(clock.CorrectVideoPts(4100.0) < 4100.0,
                "video correction was not applied");

    clock.ObserveVideoPts(100.0);
    ok &= Check(!clock.Snapshot().pts_monotonic,
                "non-monotonic PTS was not detected");

    QualityGateSample passing;
    passing.fps_p95 = 25.0;
    passing.frame_latency_p95_ms = 60.0;
    passing.drop_rate = 0.01;
    passing.av_sync_p95_ms = 50.0;
    passing.av_duration_delta_ms = 40.0;
    passing.mouth_psnr_db = 32.0;
    passing.mouth_ssim = 0.95;
    ok &= Check(QualityGate::Evaluate(passing).passed,
                "valid quality sample failed");

    passing.drop_rate = 0.20;
    passing.pts_monotonic = false;
    const auto failing = QualityGate::Evaluate(passing);
    ok &= Check(!failing.passed && failing.violations.size() == 2,
                "quality violations were not reported independently");

    if (ok) std::cout << "P2 resilience tests passed\n";
    return ok ? 0 : 1;
}
