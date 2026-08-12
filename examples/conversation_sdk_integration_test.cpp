#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include "dialog/conversation_session.h"
#include "digital_human_sdk.h"

using namespace digital_human;
namespace fs = std::filesystem;

namespace {

class FixedTextClient final : public dialog::ITextGenerationClient {
public:
    bool Generate(const dialog::GenerateRequest&,
                  const dialog::TextDeltaCallback& on_delta,
                  const dialog::CancelCheck& cancelled,
                  std::string&) override {
        if (cancelled && cancelled()) return false;
        on_delta("端到端闭环测试完成。");
        return true;
    }
};

class FixedTTSClient final : public tts::ITTSClient {
public:
    bool Synthesize(const std::string&,
                    const tts::PCMCallback& on_audio,
                    const tts::CancelCheck& cancelled,
                    std::string&) override {
        constexpr int sample_rate = 16000;
        constexpr int chunk_samples = 1600;
        for (int chunk_index = 0; chunk_index < 5; ++chunk_index) {
            if (cancelled && cancelled()) return false;
            tts::PCMChunk chunk;
            chunk.sample_rate = sample_rate;
            chunk.channels = 1;
            chunk.samples.resize(chunk_samples);
            for (int i = 0; i < chunk_samples; ++i) {
                const int sample_index = chunk_index * chunk_samples + i;
                chunk.samples[static_cast<size_t>(i)] = static_cast<float>(
                    0.08 * std::sin(2.0 * 3.141592653589793 * 220.0
                                    * sample_index / sample_rate));
            }
            if (!on_audio(std::move(chunk))) return false;
        }
        return true;
    }
};

}  // namespace

int main() {
    const fs::path project(PROJECT_SOURCE_DIR);
    const fs::path face_dir = project / "models" / "face";
    const fs::path avatar_path = project / "assets" / "face.jpg";
    if (!fs::exists(project / "models" / "Wav2Lip-SD-GAN-opt.param")
        || !fs::exists(project / "models" / "Wav2Lip-SD-GAN-opt.bin")
        || !fs::exists(face_dir) || !fs::exists(avatar_path)) {
        std::cout << "[SKIP] integration models/avatar are missing\n";
        return 0;
    }

    SDKConfig sdk_config;
    sdk_config.lipsync_model_dir = (project / "models").string();
    sdk_config.face_model_dir = face_dir.string();
    sdk_config.enable_frame_pacing = false;
    sdk_config.output_queue_size = 30;
    DigitalHumanSDK sdk;
    if (sdk.Init(sdk_config) != SDKError::OK
        || sdk.Start() != SDKError::OK) {
        std::cerr << "[FAIL] SDK start: " << sdk.GetLastError() << '\n';
        return 1;
    }

    std::atomic<int64_t> output_frames{0};
    std::atomic<bool> pts_ok{true};
    std::thread consumer([&]() {
        int64_t previous_pts = -1;
        while (true) {
            cv::Mat frame;
            int64_t pts_ms = 0;
            const auto result = sdk.GetOutputFrame(frame, pts_ms, 500);
            if (result == SDKError::OK) {
                if (frame.empty() || pts_ms < previous_pts) pts_ok.store(false);
                previous_pts = pts_ms;
                output_frames.fetch_add(1);
            } else if (result == SDKError::TIMEOUT) {
                continue;
            } else {
                break;
            }
        }
    });

    FixedTextClient text_client;
    FixedTTSClient tts_client;
    dialog::SDKDigitalHumanSink sink(sdk);
    dialog::ConversationSession session(text_client, tts_client, sink);
    dialog::ConversationConfig config;
    config.session_id = "sdk-integration";
    cv::Mat avatar = cv::imread(avatar_path.string());
    if (!session.Start(config, avatar)
        || session.SubmitUserText("开始测试") == 0
        || !session.WaitUntilIdle(std::chrono::seconds(15))) {
        std::cerr << "[FAIL] conversation did not reach idle\n";
        session.Stop(false);
        if (consumer.joinable()) consumer.join();
        sdk.Stop();
        return 1;
    }
    session.Stop(true);
    if (consumer.joinable()) consumer.join();
    sdk.Stop();

    if (output_frames.load() <= 0 || !pts_ok.load()) {
        std::cerr << "[FAIL] invalid SDK output: frames="
                  << output_frames.load() << ", pts_ok=" << pts_ok.load()
                  << '\n';
        return 1;
    }
    std::cout << "[PASS] service text -> TTS PCM -> SDK -> rendered frames: "
              << output_frames.load() << '\n';
    return 0;
}
