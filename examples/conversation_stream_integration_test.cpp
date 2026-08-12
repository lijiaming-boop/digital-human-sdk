#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
}

#include <opencv2/imgcodecs.hpp>

#include "dialog/conversation_session.h"
#include "digital_human_sdk.h"
#include "media/conversation_stream_bridge.h"
#include "media/stream_publisher.h"

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
        on_delta("数字人推流闭环测试。");
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
                const int index = chunk_index * chunk_samples + i;
                chunk.samples[static_cast<size_t>(i)] = static_cast<float>(
                    0.08 * std::sin(2.0 * 3.141592653589793 * 220.0
                                    * index / sample_rate));
            }
            if (!on_audio(std::move(chunk))) return false;
        }
        return true;
    }
};

bool HasAudioAndVideo(const fs::path& path) {
    AVFormatContext* context = nullptr;
    if (avformat_open_input(&context, path.string().c_str(), nullptr, nullptr) < 0) {
        return false;
    }
    const bool info_ok = avformat_find_stream_info(context, nullptr) >= 0;
    bool video = false;
    bool audio = false;
    if (info_ok) {
        for (unsigned i = 0; i < context->nb_streams; ++i) {
            video |= context->streams[i]->codecpar->codec_type
                  == AVMEDIA_TYPE_VIDEO;
            audio |= context->streams[i]->codecpar->codec_type
                  == AVMEDIA_TYPE_AUDIO;
        }
    }
    avformat_close_input(&context);
    return info_ok && video && audio;
}

}  // namespace

int main() {
    const fs::path project(PROJECT_SOURCE_DIR);
    const fs::path avatar_path = project / "assets" / "face.jpg";
    const fs::path face_dir = project / "models" / "face";
    if (!fs::exists(avatar_path) || !fs::exists(face_dir)
        || !fs::exists(project / "models" / "Wav2Lip-SD-GAN-opt.param")
        || !fs::exists(project / "models" / "Wav2Lip-SD-GAN-opt.bin")) {
        std::cout << "[SKIP] integration models/avatar are missing\n";
        return 0;
    }

    cv::Mat avatar = cv::imread(avatar_path.string());
    const fs::path output = fs::temp_directory_path()
        / "digital_human_conversation_stream_test.flv";
    std::error_code ignored;
    fs::remove(output, ignored);

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

    media::StreamPublisherConfig publisher_config;
    publisher_config.url = output.string();
    publisher_config.protocol = media::StreamProtocol::FILE;
    publisher_config.width = 640;
    publisher_config.height = 360;
    publisher_config.fps = 25.0;
    publisher_config.video_encoder = "libx264";
    publisher_config.video_bitrate = 800'000;
    publisher_config.max_video_queue = 30;
    media::StreamPublisher publisher;
    std::string error;
    if (!publisher.Open(publisher_config, error)) {
        std::cerr << "[FAIL] publisher open: " << error << '\n';
        sdk.Stop();
        return 1;
    }

    media::ConversationStreamBridge bridge(sdk, publisher);
    if (!bridge.Start(error)) {
        std::cerr << "[FAIL] bridge start: " << error << '\n';
        publisher.Close(false, error);
        sdk.Stop();
        return 1;
    }

    FixedTextClient text_client;
    FixedTTSClient tts_client;
    dialog::ConversationSession session(text_client, tts_client, bridge);
    dialog::ConversationConfig conversation_config;
    conversation_config.session_id = "stream-integration";
    if (!session.Start(conversation_config, avatar)
        || session.SubmitUserText("开始推流测试") == 0
        || !session.WaitUntilIdle(std::chrono::seconds(15))) {
        std::cerr << "[FAIL] conversation did not complete\n";
        session.Stop(false);
        sdk.Stop();
        return 1;
    }
    session.Stop(true);
    sdk.Stop();

    const auto metrics = publisher.GetMetrics();
    const std::string bridge_error = bridge.GetLastError();
    const bool valid = bridge_error.empty() && fs::exists(output)
        && fs::file_size(output) > 0 && HasAudioAndVideo(output)
        && metrics.video_frames_encoded > 0
        && metrics.audio_frames_encoded > 0;
    if (!valid) {
        std::cerr << "[FAIL] streamed output invalid: " << bridge_error
                  << " video=" << metrics.video_frames_encoded
                  << " audio=" << metrics.audio_frames_encoded << '\n';
        return 1;
    }
    const auto bytes = fs::file_size(output);
    fs::remove(output, ignored);
    std::cout << "[PASS] text -> TTS -> Wav2Lip -> H.264/AAC FLV: video="
              << metrics.video_frames_encoded << " audio="
              << metrics.audio_frames_encoded << " bytes=" << bytes << '\n';
    return 0;
}
