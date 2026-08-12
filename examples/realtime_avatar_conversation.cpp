#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>

#include <opencv2/imgproc.hpp>

#include "avatar/avatar_image.h"
#include "dialog/conversation_session.h"
#include "dialog/llama_cpp_text_generation_client.h"
#include "digital_human_sdk.h"
#include "media/conversation_stream_bridge.h"
#include "media/stream_publisher.h"
#include "network/http_client.h"
#include "tts/tts_client.h"

using namespace digital_human;
namespace fs = std::filesystem;

namespace {

media::StreamProtocol ParseProtocol(const std::string& value,
                                    bool& valid) {
    valid = true;
    if (value == "file") return media::StreamProtocol::FILE;
    if (value == "rtmp") return media::StreamProtocol::RTMP;
    if (value == "rtsp") return media::StreamProtocol::RTSP;
    valid = false;
    return media::StreamProtocol::FILE;
}

bool LoadAvatarForCanvas(const std::string& path,
                         const cv::Size& canvas_size,
                         cv::Mat& avatar_frame,
                         std::string& error) {
    avatar::AvatarImage image;
    if (!avatar::LoadAvatarImage(path, {}, image, error)) return false;
    if (canvas_size.width > 0 && canvas_size.height > 0
        && image.bgr.size() != canvas_size) {
        cv::resize(image.bgr, avatar_frame, canvas_size, 0.0, 0.0,
                   cv::INTER_AREA);
    } else {
        avatar_frame = image.bgr;
    }
    return true;
}

bool NormalizeEncoderCanvas(cv::Mat& avatar_frame, std::string& error) {
    if (avatar_frame.cols < 2 || avatar_frame.rows < 2) {
        error = "avatar dimensions must be at least 2x2";
        return false;
    }
    const cv::Size even_size(avatar_frame.cols & ~1, avatar_frame.rows & ~1);
    if (avatar_frame.size() != even_size) {
        cv::Mat resized;
        cv::resize(avatar_frame, resized, even_size, 0.0, 0.0,
                   cv::INTER_AREA);
        avatar_frame = std::move(resized);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr << "Usage: realtime_avatar_conversation <llama_url> <tts_url>"
                     " <avatar_path> <output_url> [model]"
                     " [file|rtmp|rtsp]\n";
        return 2;
    }
    if (!network::HttpClient::IsAvailable()) {
        std::cerr << "[FAIL] libcurl HTTP client is unavailable\n";
        return 1;
    }

    const fs::path project(PROJECT_SOURCE_DIR);
    const fs::path face_model_dir = project / "models" / "face";
    cv::Mat avatar_frame;
    std::string error;
    if (!LoadAvatarForCanvas(argv[3], {}, avatar_frame, error)
        || !NormalizeEncoderCanvas(avatar_frame, error)) {
        std::cerr << "[FAIL] avatar upload: " << error << '\n';
        return 1;
    }

    bool protocol_valid = true;
    const media::StreamProtocol protocol = argc >= 7
        ? ParseProtocol(argv[6], protocol_valid)
        : media::StreamProtocol::FILE;
    if (!protocol_valid) {
        std::cerr << "[FAIL] protocol must be file, rtmp, or rtsp\n";
        return 2;
    }

    SDKConfig sdk_config;
    sdk_config.lipsync_model_dir = (project / "models").string();
    sdk_config.face_model_dir = face_model_dir.string();
    sdk_config.enable_frame_pacing = false;
    sdk_config.output_queue_size = 30;
    DigitalHumanSDK sdk;
    if (sdk.Init(sdk_config) != SDKError::OK
        || sdk.Start() != SDKError::OK) {
        std::cerr << "[FAIL] SDK start: " << sdk.GetLastError() << '\n';
        return 1;
    }

    media::StreamPublisherConfig publisher_config;
    publisher_config.url = argv[4];
    publisher_config.protocol = protocol;
    publisher_config.width = avatar_frame.cols;
    publisher_config.height = avatar_frame.rows;
    publisher_config.fps = 25.0;
    publisher_config.video_encoder = "libx264";
    publisher_config.video_bitrate = 800'000;
    publisher_config.max_video_queue = 30;
    media::StreamPublisher publisher;
    if (!publisher.Open(publisher_config, error)) {
        std::cerr << "[FAIL] publisher open: " << error << '\n';
        sdk.Stop();
        return 1;
    }

    media::ConversationStreamBridge bridge(sdk, publisher);
    if (!bridge.Start(error)) {
        std::cerr << "[FAIL] stream bridge start: " << error << '\n';
        publisher.Close(false, error);
        sdk.Stop();
        return 1;
    }

    dialog::LlamaCppTextGenerationConfig llama_config;
    llama_config.endpoint = argv[1];
    if (argc >= 6) llama_config.model = argv[5];
    llama_config.temperature = 0.2F;
    llama_config.max_tokens = 96;
    llama_config.stream = true;
    llama_config.enable_thinking = false;
    dialog::LlamaCppTextGenerationClient text_client(llama_config);

    tts::HttpTTSConfig tts_config;
    tts_config.endpoint = argv[2];
    tts_config.response_format = tts::TTSAudioFormat::PCM_S16LE;
    tts_config.sample_rate = 16000;
    tts_config.channels = 1;
    tts_config.chunk_samples = 1600;
    tts::HttpTTSClient tts_client(tts_config);

    std::mutex output_mutex;
    std::atomic<bool> callback_failed{false};
    dialog::ConversationCallbacks callbacks;
    callbacks.on_text_delta = [&](uint64_t, const std::string& delta) {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << delta << std::flush;
    };
    callbacks.on_reply_ready = [&](uint64_t, const std::string&) {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << '\n';
    };
    callbacks.on_error = [&](uint64_t, const std::string& callback_error) {
        std::lock_guard<std::mutex> lock(output_mutex);
        callback_failed.store(true);
        std::cerr << "\n[ERROR] " << callback_error << '\n';
    };

    dialog::ConversationConfig conversation_config;
    conversation_config.session_id = "realtime-avatar-conversation";
    conversation_config.system_prompt =
        u8"\u4f60\u662f\u5b9e\u65f6\u6570\u5b57\u4eba\u52a9\u624b\uff0c\u8bf7\u7528\u7b80\u6d01\u81ea\u7136\u7684\u4e2d\u6587\u56de\u7b54\u3002";
    conversation_config.min_tts_clause_chars = 4;
    dialog::ConversationSession session(text_client, tts_client, bridge);
    if (!session.Start(conversation_config, avatar_frame,
                       std::move(callbacks))) {
        std::cerr << "[FAIL] conversation session start\n";
        bridge.Finish();
        sdk.Stop();
        return 1;
    }

    std::cout << "Realtime avatar conversation started.\n"
                 "Enter text, /avatar <jpeg-or-png-path>, or /quit.\n";
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "/quit") break;
        if (line.rfind("/avatar ", 0) == 0) {
            cv::Mat uploaded;
            const std::string path = line.substr(8);
            if (!LoadAvatarForCanvas(path, avatar_frame.size(), uploaded, error)
                || !session.UpdateAvatar(uploaded)) {
                std::cerr << "[FAIL] avatar update: "
                          << (error.empty() ? "session rejected avatar" : error)
                          << '\n';
            } else {
                std::cout << "[OK] avatar updated: " << path << '\n';
            }
            continue;
        }
        if (line.empty()) continue;

        callback_failed.store(false);
        const uint64_t turn_id = session.SubmitUserText(line);
        if (turn_id == 0) {
            std::cerr << "[FAIL] session is busy or stopped\n";
            continue;
        }
        if (!session.WaitUntilIdle(std::chrono::seconds(120))) {
            std::cerr << "[FAIL] conversation turn timed out\n";
            session.Interrupt();
            session.WaitUntilIdle(std::chrono::seconds(5));
        } else if (!callback_failed.load()) {
            std::cout << "[OK] turn " << turn_id << " complete\n";
        }
    }

    session.Stop(true);
    sdk.Stop();
    const media::StreamPublisherMetrics metrics = publisher.GetMetrics();
    if (!bridge.GetLastError().empty() || metrics.video_frames_encoded <= 0
        || metrics.audio_frames_encoded <= 0 || metrics.packets_written <= 0) {
        std::cerr << "[FAIL] media output: " << bridge.GetLastError() << '\n';
        return 1;
    }
    std::cout << "[PASS] realtime conversation output=" << argv[4]
              << " video=" << metrics.video_frames_encoded
              << " audio=" << metrics.audio_frames_encoded
              << " packets=" << metrics.packets_written << '\n';
    return 0;
}
