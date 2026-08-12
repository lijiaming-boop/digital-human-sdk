#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
}

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

struct MediaInspection {
    bool has_h264 = false;
    bool has_aac = false;
    int video_packets = 0;
    int audio_packets = 0;
    int width = 0;
    int height = 0;
    int sample_rate = 0;
};

bool InspectMedia(const fs::path& path, MediaInspection& inspection) {
    AVFormatContext* context = nullptr;
    if (avformat_open_input(
            &context, path.string().c_str(), nullptr, nullptr) < 0) {
        return false;
    }
    if (avformat_find_stream_info(context, nullptr) < 0) {
        avformat_close_input(&context);
        return false;
    }
    for (unsigned i = 0; i < context->nb_streams; ++i) {
        const AVCodecParameters* parameters = context->streams[i]->codecpar;
        if (parameters->codec_type == AVMEDIA_TYPE_VIDEO) {
            inspection.has_h264 |= parameters->codec_id == AV_CODEC_ID_H264;
            inspection.width = parameters->width;
            inspection.height = parameters->height;
        } else if (parameters->codec_type == AVMEDIA_TYPE_AUDIO) {
            inspection.has_aac |= parameters->codec_id == AV_CODEC_ID_AAC;
            inspection.sample_rate = parameters->sample_rate;
        }
    }
    AVPacket* packet = av_packet_alloc();
    while (packet && av_read_frame(context, packet) >= 0) {
        const AVMediaType type = context->streams[packet->stream_index]
            ->codecpar->codec_type;
        if (type == AVMEDIA_TYPE_VIDEO) ++inspection.video_packets;
        if (type == AVMEDIA_TYPE_AUDIO) ++inspection.audio_packets;
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
    avformat_close_input(&context);
    return true;
}

media::StreamProtocol ParseProtocol(const std::string& value) {
    if (value == "rtmp") return media::StreamProtocol::RTMP;
    if (value == "rtsp") return media::StreamProtocol::RTSP;
    return media::StreamProtocol::FILE;
}

}  // namespace

int main(int argc, char** argv) {
    if (!network::HttpClient::IsAvailable()) {
        std::cout << "[SKIP] libcurl HTTP client is not available\n";
        return 0;
    }
    if (argc < 3 || argc > 8) {
        std::cerr << "Usage: full_conversation_chain_test <llama_url> <tts_url>"
                     " [output_url] [file|rtmp|rtsp] [model]"
                     " [avatar_path] [user_text]\n";
        return 2;
    }

    const fs::path project(PROJECT_SOURCE_DIR);
    const fs::path avatar_path = argc >= 7
        ? fs::path(argv[6]) : project / "assets" / "face.jpg";
    const fs::path face_dir = project / "models" / "face";
    if (!fs::exists(avatar_path) || !fs::exists(face_dir)
        || !fs::exists(project / "models" / "Wav2Lip-SD-GAN-opt.param")
        || !fs::exists(project / "models" / "Wav2Lip-SD-GAN-opt.bin")) {
        std::cout << "[SKIP] integration models/avatar are missing\n";
        return 0;
    }

    const bool explicit_output = argc >= 4;
    const fs::path default_output = fs::temp_directory_path()
        / "digital_human_full_conversation_chain.flv";
    const std::string output_url = explicit_output
        ? argv[3] : default_output.string();
    const std::string protocol_name = argc >= 5 ? argv[4] : "file";
    const auto protocol = ParseProtocol(protocol_name);
    if (protocol == media::StreamProtocol::FILE) {
        const fs::path output_path(output_url);
        if (!output_path.parent_path().empty()) {
            std::error_code directory_error;
            fs::create_directories(output_path.parent_path(), directory_error);
            if (directory_error) {
                std::cerr << "[FAIL] create output directory: "
                          << directory_error.message() << '\n';
                return 1;
            }
        }
        std::error_code ignored;
        fs::remove(output_path, ignored);
    }

    avatar::AvatarImage avatar_image;
    avatar::AvatarUploadLimits avatar_limits;
    std::string error;
    if (!avatar::LoadAvatarImage(avatar_path.string(), avatar_limits,
                                  avatar_image, error)) {
        std::cerr << "[FAIL] avatar upload: " << error << '\n';
        return 1;
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

    media::StreamPublisherConfig publisher_config;
    publisher_config.url = output_url;
    publisher_config.protocol = protocol;
    publisher_config.width = avatar_image.bgr.cols;
    publisher_config.height = avatar_image.bgr.rows;
    publisher_config.fps = 25.0;
    publisher_config.video_encoder = "libx264";
    publisher_config.video_bitrate = 800'000;
    publisher_config.max_video_queue = 30;
    publisher_config.io_timeout_ms = 5000;
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
    llama_config.temperature = 0.0F;
    llama_config.max_tokens = 48;
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

    std::mutex callback_mutex;
    std::string generated_reply;
    std::string conversation_error;
    dialog::ConversationCallbacks callbacks;
    callbacks.on_text_delta = [&](uint64_t, const std::string& delta) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        generated_reply += delta;
    };
    callbacks.on_error = [&](uint64_t, const std::string& callback_error) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        conversation_error = callback_error;
    };

    dialog::ConversationSession session(text_client, tts_client, bridge);
    dialog::ConversationConfig conversation_config;
    conversation_config.session_id = "full-chain-test";
    conversation_config.system_prompt =
        u8"\u4f60\u662f\u6570\u5b57\u4eba\u6d4b\u8bd5\u52a9\u624b\uff0c\u8bf7\u7528\u4e00\u53e5\u7b80\u77ed\u4e2d\u6587\u56de\u7b54\u3002";
    conversation_config.min_tts_clause_chars = 1;
    const bool started = session.Start(
        conversation_config, avatar_image.bgr, std::move(callbacks));
    const std::string user_text = argc >= 8
        ? argv[7] : u8"\u8bf7\u4ecb\u7ecd\u4e00\u4e0b\u4f60\u81ea\u5df1\u3002";
    const uint64_t task_id = started
        ? session.SubmitUserText(user_text) : 0;
    const bool completed = task_id != 0
        && session.WaitUntilIdle(std::chrono::seconds(120));
    session.Stop(completed);
    sdk.Stop();

    std::string reply;
    std::string callback_error;
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        reply = generated_reply;
        callback_error = conversation_error;
    }
    const auto metrics = publisher.GetMetrics();
    const std::string bridge_error = bridge.GetLastError();
    if (!started || !completed || reply.empty() || !callback_error.empty()
        || !bridge_error.empty() || metrics.video_frames_encoded <= 0
        || metrics.audio_frames_encoded <= 0
        || metrics.packets_written <= 0) {
        std::cerr << "[FAIL] full chain: completed=" << completed
                  << " reply=" << reply
                  << " conversation_error=" << callback_error
                  << " bridge_error=" << bridge_error
                  << " video=" << metrics.video_frames_encoded
                  << " audio=" << metrics.audio_frames_encoded
                  << " packets=" << metrics.packets_written << '\n';
        return 1;
    }

    MediaInspection inspection;
    if (protocol == media::StreamProtocol::FILE) {
        if (!InspectMedia(output_url, inspection) || !inspection.has_h264
            || !inspection.has_aac || inspection.video_packets <= 0
            || inspection.audio_packets <= 0) {
            std::cerr << "[FAIL] output media inspection failed\n";
            return 1;
        }
    }

    std::cout << "[PASS] user -> llama.cpp -> HTTP TTS -> Wav2Lip -> "
                 "H.264/AAC -> " << protocol_name << '\n'
              << "       reply=" << reply << '\n'
              << "       video=" << metrics.video_frames_encoded
              << " audio=" << metrics.audio_frames_encoded
              << " packets=" << metrics.packets_written;
    if (protocol == media::StreamProtocol::FILE) {
        std::cout << " inspected_video=" << inspection.video_packets
                  << " inspected_audio=" << inspection.audio_packets
                  << " resolution=" << inspection.width << 'x'
                  << inspection.height
                  << " sample_rate=" << inspection.sample_rate
                  << " avatar=" << avatar_path.string()
                  << " output=" << output_url;
    }
    std::cout << '\n';

    if (!explicit_output && protocol == media::StreamProtocol::FILE) {
        std::error_code ignored;
        fs::remove(default_output, ignored);
    }
    return 0;
}
