#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
}

#include <opencv2/core.hpp>

#include "media/stream_publisher.h"

using namespace digital_human;
namespace fs = std::filesystem;

namespace {

bool InspectMedia(const fs::path& path,
                  int& video_streams,
                  int& audio_streams,
                  int64_t& packets) {
    AVFormatContext* context = nullptr;
    if (avformat_open_input(&context, path.string().c_str(), nullptr, nullptr) < 0) {
        return false;
    }
    if (avformat_find_stream_info(context, nullptr) < 0) {
        avformat_close_input(&context);
        return false;
    }
    for (unsigned i = 0; i < context->nb_streams; ++i) {
        if (context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            ++video_streams;
        } else if (context->streams[i]->codecpar->codec_type
                   == AVMEDIA_TYPE_AUDIO) {
            ++audio_streams;
        }
    }
    AVPacket* packet = av_packet_alloc();
    while (packet && av_read_frame(context, packet) >= 0) {
        ++packets;
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
    avformat_close_input(&context);
    return true;
}

}  // namespace

int main() {
    const fs::path output = fs::temp_directory_path()
        / "digital_human_stream_publisher_test.flv";
    std::error_code ignored;
    fs::remove(output, ignored);

    media::StreamPublisherConfig config;
    config.url = output.string();
    config.protocol = media::StreamProtocol::FILE;
    config.width = 320;
    config.height = 240;
    config.fps = 25.0;
    config.video_encoder = "libx264";
    config.video_bitrate = 500'000;
    config.max_video_queue = 60;

    media::StreamPublisher publisher;
    std::string error;
    if (!publisher.Open(config, error)) {
        std::cerr << "[FAIL] publisher open: " << error << '\n';
        return 1;
    }

    constexpr int sample_rate = 16000;
    constexpr int samples_per_video_frame = sample_rate / 25;
    for (int frame_index = 0; frame_index < 25; ++frame_index) {
        const int64_t pts_ms = frame_index * 40;
        std::vector<float> audio(samples_per_video_frame);
        for (int i = 0; i < samples_per_video_frame; ++i) {
            const int sample_index = frame_index * samples_per_video_frame + i;
            audio[static_cast<size_t>(i)] = static_cast<float>(
                0.08 * std::sin(2.0 * 3.141592653589793 * 220.0
                                * sample_index / sample_rate));
        }
        cv::Mat frame(config.height, config.width, CV_8UC3,
                      cv::Scalar(frame_index * 5 % 255,
                                 frame_index * 9 % 255,
                                 frame_index * 13 % 255));
        if (!publisher.PushAudio(audio, pts_ms, error)
            || !publisher.PushVideo(frame, pts_ms, error)) {
            std::cerr << "[FAIL] publisher input: " << error << '\n';
            publisher.Close(false, error);
            return 1;
        }
    }
    if (!publisher.Close(true, error)) {
        std::cerr << "[FAIL] publisher close: " << error << '\n';
        return 1;
    }

    int video_streams = 0;
    int audio_streams = 0;
    int64_t packets = 0;
    const bool inspected = InspectMedia(
        output, video_streams, audio_streams, packets);
    const auto metrics = publisher.GetMetrics();
    const bool valid = inspected && video_streams == 1 && audio_streams == 1
        && packets > 0 && metrics.video_frames_encoded == 25
        && metrics.audio_samples_in == sample_rate
        && metrics.packets_written > 0;
    if (!valid) {
        std::cerr << "[FAIL] invalid FLV: video=" << video_streams
                  << " audio=" << audio_streams << " packets=" << packets
                  << " encoded_video=" << metrics.video_frames_encoded
                  << " input_audio=" << metrics.audio_samples_in
                  << " written=" << metrics.packets_written << '\n';
        return 1;
    }

    const auto bytes = fs::file_size(output);
    fs::remove(output, ignored);
    std::cout << "[PASS] BGR/PCM -> H.264/AAC -> FLV: packets=" << packets
              << ", bytes=" << bytes << '\n';
    return 0;
}
