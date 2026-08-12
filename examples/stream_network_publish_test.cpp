#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "media/stream_publisher.h"

using namespace digital_human;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: stream_network_publish_test <rtmp|rtsp> <url>\n";
        return 2;
    }
    const std::string protocol = argv[1];
    media::StreamPublisherConfig config;
    config.url = argv[2];
    config.protocol = protocol == "rtmp"
        ? media::StreamProtocol::RTMP : media::StreamProtocol::RTSP;
    config.width = 320;
    config.height = 240;
    config.fps = 25.0;
    config.video_encoder = "libx264";
    config.video_bitrate = 500'000;
    config.max_video_queue = 60;
    config.io_timeout_ms = 3000;

    media::StreamPublisher publisher;
    std::string error;
    if (!publisher.Open(config, error)) {
        std::cerr << "[FAIL] " << protocol << " open: " << error << '\n';
        return 1;
    }
    constexpr int sample_rate = 16000;
    constexpr int samples_per_frame = sample_rate / 25;
    for (int frame_index = 0; frame_index < 30; ++frame_index) {
        const int64_t pts_ms = frame_index * 40;
        std::vector<float> audio(samples_per_frame);
        for (int i = 0; i < samples_per_frame; ++i) {
            const int index = frame_index * samples_per_frame + i;
            audio[static_cast<size_t>(i)] = static_cast<float>(
                0.05 * std::sin(2.0 * 3.141592653589793 * 220.0
                                * index / sample_rate));
        }
        cv::Mat frame(config.height, config.width, CV_8UC3,
                      cv::Scalar(frame_index * 7 % 255,
                                 frame_index * 11 % 255,
                                 frame_index * 17 % 255));
        if (!publisher.PushAudio(audio, pts_ms, error)
            || !publisher.PushVideo(frame, pts_ms, error)) {
            std::cerr << "[FAIL] " << protocol << " input: " << error << '\n';
            publisher.Close(false, error);
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    if (!publisher.Close(true, error)) {
        std::cerr << "[FAIL] " << protocol << " close: " << error << '\n';
        return 1;
    }
    const auto metrics = publisher.GetMetrics();
    if (metrics.video_frames_encoded == 0
        || metrics.audio_frames_encoded == 0
        || metrics.packets_written == 0) {
        std::cerr << "[FAIL] " << protocol << " wrote no media packets\n";
        return 1;
    }
    std::cout << "[PASS] " << protocol << " publish: video="
              << metrics.video_frames_encoded << " audio="
              << metrics.audio_frames_encoded << " packets="
              << metrics.packets_written << '\n';
    return 0;
}
