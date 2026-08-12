#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "media/stream_publisher.h"

namespace fs = std::filesystem;
using digital_human::media::StreamProtocol;
using digital_human::media::StreamPublisher;
using digital_human::media::StreamPublisherConfig;

namespace {

StreamPublisherConfig ConfigFor(const fs::path& output) {
    StreamPublisherConfig config;
    config.url = output.string();
    config.protocol = StreamProtocol::FILE;
    config.width = 64;
    config.height = 64;
    config.fps = 25.0;
    config.video_encoder = "libx264";
    return config;
}

bool DiskFullFailure(const fs::path& output) {
    auto config = ConfigFor(output);
    config.debug_fail_after_packets = 0;
    StreamPublisher publisher;
    std::string error;
    if (!publisher.Open(config, error)) return false;
    std::vector<float> audio(640, 0.0f);
    cv::Mat video(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    if (!publisher.PushAudio(audio, 0, error)
        || !publisher.PushVideo(video, 0, error)) {
        return false;
    }
    const bool closed = publisher.Close(true, error);
    return !closed && error.find("ENOSPC") != std::string::npos;
}

bool SlowConsumerFailure(const fs::path& output) {
    auto config = ConfigFor(output);
    config.max_audio_queue = 1;
    config.max_audio_push_wait_ms = 10;
    StreamPublisher publisher;
    std::string error;
    if (!publisher.Open(config, error)) return false;
    std::vector<float> audio(640, 0.0f);
    if (!publisher.PushAudio(audio, 0, error)) return false;
    const auto started = std::chrono::steady_clock::now();
    const bool accepted = publisher.PushAudio(audio, 40, error);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    std::string close_error;
    publisher.Close(false, close_error);
    return !accepted && elapsed >= 5
        && error.find("slow consumer") != std::string::npos;
}

}  // namespace

int main() {
    const fs::path temp = fs::temp_directory_path();
    const fs::path disk_full = temp / "dh_publisher_disk_full.flv";
    const fs::path slow = temp / "dh_publisher_slow.flv";
    std::error_code ignored;
    fs::remove(disk_full, ignored);
    fs::remove(slow, ignored);

    const bool disk_ok = DiskFullFailure(disk_full);
    const bool slow_ok = SlowConsumerFailure(slow);
    fs::remove(disk_full, ignored);
    fs::remove(slow, ignored);
    if (!disk_ok || !slow_ok) {
        std::cerr << "publisher resilience failed: disk=" << disk_ok
                  << " slow=" << slow_ok << '\n';
        return 1;
    }
    std::cout << "publisher resilience tests passed\n";
    return 0;
}
