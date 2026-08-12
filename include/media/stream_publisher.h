#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "digital_human/export.h"

namespace digital_human {
namespace media {

enum class StreamProtocol {
    AUTO,
    RTMP,
    RTSP,
    FILE,
};

enum class ReconnectAudioPolicy {
    DROP,
    RETAIN,
    FAIL_SESSION,
};

struct StreamPublisherConfig {
    std::string url;
    StreamProtocol protocol = StreamProtocol::AUTO;

    int width = 0;
    int height = 0;
    double fps = 25.0;
    int video_bitrate = 2'000'000;
    int gop_size = 50;
    std::string video_encoder;  // empty: hardware encoders then libx264
    std::string encoder_preset = "veryfast";

    int input_audio_sample_rate = 16000;
    int input_audio_channels = 1;
    int output_audio_sample_rate = 48000;
    int output_audio_channels = 1;
    int audio_bitrate = 96'000;
    std::string audio_encoder = "aac";

    size_t max_video_queue = 12;
    size_t max_audio_queue = 64;
    /// Drop queued video when its real PTS latency exceeds this bound.
    int max_video_queue_latency_ms = 500;
    int io_timeout_ms = 5000;
    /// 音频 PushAudio 在队列满时的最长等待毫秒数（P0 停止语义）。
    /// 超过该时间仍无法入队视为慢消费者不可恢复，置 failed 并返回 false，
    /// 避免会话停止时无限阻塞。0 表示不等待（立即失败）。
    int max_audio_push_wait_ms = 30000;
    bool rtsp_tcp = true;

    bool enable_reconnect = true;
    int reconnect_initial_backoff_ms = 250;
    int reconnect_max_backoff_ms = 4000;
    int reconnect_window_ms = 30000;
    ReconnectAudioPolicy reconnect_audio_policy =
        ReconnectAudioPolicy::RETAIN;
    int max_retained_audio_ms = 2000;

    /// Deterministic fault injection for storage/network error tests.
    /// Negative disables it; zero fails before the first muxed packet.
    int64_t debug_fail_after_packets = -1;
};

struct StreamPublisherMetrics {
    int64_t video_frames_in = 0;
    int64_t video_frames_encoded = 0;
    int64_t video_frames_dropped = 0;
    int64_t audio_samples_in = 0;
    int64_t audio_frames_encoded = 0;
    int64_t packets_written = 0;
    int64_t reconnect_attempts = 0;
    int64_t reconnect_successes = 0;
    int64_t reconnect_failures = 0;
    int64_t max_video_queue_latency_ms = 0;
    std::string selected_video_encoder;
    bool hardware_video_encoder = false;
};

/// Thread-safe BGR/PCM encoder and muxer.
///
/// RTMP uses FLV + H.264 + AAC. RTSP uses the FFmpeg RTSP muxer and requires
/// an RTSP server that accepts publishing. FILE is primarily used for tests.
class DH_API StreamPublisher {
public:
    StreamPublisher();
    ~StreamPublisher();

    StreamPublisher(const StreamPublisher&) = delete;
    StreamPublisher& operator=(const StreamPublisher&) = delete;

    bool Open(const StreamPublisherConfig& config, std::string& error);
    bool PushVideo(const cv::Mat& bgr_frame,
                   int64_t pts_ms,
                   std::string& error);
    bool PushAudio(const std::vector<float>& interleaved_pcm,
                   int64_t pts_ms,
                   std::string& error);
    bool Close(bool drain, std::string& error);

    bool IsOpen() const;
    std::string GetLastError() const;
    StreamPublisherMetrics GetMetrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace media
}  // namespace digital_human
