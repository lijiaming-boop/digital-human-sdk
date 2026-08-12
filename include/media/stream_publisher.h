#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace digital_human {
namespace media {

enum class StreamProtocol {
    AUTO,
    RTMP,
    RTSP,
    FILE,
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
    int io_timeout_ms = 5000;
    /// 音频 PushAudio 在队列满时的最长等待毫秒数（P0 停止语义）。
    /// 超过该时间仍无法入队视为慢消费者不可恢复，置 failed 并返回 false，
    /// 避免会话停止时无限阻塞。0 表示不等待（立即失败）。
    int max_audio_push_wait_ms = 30000;
    bool rtsp_tcp = true;
};

struct StreamPublisherMetrics {
    int64_t video_frames_in = 0;
    int64_t video_frames_encoded = 0;
    int64_t video_frames_dropped = 0;
    int64_t audio_samples_in = 0;
    int64_t audio_frames_encoded = 0;
    int64_t packets_written = 0;
};

/// Thread-safe BGR/PCM encoder and muxer.
///
/// RTMP uses FLV + H.264 + AAC. RTSP uses the FFmpeg RTSP muxer and requires
/// an RTSP server that accepts publishing. FILE is primarily used for tests.
class StreamPublisher {
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
