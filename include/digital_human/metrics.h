#pragma once

/// @file metrics.h
/// @brief turn 级指标体系（P1-3.9c）
///
/// 覆盖会话/LLM/TTS/Avatar/Publisher 五层的 turn 级追踪。
/// 提供 p50/p95/p99 分位数统计器和统一快照 API。

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "digital_human/export.h"

namespace digital_human {

/// 分位数统计器（环形缓冲 + 延迟排序计算 p50/p95/p99）
class DH_API PercentileTracker {
public:
    explicit PercentileTracker(size_t capacity = 1024);

    void record(double value_ms);
    double p50() const;
    double p95() const;
    double p99() const;
    double avg() const;
    double min_val() const;
    double max_val() const;
    uint64_t count() const;
    void reset();

private:
    std::vector<double> sorted_snapshot() const;

    size_t              capacity_;
    std::vector<double> ring_;
    size_t              head_ = 0;
    std::atomic<uint64_t> count_{0};
    double              sum_ = 0.0;
    double              min_ = 0.0;
    double              max_ = 0.0;
    mutable bool        dirty_ = true;
    mutable std::vector<double> cached_sorted_;
    mutable std::mutex  mutex_;
};

/// 会话级指标
struct DH_API SessionMetrics {
    std::atomic<uint64_t> total_turns{0};
    std::atomic<uint64_t> successful_turns{0};
    std::atomic<uint64_t> failed_turns{0};
    std::atomic<uint64_t> interrupted_turns{0};
    std::atomic<uint64_t> barge_in_count{0};       ///< VAD 触发的 barge-in 次数
    std::atomic<uint64_t> total_sessions{0};
    std::atomic<uint64_t> active_sessions{0};
    std::atomic<size_t>   history_length{0};
};

/// LLM 指标
struct DH_API LLMMetrics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> cancelled_requests{0};
    PercentileTracker     first_token_latency;
    PercentileTracker     total_latency;
    PercentileTracker     cancel_latency;
    std::atomic<double>   tokens_per_second{0};
    std::atomic<int>      last_http_status{0};

    LLMMetrics() : first_token_latency(512), total_latency(512), cancel_latency(128) {}
};

/// TTS 指标
struct DH_API TTSMetrics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    PercentileTracker     first_byte_latency;
    PercentileTracker     first_pcm_latency;
    std::atomic<double>   realtime_factor{0};
    std::atomic<uint64_t> total_samples{0};

    TTSMetrics() : first_byte_latency(512), first_pcm_latency(512) {}
};

/// Avatar 指标
struct DH_API AvatarMetrics {
    std::atomic<uint64_t> total_uploads{0};
    std::atomic<uint64_t> rejected_uploads{0};
    std::atomic<uint64_t> total_updates{0};
    PercentileTracker     decode_latency;
    std::atomic<int>      original_width{0};
    std::atomic<int>      original_height{0};
    std::atomic<int>      canvas_width{0};
    std::atomic<int>      canvas_height{0};

    AvatarMetrics() : decode_latency(256) {}
};

/// Publisher 指标
struct DH_API PublisherMetrics {
    std::atomic<uint64_t> packets_written{0};
    std::atomic<uint64_t> video_frames_encoded{0};
    std::atomic<uint64_t> video_frames_dropped{0};
    std::atomic<uint64_t> audio_frames_encoded{0};
    std::atomic<uint64_t> reconnect_count{0};
    PercentileTracker     encode_latency;

    PublisherMetrics() : encode_latency(512) {}
};

/// 统一指标快照（聚合所有层，纯值拷贝）
struct DH_API MetricsSnapshot {
    int64_t timestamp_ms = 0;

    // Session
    uint64_t total_turns = 0;
    uint64_t successful_turns = 0;
    uint64_t failed_turns = 0;
    uint64_t interrupted_turns = 0;
    uint64_t barge_in_count = 0;
    uint64_t total_sessions = 0;
    uint64_t active_sessions = 0;
    size_t   history_length = 0;

    // LLM
    uint64_t llm_total_requests = 0;
    uint64_t llm_failed_requests = 0;
    uint64_t llm_cancelled_requests = 0;
    double   llm_first_token_p50 = 0;
    double   llm_first_token_p95 = 0;
    double   llm_first_token_p99 = 0;
    double   llm_total_p50 = 0;
    double   llm_total_p95 = 0;
    double   llm_total_p99 = 0;
    double   llm_tokens_per_second = 0;
    int      llm_last_http_status = 0;

    // TTS
    uint64_t tts_total_requests = 0;
    uint64_t tts_failed_requests = 0;
    double   tts_first_pcm_p50 = 0;
    double   tts_first_pcm_p95 = 0;
    double   tts_first_pcm_p99 = 0;
    double   tts_realtime_factor = 0;
    uint64_t tts_total_samples = 0;

    // Avatar
    uint64_t avatar_total_uploads = 0;
    uint64_t avatar_rejected_uploads = 0;
    uint64_t avatar_total_updates = 0;
    double   avatar_decode_p50 = 0;
    double   avatar_decode_p95 = 0;
    int      avatar_canvas_width = 0;
    int      avatar_canvas_height = 0;

    // Publisher
    uint64_t pub_packets_written = 0;
    uint64_t pub_video_frames_encoded = 0;
    uint64_t pub_video_frames_dropped = 0;
    uint64_t pub_reconnect_count = 0;
    double   pub_encode_p95 = 0;
};

/// 指标注册中心：全局单例，各模块注册自己的指标采集器
class DH_API MetricsRegistry {
public:
    static MetricsRegistry& instance();

    /// 获取当前快照（线程安全）
    MetricsSnapshot snapshot() const;

    /// 重置所有计数器（用于测试）
    void reset();

    // 各模块注册自己的指标更新接口
    void record_turn_complete(bool success, bool cancelled);
    void record_session_start();
    void record_session_end();
    void record_history_length(size_t length);
    void record_barge_in();
    void record_llm_request_start();
    void record_llm_first_token(double latency_ms);
    void record_llm_complete(double total_ms, double tps, int http_status);
    void record_llm_failed();
    void record_llm_cancelled(double cancel_ms);
    void record_tts_request_start();
    void record_tts_first_byte(double latency_ms);
    void record_tts_first_pcm(double latency_ms);
    void record_tts_complete(double realtime_factor, uint64_t total_samples);
    void record_tts_failed();
    void record_avatar_upload(bool accepted, double decode_ms,
                               int orig_w, int orig_h, int canvas_w, int canvas_h);
    void record_avatar_update();
    void record_publisher_packet();
    void record_publisher_frame_encoded();
    void record_publisher_frame_dropped();
    void record_publisher_reconnect();
    void record_publisher_encode_latency(double latency_ms);

    // 直接访问（内部使用）
    SessionMetrics&   session()       { return session_; }
    LLMMetrics&       llm()           { return llm_; }
    TTSMetrics&       tts()           { return tts_; }
    AvatarMetrics&    avatar()        { return avatar_; }
    PublisherMetrics& publisher()     { return publisher_; }

private:
    MetricsRegistry();

    SessionMetrics   session_;
    LLMMetrics       llm_;
    TTSMetrics       tts_;
    AvatarMetrics    avatar_;
    PublisherMetrics publisher_;
};

/// 将 MetricsSnapshot 转为 Prometheus 文本格式
DH_API std::string to_prometheus(const MetricsSnapshot& snapshot);

}  // namespace digital_human
