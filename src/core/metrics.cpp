#include "digital_human/metrics.h"

#include <algorithm>
#include <chrono>
#include <sstream>

namespace digital_human {

//--------------
// PercentileTracker
//--------------
PercentileTracker::PercentileTracker(size_t capacity)
    : capacity_(capacity > 0 ? capacity : 1)
    , ring_(capacity_) {
}

void PercentileTracker::record(double value_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t idx = head_ % capacity_;
    ring_[idx] = value_ms;
    head_ = (head_ + 1) % capacity_;
    const uint64_t c = count_.fetch_add(1, std::memory_order_relaxed) + 1;
    sum_ += value_ms;
    if (c == 1) {
        min_ = max_ = value_ms;
    } else {
        if (value_ms < min_) min_ = value_ms;
        if (value_ms > max_) max_ = value_ms;
    }
    dirty_ = true;
}

std::vector<double> PercentileTracker::sorted_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dirty_) return cached_sorted_;
    const uint64_t c = count_.load(std::memory_order_relaxed);
    const size_t n = static_cast<size_t>(std::min<uint64_t>(c, capacity_));
    cached_sorted_.assign(ring_.begin(), ring_.begin() + n);
    std::sort(cached_sorted_.begin(), cached_sorted_.end());
    dirty_ = false;
    return cached_sorted_;
}

double PercentileTracker::p50() const {
    auto s = sorted_snapshot();
    if (s.empty()) return 0;
    return s[s.size() / 2];
}

double PercentileTracker::p95() const {
    auto s = sorted_snapshot();
    if (s.empty()) return 0;
    return s[static_cast<size_t>(s.size() * 0.95)];
}

double PercentileTracker::p99() const {
    auto s = sorted_snapshot();
    if (s.empty()) return 0;
    return s[static_cast<size_t>(s.size() * 0.99)];
}

double PercentileTracker::avg() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t c = count_.load(std::memory_order_relaxed);
    if (c == 0) return 0;
    return sum_ / static_cast<double>(c);
}

double PercentileTracker::min_val() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_.load(std::memory_order_relaxed) > 0 ? min_ : 0;
}

double PercentileTracker::max_val() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_.load(std::memory_order_relaxed) > 0 ? max_ : 0;
}
uint64_t PercentileTracker::count() const { return count_.load(std::memory_order_relaxed); }

void PercentileTracker::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    head_ = 0;
    count_.store(0, std::memory_order_relaxed);
    sum_ = 0;
    min_ = 0;
    max_ = 0;
    dirty_ = true;
    cached_sorted_.clear();
}

//--------------
// MetricsRegistry
//--------------
MetricsRegistry::MetricsRegistry() {}

MetricsRegistry& MetricsRegistry::instance() {
    static MetricsRegistry reg;
    return reg;
}

MetricsSnapshot MetricsRegistry::snapshot() const {
    MetricsSnapshot s;
    s.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    s.total_turns = session_.total_turns.load();
    s.successful_turns = session_.successful_turns.load();
    s.failed_turns = session_.failed_turns.load();
    s.interrupted_turns = session_.interrupted_turns.load();
    s.barge_in_count = session_.barge_in_count.load();
    s.total_sessions = session_.total_sessions.load();
    s.active_sessions = session_.active_sessions.load();
    s.history_length = session_.history_length.load();

    s.llm_total_requests = llm_.total_requests.load();
    s.llm_failed_requests = llm_.failed_requests.load();
    s.llm_cancelled_requests = llm_.cancelled_requests.load();
    s.llm_first_token_p50 = llm_.first_token_latency.p50();
    s.llm_first_token_p95 = llm_.first_token_latency.p95();
    s.llm_first_token_p99 = llm_.first_token_latency.p99();
    s.llm_total_p50 = llm_.total_latency.p50();
    s.llm_total_p95 = llm_.total_latency.p95();
    s.llm_total_p99 = llm_.total_latency.p99();
    s.llm_tokens_per_second = llm_.tokens_per_second.load();
    s.llm_last_http_status = llm_.last_http_status.load();

    s.tts_total_requests = tts_.total_requests.load();
    s.tts_failed_requests = tts_.failed_requests.load();
    s.tts_first_pcm_p50 = tts_.first_pcm_latency.p50();
    s.tts_first_pcm_p95 = tts_.first_pcm_latency.p95();
    s.tts_first_pcm_p99 = tts_.first_pcm_latency.p99();
    s.tts_realtime_factor = tts_.realtime_factor.load();
    s.tts_total_samples = tts_.total_samples.load();

    s.avatar_total_uploads = avatar_.total_uploads.load();
    s.avatar_rejected_uploads = avatar_.rejected_uploads.load();
    s.avatar_total_updates = avatar_.total_updates.load();
    s.avatar_decode_p50 = avatar_.decode_latency.p50();
    s.avatar_decode_p95 = avatar_.decode_latency.p95();
    s.avatar_canvas_width = avatar_.canvas_width.load();
    s.avatar_canvas_height = avatar_.canvas_height.load();

    s.pub_packets_written = publisher_.packets_written.load();
    s.pub_video_frames_encoded = publisher_.video_frames_encoded.load();
    s.pub_video_frames_dropped = publisher_.video_frames_dropped.load();
    s.pub_reconnect_count = publisher_.reconnect_count.load();
    s.pub_encode_p95 = publisher_.encode_latency.p95();

    return s;
}

void MetricsRegistry::reset() {
    session_.total_turns.store(0);
    session_.successful_turns.store(0);
    session_.failed_turns.store(0);
    session_.interrupted_turns.store(0);
    session_.barge_in_count.store(0);
    session_.total_sessions.store(0);
    session_.active_sessions.store(0);
    session_.history_length.store(0);

    llm_.total_requests.store(0);
    llm_.failed_requests.store(0);
    llm_.cancelled_requests.store(0);
    llm_.first_token_latency.reset();
    llm_.total_latency.reset();
    llm_.cancel_latency.reset();
    llm_.tokens_per_second.store(0);
    llm_.last_http_status.store(0);

    tts_.total_requests.store(0);
    tts_.failed_requests.store(0);
    tts_.first_byte_latency.reset();
    tts_.first_pcm_latency.reset();
    tts_.realtime_factor.store(0);
    tts_.total_samples.store(0);

    avatar_.total_uploads.store(0);
    avatar_.rejected_uploads.store(0);
    avatar_.total_updates.store(0);
    avatar_.decode_latency.reset();

    publisher_.packets_written.store(0);
    publisher_.video_frames_encoded.store(0);
    publisher_.video_frames_dropped.store(0);
    publisher_.reconnect_count.store(0);
    publisher_.encode_latency.reset();
}

void MetricsRegistry::record_turn_complete(bool success, bool cancelled) {
    session_.total_turns.fetch_add(1, std::memory_order_relaxed);
    if (success && !cancelled)
        session_.successful_turns.fetch_add(1, std::memory_order_relaxed);
    else if (cancelled)
        session_.interrupted_turns.fetch_add(1, std::memory_order_relaxed);
    else
        session_.failed_turns.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_session_start() {
    session_.total_sessions.fetch_add(1, std::memory_order_relaxed);
    session_.active_sessions.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_session_end() {
    uint64_t v = session_.active_sessions.load(std::memory_order_relaxed);
    while (v > 0 && !session_.active_sessions.compare_exchange_weak(v, v - 1)) {}
}

void MetricsRegistry::record_history_length(size_t length) {
    session_.history_length.store(length, std::memory_order_relaxed);
}

void MetricsRegistry::record_barge_in() {
    session_.barge_in_count.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_llm_request_start() {
    llm_.total_requests.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_llm_first_token(double latency_ms) {
    llm_.first_token_latency.record(latency_ms);
}

void MetricsRegistry::record_llm_complete(double total_ms, double tps, int http_status) {
    llm_.total_latency.record(total_ms);
    llm_.tokens_per_second.store(tps, std::memory_order_relaxed);
    llm_.last_http_status.store(http_status, std::memory_order_relaxed);
}

void MetricsRegistry::record_llm_failed() {
    llm_.failed_requests.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_llm_cancelled(double cancel_ms) {
    llm_.cancelled_requests.fetch_add(1, std::memory_order_relaxed);
    llm_.cancel_latency.record(cancel_ms);
}

void MetricsRegistry::record_tts_request_start() {
    tts_.total_requests.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_tts_first_byte(double latency_ms) {
    tts_.first_byte_latency.record(latency_ms);
}

void MetricsRegistry::record_tts_first_pcm(double latency_ms) {
    tts_.first_pcm_latency.record(latency_ms);
}

void MetricsRegistry::record_tts_complete(double realtime_factor, uint64_t total_samples) {
    tts_.realtime_factor.store(realtime_factor, std::memory_order_relaxed);
    tts_.total_samples.fetch_add(total_samples, std::memory_order_relaxed);
}

void MetricsRegistry::record_tts_failed() {
    tts_.failed_requests.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_avatar_upload(bool accepted, double decode_ms,
                                            int orig_w, int orig_h,
                                            int canvas_w, int canvas_h) {
    avatar_.total_uploads.fetch_add(1, std::memory_order_relaxed);
    if (!accepted)
        avatar_.rejected_uploads.fetch_add(1, std::memory_order_relaxed);
    avatar_.decode_latency.record(decode_ms);
    avatar_.original_width.store(orig_w, std::memory_order_relaxed);
    avatar_.original_height.store(orig_h, std::memory_order_relaxed);
    avatar_.canvas_width.store(canvas_w, std::memory_order_relaxed);
    avatar_.canvas_height.store(canvas_h, std::memory_order_relaxed);
}

void MetricsRegistry::record_avatar_update() {
    avatar_.total_updates.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_publisher_packet() {
    publisher_.packets_written.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_publisher_frame_encoded() {
    publisher_.video_frames_encoded.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_publisher_frame_dropped() {
    publisher_.video_frames_dropped.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_publisher_reconnect() {
    publisher_.reconnect_count.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::record_publisher_encode_latency(double latency_ms) {
    publisher_.encode_latency.record(latency_ms);
}

//--------------
// Prometheus export
//--------------
std::string to_prometheus(const MetricsSnapshot& s) {
    std::ostringstream out;
    out.precision(2);
    out << std::fixed;

    // Session metrics
    out << "# TYPE dh_session_total_turns counter\n";
    out << "dh_session_total_turns " << s.total_turns << "\n";
    out << "# TYPE dh_session_successful_turns counter\n";
    out << "dh_session_successful_turns " << s.successful_turns << "\n";
    out << "# TYPE dh_session_failed_turns counter\n";
    out << "dh_session_failed_turns " << s.failed_turns << "\n";
    out << "# TYPE dh_session_interrupted_turns counter\n";
    out << "dh_session_interrupted_turns " << s.interrupted_turns << "\n";
    out << "# TYPE dh_session_barge_in_count counter\n";
    out << "dh_session_barge_in_count " << s.barge_in_count << "\n";
    out << "# TYPE dh_session_active gauge\n";
    out << "dh_session_active " << s.active_sessions << "\n";
    out << "# TYPE dh_session_history_length gauge\n";
    out << "dh_session_history_length " << s.history_length << "\n";

    // LLM metrics
    out << "# TYPE dh_llm_total_requests counter\n";
    out << "dh_llm_total_requests " << s.llm_total_requests << "\n";
    out << "# TYPE dh_llm_failed_requests counter\n";
    out << "dh_llm_failed_requests " << s.llm_failed_requests << "\n";
    out << "# TYPE dh_llm_first_token_ms summary\n";
    out << "dh_llm_first_token_ms{quantile=\"0.5\"} " << s.llm_first_token_p50 << "\n";
    out << "dh_llm_first_token_ms{quantile=\"0.95\"} " << s.llm_first_token_p95 << "\n";
    out << "dh_llm_first_token_ms{quantile=\"0.99\"} " << s.llm_first_token_p99 << "\n";
    out << "# TYPE dh_llm_total_ms summary\n";
    out << "dh_llm_total_ms{quantile=\"0.5\"} " << s.llm_total_p50 << "\n";
    out << "dh_llm_total_ms{quantile=\"0.95\"} " << s.llm_total_p95 << "\n";
    out << "dh_llm_total_ms{quantile=\"0.99\"} " << s.llm_total_p99 << "\n";
    out << "# TYPE dh_llm_tokens_per_second gauge\n";
    out << "dh_llm_tokens_per_second " << s.llm_tokens_per_second << "\n";

    // TTS metrics
    out << "# TYPE dh_tts_total_requests counter\n";
    out << "dh_tts_total_requests " << s.tts_total_requests << "\n";
    out << "# TYPE dh_tts_failed_requests counter\n";
    out << "dh_tts_failed_requests " << s.tts_failed_requests << "\n";
    out << "# TYPE dh_tts_first_pcm_ms summary\n";
    out << "dh_tts_first_pcm_ms{quantile=\"0.5\"} " << s.tts_first_pcm_p50 << "\n";
    out << "dh_tts_first_pcm_ms{quantile=\"0.95\"} " << s.tts_first_pcm_p95 << "\n";
    out << "dh_tts_first_pcm_ms{quantile=\"0.99\"} " << s.tts_first_pcm_p99 << "\n";
    out << "# TYPE dh_tts_realtime_factor gauge\n";
    out << "dh_tts_realtime_factor " << s.tts_realtime_factor << "\n";

    // Avatar metrics
    out << "# TYPE dh_avatar_total_uploads counter\n";
    out << "dh_avatar_total_uploads " << s.avatar_total_uploads << "\n";
    out << "# TYPE dh_avatar_rejected_uploads counter\n";
    out << "dh_avatar_rejected_uploads " << s.avatar_rejected_uploads << "\n";
    out << "# TYPE dh_avatar_decode_ms summary\n";
    out << "dh_avatar_decode_ms{quantile=\"0.5\"} " << s.avatar_decode_p50 << "\n";
    out << "dh_avatar_decode_ms{quantile=\"0.95\"} " << s.avatar_decode_p95 << "\n";

    // Publisher metrics
    out << "# TYPE dh_pub_packets_written counter\n";
    out << "dh_pub_packets_written " << s.pub_packets_written << "\n";
    out << "# TYPE dh_pub_video_frames_encoded counter\n";
    out << "dh_pub_video_frames_encoded " << s.pub_video_frames_encoded << "\n";
    out << "# TYPE dh_pub_video_frames_dropped counter\n";
    out << "dh_pub_video_frames_dropped " << s.pub_video_frames_dropped << "\n";
    out << "# TYPE dh_pub_reconnect_count counter\n";
    out << "dh_pub_reconnect_count " << s.pub_reconnect_count << "\n";
    out << "# TYPE dh_pub_encode_ms summary\n";
    out << "dh_pub_encode_ms{quantile=\"0.95\"} " << s.pub_encode_p95 << "\n";

    return out.str();
}

}  // namespace digital_human
