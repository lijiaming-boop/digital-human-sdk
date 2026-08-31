#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/media_clock.h"
#include "core/thread_base.h"
#include "core/thread_safe_queue.h"
#include "core/worker_registry.h"
#include "digital_human/error.h"
#include "digital_human/log_context.h"
#include "digital_human/logger.h"
#include "digital_human/metrics.h"
#include "digital_human/quality_gate.h"
#include "test_support.h"

namespace {

using namespace digital_human;
using namespace digital_human::core;

class CaptureLogger final : public ILogger {
public:
    explicit CaptureLogger(LogLevel minimum) : minimum_(minimum) {}

    LogLevel min_level() const override { return minimum_; }

    void log(const LogRecord& record) override {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back(record);
    }

    std::vector<LogRecord> Records() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_;
    }

private:
    LogLevel minimum_;
    mutable std::mutex mutex_;
    std::vector<LogRecord> records_;
};

class LoopWorker final : public ThreadBase {
public:
    explicit LoopWorker(const std::string& name) : ThreadBase(name) {}

protected:
    void Run() override {
        while (!IsStopping()) {
            iterations.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

public:
    std::atomic<int> iterations{0};
};

class DelayedWorker final : public ThreadBase {
public:
    DelayedWorker() : ThreadBase("delayed") {}

protected:
    void Run() override {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
};

class ThrowingWorker final : public ThreadBase {
public:
    ThrowingWorker() : ThreadBase("throwing") {}

protected:
    void Run() override { throw std::runtime_error("injected"); }
};

void TestQueue(TestSuite& suite) {
    ThreadSafeQueue<int> queue(2, "coverage-queue", 1000);
    suite.Check(queue.TryPush(1), "queue accepts first item");
    suite.Check(queue.TryPush(2), "queue accepts item up to capacity");
    suite.Check(!queue.TryPush(3), "queue reports non-blocking overflow");
    const auto full_metrics = queue.GetMetrics();
    suite.Equal(full_metrics.current_size, size_t{2}, "queue size metric");
    suite.Equal(full_metrics.peak_size, size_t{2}, "queue peak metric");
    suite.Equal(full_metrics.total_overflows, size_t{1},
                "queue overflow metric");

    std::vector<int> batch;
    suite.Equal(queue.TryPopBatch(batch, 1), size_t{1},
                "queue bounded batch pop");
    suite.Equal(batch.front(), 1, "queue preserves FIFO order");
    suite.Check(queue.Emplace(3), "queue emplace succeeds");
    batch.clear();
    suite.Equal(queue.TryPopBatch(batch), size_t{2},
                "queue drains all items");
    suite.Check(batch == std::vector<int>({2, 3}),
                "batch pop preserves remaining order");

    ThreadSafeQueue<int> blocking(1, "blocking");
    suite.Check(blocking.Push(7), "blocking queue seed push");
    auto producer = std::async(std::launch::async, [&]() {
        return blocking.Push(8);
    });
    suite.Check(producer.wait_for(std::chrono::milliseconds(10))
                    == std::future_status::timeout,
                "producer blocks at capacity");
    int value = 0;
    suite.Check(blocking.WaitAndPop(value, 10) && value == 7,
                "consumer releases producer capacity");
    suite.Check(producer.get(), "blocked producer resumes");
    blocking.Stop();
    suite.Check(blocking.WaitAndPop(value, 0) && value == 8,
                "stopped queue drains existing item");
    suite.Check(!blocking.WaitAndPop(value, 0),
                "stopped empty queue rejects pop");
    suite.Check(!blocking.TryPush(9), "stopped queue rejects push");

    ThreadSafeQueue<int> wakeup(1, "wakeup");
    auto consumer = std::async(std::launch::async, [&]() {
        int output = 0;
        return wakeup.WaitAndPop(output, -1);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    wakeup.Stop();
    suite.Check(!consumer.get(), "Stop wakes an empty blocking consumer");
}

void TestWorkers(TestSuite& suite) {
    DelayedWorker delayed;
    suite.Check(delayed.Start(), "thread starts");
    suite.Check(!delayed.Start(), "thread cannot start twice");
    suite.Check(!delayed.Wait(1), "thread wait timeout is reported");
    suite.Check(delayed.Wait(200), "thread can be joined after timeout");
    suite.Check(delayed.GetState() == ThreadState::STOPPED,
                "normal thread reaches STOPPED");

    ThrowingWorker throwing;
    suite.Check(throwing.Start(), "throwing thread starts");
    suite.Check(throwing.Wait(200), "throwing thread is joined");
    suite.Check(throwing.IsError(), "uncaught worker exception reaches ERROR");

    LoopWorker first("first");
    LoopWorker second("second");
    WorkerRegistry registry;
    registry.Add("ignored", nullptr);
    registry.Add("first", &first);
    registry.Add("second", &second);
    suite.Equal(registry.Size(), size_t{2}, "registry ignores null workers");
    suite.Check(registry.StartAll(), "registry starts all workers");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    registry.RequestStopAll();
    const auto report = registry.WaitAllFor(500);
    suite.Check(report.all_stopped, "registry stops all workers");
    suite.Equal(report.workers.size(), size_t{2}, "registry wait report size");
    suite.Check(report.workers[0].name == "second"
                    && report.workers[1].name == "first",
                "registry waits in reverse registration order");
    suite.Check(first.iterations.load() > 0 && second.iterations.load() > 0,
                "registered workers executed");
    registry.Clear();
    suite.Equal(registry.Size(), size_t{0}, "registry clear");
}

void TestLoggingAndMetrics(TestSuite& suite) {
    auto logger = std::make_shared<CaptureLogger>(LogLevel::Warn);
    set_global_logger(logger);
    log_message(LogLevel::Info, "coverage", "filtered");
    {
        LogContext outer("session-A", 7);
        log_message(LogLevel::Warn, "coverage", "outer");
        {
            LogContext inner("session-B", 8);
            log_message(LogLevel::Error, "coverage", "inner");
        }
        log_message(LogLevel::Warn, "coverage", "restored");
    }
    const auto records = logger->Records();
    suite.Equal(records.size(), size_t{3}, "logger applies level filtering");
    suite.Check(records[0].session_id == "session-A"
                    && records[0].turn_id == 7,
                "logger captures outer context");
    suite.Check(records[1].session_id == "session-B"
                    && records[1].turn_id == 8,
                "logger captures nested context");
    suite.Check(records[2].session_id == "session-A",
                "logger restores previous context");
    suite.Check(LogContext::current() == nullptr,
                "log context is cleared after scope");
    set_global_logger(nullptr);
    suite.Check(static_cast<bool>(get_global_logger()),
                "null logger restores default logger");

    PercentileTracker tracker(8);
    tracker.record(10.0);
    tracker.record(20.0);
    tracker.record(30.0);
    suite.Near(tracker.p50(), 20.0, 0.001, "percentile p50");
    suite.Near(tracker.p95(), 30.0, 0.001, "percentile p95");
    suite.Near(tracker.avg(), 20.0, 0.001, "percentile average");
    suite.Near(tracker.min_val(), 10.0, 0.001, "percentile minimum");
    suite.Near(tracker.max_val(), 30.0, 0.001, "percentile maximum");
    tracker.reset();
    suite.Equal(tracker.count(), uint64_t{0}, "percentile reset");

    auto& metrics = MetricsRegistry::instance();
    metrics.reset();
    metrics.record_session_start();
    metrics.record_turn_complete(true, false);
    metrics.record_turn_complete(false, true);
    metrics.record_history_length(4);
    metrics.record_barge_in();
    metrics.record_llm_request_start();
    metrics.record_llm_first_token(12.0);
    metrics.record_llm_complete(40.0, 25.0, 200);
    metrics.record_tts_request_start();
    metrics.record_tts_first_pcm(15.0);
    metrics.record_tts_complete(0.5, 1600);
    metrics.record_avatar_upload(true, 3.0, 10, 20, 32, 32);
    metrics.record_avatar_update();
    metrics.record_publisher_packet();
    metrics.record_publisher_frame_encoded();
    metrics.record_publisher_frame_dropped();
    metrics.record_publisher_reconnect();
    metrics.record_publisher_encode_latency(5.0);
    metrics.record_session_end();
    const auto snapshot = metrics.snapshot();
    suite.Equal(snapshot.total_turns, uint64_t{2}, "turn counter snapshot");
    suite.Equal(snapshot.successful_turns, uint64_t{1},
                "successful turn snapshot");
    suite.Equal(snapshot.interrupted_turns, uint64_t{1},
                "interrupted turn snapshot");
    suite.Equal(snapshot.active_sessions, uint64_t{0},
                "active session counter is balanced");
    suite.Near(snapshot.llm_first_token_p95, 12.0, 0.001,
               "LLM latency snapshot");
    suite.Equal(snapshot.pub_reconnect_count, uint64_t{1},
                "publisher reconnect snapshot");
    const auto prometheus = to_prometheus(snapshot);
    suite.Check(prometheus.find("dh_session_total_turns 2")
                    != std::string::npos,
                "Prometheus output contains session metric");
}

void TestClockQualityAndError(TestSuite& suite) {
    MediaClock clock;
    MediaClockConfig invalid;
    invalid.audio_sample_rate = 0;
    suite.Check(!clock.Configure(invalid), "media clock rejects invalid config");

    MediaClockConfig config;
    config.audio_sample_rate = 1000;
    config.expected_video_interval_ms = 40.0;
    config.drift_correction_threshold_ms = 20.0;
    config.max_correction_step_ms = 4.0;
    suite.Check(clock.Configure(config), "media clock accepts valid config");
    for (int i = 1; i <= 8; ++i) {
        clock.AdvanceAudioFrames(40);
        clock.ObserveVideoPts(i * 44.0);
    }
    auto clock_snapshot = clock.Snapshot();
    suite.Check(clock_snapshot.corrections > 0,
                "media clock corrects accumulated drift");
    suite.Near(clock_snapshot.jitter_p95_ms, 4.0, 0.001,
               "media clock jitter percentile");
    clock.ObserveVideoPts(1.0);
    suite.Check(!clock.Snapshot().pts_monotonic,
                "media clock detects backward PTS");
    clock.Reset();
    suite.Equal(clock.Snapshot().observations, uint64_t{0},
                "media clock reset");

    QualityGateSample sample;
    sample.fps_p95 = 25.0;
    sample.frame_latency_p95_ms = 60.0;
    sample.drop_rate = 0.01;
    sample.av_sync_p95_ms = 40.0;
    sample.av_duration_delta_ms = -50.0;
    sample.mouth_psnr_db = 30.0;
    sample.mouth_ssim = 0.95;
    suite.Check(QualityGate::Evaluate(sample).passed,
                "quality gate accepts a valid sample");
    sample.mouth_ssim = 0.1;
    sample.pts_monotonic = false;
    const auto gate = QualityGate::Evaluate(sample);
    suite.Check(!gate.passed && gate.violations.size() == 2,
                "quality gate reports independent failures");

    const auto error = Error::Make(
        ErrorCategory::Network, "network.http", "request failed", "timeout");
    suite.Check(!error.ok(), "non-OK error reports failure");
    suite.Check(error.to_string().find("timeout") != std::string::npos,
                "error string contains cause");
    suite.Check(Error::OK().ok(), "OK error reports success");
    suite.Check(std::string(category_name(ErrorCategory::Cancelled))
                    == "cancelled",
                "error category name mapping");
}

}  // namespace

int main() {
    TestSuite suite;
    TestQueue(suite);
    TestWorkers(suite);
    TestLoggingAndMetrics(suite);
    TestClockQualityAndError(suite);
    return suite.Finish("core_infrastructure_coverage_test");
}
