#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "core/pipeline.h"
#include "core/thread_base.h"
#include "core/worker_registry.h"
#include "digital_human_sdk.h"

namespace {

class BlockingThread final : public digital_human::core::ThreadBase {
public:
    BlockingThread()
        : ThreadBase("LifecycleSafetyTestWorker") {}

    ~BlockingThread() override {
        Release();
        Shutdown();
    }

    bool WaitUntilEntered(int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
            return entered_;
        });
    }

    void Release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

protected:
    void Run() override {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this]() {
            return released_;
        });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false;
    bool released_ = false;
};

class CooperativeThread final : public digital_human::core::ThreadBase {
public:
    explicit CooperativeThread(const std::string& name)
        : ThreadBase(name) {}

    ~CooperativeThread() override {
        Shutdown();
    }

protected:
    void Run() override {
        while (!IsStopping()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

bool Expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << std::endl;
        return false;
    }
    std::cout << "[PASS] " << message << std::endl;
    return true;
}

bool TestWaitTimeoutCanBeRetried() {
    BlockingThread worker;
    bool ok = Expect(worker.Start(), "ThreadBase starts once");
    ok = Expect(worker.WaitUntilEntered(1000), "worker entered Run") && ok;
    ok = Expect(!worker.Wait(20), "finite Wait reports timeout") && ok;
    worker.Release();
    ok = Expect(worker.Wait(1000), "Wait succeeds after a previous timeout") && ok;
    ok = Expect(worker.IsStopped(), "successful Wait finalizes STOPPED state") && ok;
    return ok;
}

bool TestConcurrentWaitIsSerialized() {
    BlockingThread worker;
    bool ok = Expect(worker.Start(), "concurrent-wait worker starts");
    ok = Expect(worker.WaitUntilEntered(1000), "concurrent-wait worker entered Run") && ok;

    std::atomic<bool> first_result{false};
    std::atomic<bool> second_result{false};
    std::thread first([&]() {
        first_result.store(worker.Wait(1000), std::memory_order_release);
    });
    std::thread second([&]() {
        second_result.store(worker.Wait(1000), std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    worker.Release();
    first.join();
    second.join();

    ok = Expect(first_result.load(std::memory_order_acquire),
                "first concurrent Wait succeeds") && ok;
    ok = Expect(second_result.load(std::memory_order_acquire),
                "second concurrent Wait observes the reclaimed thread") && ok;
    return ok;
}

bool TestWorkerRegistrySharedDeadlineAndRetry() {
    using digital_human::core::WorkerRegistry;

    BlockingThread first;
    BlockingThread second;
    WorkerRegistry registry;
    registry.Add("first", &first);
    registry.Add("second", &second);

    bool ok = Expect(registry.StartAll(), "WorkerRegistry starts registered workers");
    ok = Expect(first.WaitUntilEntered(1000) && second.WaitUntilEntered(1000),
                "registered workers entered Run") && ok;
    registry.RequestStopAll();

    const auto wait_start = std::chrono::steady_clock::now();
    const auto timed_out = registry.WaitAllFor(40);
    const double observed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wait_start).count();
    ok = Expect(!timed_out.all_stopped,
                "WorkerRegistry reports workers that exceed the deadline") && ok;
    ok = Expect(timed_out.workers.size() == 2,
                "WorkerRegistry reports every registered worker") && ok;
    ok = Expect(observed_ms < 150.0,
                "WorkerRegistry shares one timeout budget across workers") && ok;

    first.Release();
    second.Release();
    const auto retried = registry.WaitAllFor(1000);
    ok = Expect(retried.all_stopped,
                "WorkerRegistry wait can be retried after timeout") && ok;
    return ok;
}

bool TestWorkerRegistryStartRollback() {
    using digital_human::core::WorkerRegistry;

    CooperativeThread first("RollbackWorker");
    BlockingThread already_started;
    bool ok = Expect(already_started.Start(),
                     "rollback sentinel is pre-started") && true;
    ok = Expect(already_started.WaitUntilEntered(1000),
                "rollback sentinel entered Run") && ok;

    WorkerRegistry registry;
    registry.Add("first", &first);
    registry.Add("already_started", &already_started);
    ok = Expect(!registry.StartAll(),
                "WorkerRegistry reports partial startup failure") && ok;
    ok = Expect(first.IsStopped(),
                "WorkerRegistry rolls back and joins workers it started") && ok;

    already_started.Release();
    already_started.Stop();
    ok = Expect(already_started.Wait(1000),
                "pre-started sentinel remains caller-owned") && ok;
    return ok;
}

bool TestPipelineLifecycleAndValidation() {
    using digital_human::core::Pipeline;
    using digital_human::core::PipelineConfig;

    bool ok = true;
    {
        Pipeline pipeline;
        ok = Expect(pipeline.Stop(), "Stop before Init is idempotent") && ok;
    }
    {
        Pipeline pipeline;
        PipelineConfig config;
        config.audio_channels = 0;
        ok = Expect(!pipeline.Init(config), "Pipeline rejects zero audio channels") && ok;
    }
    {
        Pipeline pipeline;
        PipelineConfig config;
        config.audio_hop_size = config.audio_frame_size + 1;
        ok = Expect(!pipeline.Init(config), "Pipeline rejects hop size larger than frame size") && ok;
    }
    {
        Pipeline pipeline;
        PipelineConfig config;
        config.output_queue_size = -1;
        ok = Expect(!pipeline.Init(config), "Pipeline rejects negative queue capacity") && ok;
    }
    {
        Pipeline pipeline;
        PipelineConfig config;
        ok = Expect(pipeline.Init(config), "Pipeline accepts the default configuration") && ok;
        const auto initialized_metrics = pipeline.GetMetrics();
        ok = Expect(initialized_metrics.lifecycle_transition_count >= 1,
                    "Pipeline records lifecycle transitions") && ok;
        ok = Expect(initialized_metrics.queue_depths.audio_raw == 0
                    && initialized_metrics.queue_depths.mel_features == 0
                    && initialized_metrics.queue_depths.video_raw == 0
                    && initialized_metrics.queue_depths.processed_faces == 0
                    && initialized_metrics.queue_depths.inference_tasks == 0
                    && initialized_metrics.queue_depths.inference_output == 0
                    && initialized_metrics.queue_depths.output_frames == 0,
                    "Pipeline exposes initial queue depths") && ok;
        ok = Expect(pipeline.Stop(), "initialized Pipeline stops without starting workers") && ok;
        const auto stopped_metrics = pipeline.GetMetrics();
        ok = Expect(stopped_metrics.shutdown_attempt_count == 1,
                    "Pipeline records shutdown attempts") && ok;
        ok = Expect(stopped_metrics.last_shutdown_ms >= 0.0
                    && stopped_metrics.max_shutdown_ms >= stopped_metrics.last_shutdown_ms,
                    "Pipeline records shutdown latency") && ok;
        ok = Expect(!pipeline.Start(), "Pipeline rejects restart after Stop") && ok;
        ok = Expect(pipeline.Stop(), "repeated Pipeline Stop is idempotent") && ok;
    }
    return ok;
}

bool TestSdkValidation() {
    using digital_human::DigitalHumanSDK;
    using digital_human::SDKConfig;
    using digital_human::SDKError;

    bool ok = true;
    {
        DigitalHumanSDK sdk;
        SDKConfig config;
        config.audio_channels = 0;
        ok = Expect(sdk.Init(config) == SDKError::INVALID_CONFIG,
                    "SDK rejects zero audio channels") && ok;
    }
    {
        DigitalHumanSDK sdk;
        SDKConfig config;
        config.mel_context_frames = config.mel_window_frames - 1;
        ok = Expect(sdk.Init(config) == SDKError::INVALID_CONFIG,
                    "SDK rejects a Mel context shorter than the Mel window") && ok;
    }
    {
        DigitalHumanSDK sdk;
        SDKConfig config;
        config.audio_raw_queue_size = -1;
        ok = Expect(sdk.Init(config) == SDKError::INVALID_CONFIG,
                    "SDK rejects negative queue capacity") && ok;
    }
    {
        DigitalHumanSDK sdk;
        ok = Expect(sdk.Stop() == SDKError::OK,
                    "SDK Stop before Init is idempotent") && ok;
    }
    {
        DigitalHumanSDK sdk;
        SDKConfig config;
        ok = Expect(sdk.Init(config) == SDKError::OK,
                    "SDK initializes without model paths") && ok;
        const auto initialized_metrics = sdk.GetMetrics();
        ok = Expect(initialized_metrics.lifecycle_transition_count == 1,
                    "SDK state is the lifecycle transition source of truth") && ok;
        ok = Expect(sdk.Stop() == SDKError::OK,
                    "initialized SDK stops cleanly") && ok;
        const auto stopped_metrics = sdk.GetMetrics();
        ok = Expect(stopped_metrics.lifecycle_transition_count == 3,
                    "SDK records STOPPING and STOPPED transitions") && ok;
        ok = Expect(stopped_metrics.shutdown_attempt_count == 1,
                    "SDK maps Pipeline shutdown metrics") && ok;
        ok = Expect(sdk.Start() == SDKError::ALREADY_TERMINATED,
                    "SDK rejects restart based on terminal SDKState") && ok;
        ok = Expect(sdk.SetInferenceThreads(1) == SDKError::ALREADY_TERMINATED,
                    "SDK rejects configuration mutation after termination") && ok;
    }
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    ok = TestWaitTimeoutCanBeRetried() && ok;
    ok = TestConcurrentWaitIsSerialized() && ok;
    ok = TestWorkerRegistrySharedDeadlineAndRetry() && ok;
    ok = TestWorkerRegistryStartRollback() && ok;
    ok = TestPipelineLifecycleAndValidation() && ok;
    ok = TestSdkValidation() && ok;

    if (!ok) {
        std::cerr << "Lifecycle safety regression test failed." << std::endl;
        return 1;
    }
    std::cout << "Lifecycle safety regression test passed." << std::endl;
    return 0;
}