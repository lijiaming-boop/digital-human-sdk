#include "core/worker_registry.h"

#include <algorithm>
#include <limits>

namespace digital_human {
namespace core {

void WorkerRegistry::Clear() {
    workers_.clear();
}

void WorkerRegistry::Add(const std::string& name, ThreadBase* worker) {
    if (worker != nullptr) {
        workers_.push_back(Entry{name, worker});
    }
}

size_t WorkerRegistry::Size() const {
    return workers_.size();
}

bool WorkerRegistry::StartAll() {
    size_t started_count = 0;
    const auto rollback = [this, &started_count]() {
        for (size_t i = started_count; i > 0; --i) {
            workers_[i - 1].worker->Stop();
        }
        for (size_t i = started_count; i > 0; --i) {
            workers_[i - 1].worker->Wait();
        }
    };

    try {
        for (const auto& entry : workers_) {
            if (!entry.worker->Start()) {
                rollback();
                return false;
            }
            ++started_count;
        }
    } catch (...) {
        rollback();
        return false;
    }
    return true;
}

void WorkerRegistry::RequestStopAll() {
    for (auto it = workers_.rbegin(); it != workers_.rend(); ++it) {
        it->worker->Stop();
    }
}

WorkerShutdownReport WorkerRegistry::WaitAllFor(int timeout_ms) {
    if (timeout_ms < 0) {
        return WaitAll();
    }
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(timeout_ms);
    return WaitAllUntil(&deadline);
}

WorkerShutdownReport WorkerRegistry::WaitAll() {
    return WaitAllUntil(nullptr);
}

WorkerShutdownReport WorkerRegistry::WaitAllUntil(
    const std::chrono::steady_clock::time_point* deadline) {
    WorkerShutdownReport report;
    report.workers.reserve(workers_.size());
    const auto all_start = std::chrono::steady_clock::now();

    for (auto it = workers_.rbegin(); it != workers_.rend(); ++it) {
        const auto worker_start = std::chrono::steady_clock::now();
        bool stopped = false;
        if (deadline == nullptr) {
            stopped = it->worker->Wait();
        } else {
            const auto now = std::chrono::steady_clock::now();
            const auto remaining = *deadline > now
                ? std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now)
                : std::chrono::milliseconds(0);
            const auto capped = std::min<int64_t>(
                remaining.count(), std::numeric_limits<int>::max());
            stopped = it->worker->Wait(static_cast<int>(std::max<int64_t>(0, capped)));
        }
        const auto worker_end = std::chrono::steady_clock::now();
        report.workers.push_back(WorkerWaitResult{
            it->name,
            stopped,
            std::chrono::duration<double, std::milli>(
                worker_end - worker_start).count()
        });
        report.all_stopped = report.all_stopped && stopped;
    }

    report.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - all_start).count();
    return report;
}

}  // namespace core
}  // namespace digital_human