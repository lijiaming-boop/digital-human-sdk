#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "core/thread_base.h"

namespace digital_human {
namespace core {

struct WorkerWaitResult {
    std::string name;
    bool stopped = false;
    double wait_ms = 0.0;
};

struct WorkerShutdownReport {
    bool all_stopped = true;
    double elapsed_ms = 0.0;
    std::vector<WorkerWaitResult> workers;
};

/**
 * @brief Non-owning registry for a Pipeline worker group.
 *
 * Workers are registered in startup order. Stop and Wait use reverse order so
 * upstream producers are stopped before downstream consumers are reclaimed.
 */
class WorkerRegistry {
public:
    void Clear();
    void Add(const std::string& name, ThreadBase* worker);

    size_t Size() const;

    /// Start in registration order. Already-started workers are rolled back on failure.
    bool StartAll();

    /// Send a cooperative stop request in reverse registration order.
    void RequestStopAll();

    /// Wait for all workers using one shared timeout budget.
    WorkerShutdownReport WaitAllFor(int timeout_ms);

    /// Final destruction path: wait without a timeout until every worker is joined.
    WorkerShutdownReport WaitAll();

private:
    struct Entry {
        std::string name;
        ThreadBase* worker = nullptr;
    };

    WorkerShutdownReport WaitAllUntil(
        const std::chrono::steady_clock::time_point* deadline);

    std::vector<Entry> workers_;
};

}  // namespace core
}  // namespace digital_human