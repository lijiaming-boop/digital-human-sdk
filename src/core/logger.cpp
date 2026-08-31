#include "digital_human/logger.h"
#include "digital_human/log_context.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <utility>

namespace digital_human {

namespace {

const char* level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "?";
}

const char* level_color(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "\033[37m";  // gray
        case LogLevel::Info:  return "\033[32m";  // green
        case LogLevel::Warn:  return "\033[33m";  // yellow
        case LogLevel::Error: return "\033[31m";  // red
    }
    return "";
}

class DefaultLogger : public ILogger {
public:
    explicit DefaultLogger(LogLevel min) : min_(min) {}

    LogLevel min_level() const override { return min_; }

    void log(const LogRecord& record) override {
        if (record.level < min_) return;
        int64_t ts = record.timestamp_ms;
        if (ts == 0) {
            ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        }
        std::time_t secs = static_cast<std::time_t>(ts / 1000);
        int millis = static_cast<int>(ts % 1000);
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &secs);
#else
        localtime_r(&secs, &tm_buf);
#endif
        char time_str[32];
        std::snprintf(time_str, sizeof(time_str), "%02d:%02d:%02d.%03d",
                      tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, millis);

        std::string header = std::string(level_color(record.level))
                           + "[" + time_str + "]"
                           + "[" + level_name(record.level) + "]"
                           + "[" + record.module + "]";
        if (!record.session_id.empty()) {
            header += "[s:" + record.session_id + "]";
        }
        if (record.turn_id != 0) {
            header += "[t:" + std::to_string(record.turn_id) + "]";
        }
        header += " ";

        std::lock_guard<std::mutex> lock(mutex_);
        std::fputs(header.c_str(), stderr);
        std::fputs(record.message.c_str(), stderr);
        std::fputs("\033[0m\n", stderr);
    }

private:
    LogLevel min_;
    std::mutex mutex_;
};

std::shared_ptr<ILogger>& global_logger_storage() {
    static std::shared_ptr<ILogger> logger = std::make_shared<DefaultLogger>(LogLevel::Info);
    return logger;
}

std::mutex& global_logger_mutex() {
    static std::mutex mtx;
    return mtx;
}

}  // namespace

std::shared_ptr<ILogger> create_default_logger(LogLevel min) {
    return std::make_shared<DefaultLogger>(min);
}

void set_global_logger(std::shared_ptr<ILogger> logger) {
    std::lock_guard<std::mutex> lock(global_logger_mutex());
    if (logger) {
        global_logger_storage() = std::move(logger);
    } else {
        global_logger_storage() = std::make_shared<DefaultLogger>(LogLevel::Info);
    }
}

std::shared_ptr<ILogger> get_global_logger() {
    std::lock_guard<std::mutex> lock(global_logger_mutex());
    return global_logger_storage();
}

void log_message(LogLevel level, const char* module, const std::string& message) {
    auto logger = get_global_logger();
    if (level < logger->min_level()) return;

    LogRecord record;
    record.level = level;
    record.module = module;
    record.message = message;

    // 自动从线程局部上下文填充 session_id 和 turn_id
    if (const LogContext* ctx = LogContext::current()) {
        record.session_id = ctx->session_id();
        record.turn_id = ctx->turn_id();
    }

    logger->log(record);
}

}  // namespace digital_human
