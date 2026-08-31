#pragma once

/// @file logger.h
/// @brief 统一日志接口（P1-3.9a）
///
/// SDK 不直接决定日志输出位置，通过 ILogger 接口由调用方注入。
/// 日志记录携带 session_id、turn_id、模块名和级别。
/// 替代原有 154 处 std::cout/std::cerr 直出。

#include <cstdint>
#include <memory>
#include <string>
#include "digital_human/export.h"

namespace digital_human {

/// 日志级别
enum class LogLevel {
    Debug,
    Info,
    Warn,
    Error,
};

/// 日志记录（单条）
struct LogRecord {
    LogLevel    level = LogLevel::Info;
    std::string module;       ///< "pipeline.audio_processor"
    std::string session_id;   ///< 可为空（非会话上下文）
    uint64_t    turn_id = 0;  ///< 0 表示非 turn 上下文
    std::string message;
    int64_t     timestamp_ms = 0;  ///< epoch ms，0 表示由 logger 填充
};

/// 抽象 logger 接口：SDK 不直接决定输出位置
class DH_API ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(const LogRecord& record) = 0;
    /// 返回当前允许的最低级别，低于此级别的日志将被丢弃
    virtual LogLevel min_level() const { return LogLevel::Info; }
};

/// 默认 logger：输出到 stderr，用于无注入时的回退
DH_API std::shared_ptr<ILogger> create_default_logger(LogLevel min = LogLevel::Info);

/// 全局 logger 注入（线程安全）
/// 传入 nullptr 重置为默认 logger
DH_API void set_global_logger(std::shared_ptr<ILogger> logger);

/// 获取当前全局 logger（永不为 nullptr）
DH_API std::shared_ptr<ILogger> get_global_logger();

/// 直接记录日志（内部使用，外部应使用 log_macros.h 中的宏）
DH_API void log_message(LogLevel level, const char* module,
                        const std::string& message);

}  // namespace digital_human
