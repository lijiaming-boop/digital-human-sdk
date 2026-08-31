#pragma once

/// @file log_context.h
/// @brief 线程局部日志上下文（P1-3.9a）
///
/// RAII 上下文绑定：进入 turn 时创建 LogContext，退出时销毁。
/// 所有模块通过 LogContext::current() 自动获取 session_id/turn_id，
/// 使日志记录无需手动传递上下文。

#include <cstdint>
#include <string>
#include "digital_human/export.h"

namespace digital_human {

/// 线程局部日志上下文
class DH_API LogContext {
public:
    LogContext(std::string session_id, uint64_t turn_id);
    ~LogContext();

    LogContext(const LogContext&) = delete;
    LogContext& operator=(const LogContext&) = delete;
    LogContext(LogContext&&) = delete;
    LogContext& operator=(LogContext&&) = delete;

    const std::string& session_id() const { return session_id_; }
    uint64_t turn_id() const { return turn_id_; }

    /// 获取当前线程的上下文（无上下文时返回 nullptr）
    static const LogContext* current();

private:
    std::string session_id_;
    uint64_t    turn_id_ = 0;
    const LogContext* prev_ = nullptr;
};

}  // namespace digital_human
