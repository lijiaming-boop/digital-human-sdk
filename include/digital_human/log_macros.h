#pragma once

/// @file log_macros.h
/// @brief 日志宏（P1-3.9a）
///
/// 便捷日志宏，自动填充模块名和线程局部上下文（session_id/turn_id）。
/// 用法：
///   DH_LOG_INFO("pipeline") << "初始化成功: " << model_path;
///   DH_LOG_ERROR("dialog.session") << "text generation failed: " << error;

#include <sstream>
#include <string>
#include "digital_human/logger.h"
#include "digital_human/log_context.h"

namespace digital_human {
namespace detail {

class LogEmitter {
public:
    LogEmitter(LogLevel level, const char* module)
        : level_(level), module_(module) {}

    ~LogEmitter() {
        log_message(level_, module_, buffer_.str());
    }

    template <typename T>
    LogEmitter& operator<<(const T& value) {
        buffer_ << value;
        return *this;
    }

private:
    LogLevel    level_;
    const char* module_;
    std::ostringstream buffer_;
};

}  // namespace detail
}  // namespace digital_human

#define DH_LOG(level, module) \
    ::digital_human::detail::LogEmitter(level, module)

#define DH_LOG_DEBUG(module) DH_LOG(::digital_human::LogLevel::Debug, module)
#define DH_LOG_INFO(module)  DH_LOG(::digital_human::LogLevel::Info,  module)
#define DH_LOG_WARN(module)  DH_LOG(::digital_human::LogLevel::Warn,  module)
#define DH_LOG_ERROR(module) DH_LOG(::digital_human::LogLevel::Error, module)
