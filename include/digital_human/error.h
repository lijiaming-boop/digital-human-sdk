#pragma once

/// @file error.h
/// @brief 统一错误结构（P1-3.8c）
///
/// 替代原有分散的三套错误体系：
/// - SDK 层 SDKError 枚举
/// - 会话层 bool + string
/// - Pipeline 层 bool + stderr
///
/// 统一为 Error{category, module, message, cause} 结构，
/// 兼容映射现有 SDKError。

#include <string>
#include "digital_human/export.h"

namespace digital_human {

/// 统一错误分类
enum class ErrorCategory {
    OK = 0,
    Config,
    Lifecycle,
    Model,
    Audio,
    Video,
    Sync,
    Dialog,
    TTS,
    Network,
    Media,
    Timeout,
    Cancelled,
    Unknown,
};

/// 统一错误结构
struct DH_API Error {
    ErrorCategory category = ErrorCategory::OK;
    std::string   module;    ///< "pipeline.audio_processor" / "dialog.session" / ...
    std::string   message;   ///< 可读消息
    std::string   cause;     ///< 底层原因（可为空）

    bool ok() const { return category == ErrorCategory::OK; }

    std::string to_string() const {
        std::string result = "[" + module + "] ";
        result += message;
        if (!cause.empty()) {
            result += " (cause: " + cause + ")";
        }
        return result;
    }

    static Error OK() { return {}; }

    static Error Make(ErrorCategory cat, std::string mod,
                      std::string msg, std::string cse = {}) {
        return {cat, std::move(mod), std::move(msg), std::move(cse)};
    }
};

/// 错误分类转字符串
inline const char* category_name(ErrorCategory cat) {
    switch (cat) {
        case ErrorCategory::OK:        return "ok";
        case ErrorCategory::Config:    return "config";
        case ErrorCategory::Lifecycle: return "lifecycle";
        case ErrorCategory::Model:     return "model";
        case ErrorCategory::Audio:     return "audio";
        case ErrorCategory::Video:     return "video";
        case ErrorCategory::Sync:      return "sync";
        case ErrorCategory::Dialog:    return "dialog";
        case ErrorCategory::TTS:       return "tts";
        case ErrorCategory::Network:   return "network";
        case ErrorCategory::Media:     return "media";
        case ErrorCategory::Timeout:   return "timeout";
        case ErrorCategory::Cancelled: return "cancelled";
        case ErrorCategory::Unknown:   return "unknown";
    }
    return "unknown";
}

}  // namespace digital_human
