#pragma once

#include <functional>
#include <string>
#include <vector>

#include "digital_human/export.h"

namespace digital_human {
namespace dialog {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct GenerateRequest {
    std::string session_id;
    std::string system_prompt;
    std::string user_text;
    std::vector<ChatMessage> history;
};

using TextDeltaCallback = std::function<void(const std::string&)>;
using CancelCheck = std::function<bool()>;

/// 抽象文本生成接口：会话层只依赖此接口，不传递任何 HTTP/libcurl 依赖。
/// 具体适配器（HttpTextGenerationClient / LlamaCppTextGenerationClient）
/// 声明在各自独立头文件中，由需要网络能力的调用方按需引入。
class DH_API ITextGenerationClient {
public:
    virtual ~ITextGenerationClient() = default;

    /// Blocks until the response completes, fails, or is cancelled. Implementations
    /// may call on_delta once for a complete response or repeatedly for a stream.
    virtual bool Generate(const GenerateRequest& request,
                          const TextDeltaCallback& on_delta,
                          const CancelCheck& cancelled,
                          std::string& error) = 0;
};

enum class TextResponseMode {
    AUTO,
    JSON,
    SSE,
};

}  // namespace dialog
}  // namespace digital_human
