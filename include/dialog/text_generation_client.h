#pragma once

#include <functional>
#include <string>
#include <vector>

#include "network/http_client.h"

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

class ITextGenerationClient {
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

/// Generic HTTP contract:
/// request: {session_id, system_prompt, user_text, history, stream}
/// JSON response: {"reply":"..."}
/// SSE event: data: {"delta":"..."}; final event may contain {"done":true}.
struct HttpTextGenerationConfig {
    std::string endpoint;
    std::string api_key;
    std::vector<std::string> headers;
    TextResponseMode response_mode = TextResponseMode::AUTO;
    int connect_timeout_ms = 2000;
    int request_timeout_ms = 30000;
};

class HttpTextGenerationClient final : public ITextGenerationClient {
public:
    explicit HttpTextGenerationClient(HttpTextGenerationConfig config);

    bool Generate(const GenerateRequest& request,
                  const TextDeltaCallback& on_delta,
                  const CancelCheck& cancelled,
                  std::string& error) override;

private:
    HttpTextGenerationConfig config_;
    network::HttpClient http_;
};

}  // namespace dialog
}  // namespace digital_human
