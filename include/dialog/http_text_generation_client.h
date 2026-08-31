#pragma once

#include <string>
#include <vector>

#include "dialog/text_generation_client.h"
#include "network/http_client.h"

namespace digital_human {
namespace dialog {

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

/// HTTP 适配器：将通用 HTTP 文本生成端点适配为 ITextGenerationClient。
/// 此类从 text_generation_client.h 拆分出来，使抽象接口不再传递 libcurl 依赖。
class DH_API HttpTextGenerationClient final : public ITextGenerationClient {
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
