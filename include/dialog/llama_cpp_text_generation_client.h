#pragma once

#include <string>
#include <vector>

#include "dialog/text_generation_client.h"
#include "network/http_client.h"

namespace digital_human {
namespace dialog {

/// Configuration for llama.cpp's OpenAI-compatible chat completions endpoint.
struct LlamaCppTextGenerationConfig {
    std::string endpoint =
        "http://127.0.0.1:8090/v1/chat/completions";
    std::string model = "Qwen3-4B-Q4_K_M.gguf";
    std::string api_key;
    std::vector<std::string> headers;

    float temperature = 0.7F;
    float top_p = 0.9F;
    int max_tokens = 256;
    bool stream = true;
    bool enable_thinking = false;
    bool cache_prompt = true;

    int connect_timeout_ms = 2000;
    int request_timeout_ms = 120000;
};

/// Adapts llama.cpp server's /v1/chat/completions API to the model-independent
/// ITextGenerationClient consumed by ConversationSession.
class LlamaCppTextGenerationClient final : public ITextGenerationClient {
public:
    explicit LlamaCppTextGenerationClient(
        LlamaCppTextGenerationConfig config = {});

    bool Generate(const GenerateRequest& request,
                  const TextDeltaCallback& on_delta,
                  const CancelCheck& cancelled,
                  std::string& error) override;

private:
    LlamaCppTextGenerationConfig config_;
    network::HttpClient http_;
};

}  // namespace dialog
}  // namespace digital_human
