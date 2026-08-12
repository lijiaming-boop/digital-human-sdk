#include "dialog/llama_cpp_text_generation_client.h"

#include <cctype>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <utility>

#include "network/json_utils.h"

namespace digital_human {
namespace dialog {
namespace {

std::string Trim(std::string value) {
    while (!value.empty()
           && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty()
           && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

void AppendMessage(std::string& body,
                   bool& first,
                   const std::string& role,
                   const std::string& content) {
    if (!first) body.push_back(',');
    first = false;
    body += "{\"role\":\"" + network::json::Escape(role)
         + "\",\"content\":\"" + network::json::Escape(content)
         + "\"}";
}

std::string Number(float value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(6) << value;
    return stream.str();
}

std::string BuildRequestBody(const LlamaCppTextGenerationConfig& config,
                             const GenerateRequest& request) {
    std::string body = "{\"model\":\""
        + network::json::Escape(config.model) + "\",\"messages\":[";
    bool first = true;
    if (!request.system_prompt.empty()) {
        AppendMessage(body, first, "system", request.system_prompt);
    }
    for (const auto& message : request.history) {
        AppendMessage(body, first, message.role, message.content);
    }
    AppendMessage(body, first, "user", request.user_text);
    body += "],\"stream\":";
    body += config.stream ? "true" : "false";
    body += ",\"temperature\":" + Number(config.temperature);
    body += ",\"top_p\":" + Number(config.top_p);
    body += ",\"max_tokens\":" + std::to_string(config.max_tokens);
    body += ",\"cache_prompt\":";
    body += config.cache_prompt ? "true" : "false";
    body += ",\"chat_template_kwargs\":{\"enable_thinking\":";
    body += config.enable_thinking ? "true" : "false";
    body += "}}";
    return body;
}

void AddHeaders(const LlamaCppTextGenerationConfig& config,
                std::vector<std::string>& headers) {
    headers = config.headers;
    headers.emplace_back("Content-Type: application/json");
    headers.emplace_back(config.stream
        ? "Accept: text/event-stream" : "Accept: application/json");
    if (!config.api_key.empty()) {
        headers.emplace_back("Authorization: Bearer " + config.api_key);
    }
}

}  // namespace

LlamaCppTextGenerationClient::LlamaCppTextGenerationClient(
    LlamaCppTextGenerationConfig config)
    : config_(std::move(config)) {}

bool LlamaCppTextGenerationClient::Generate(
    const GenerateRequest& request,
    const TextDeltaCallback& on_delta,
    const CancelCheck& cancelled,
    std::string& error) {
    error.clear();
    if (!on_delta) {
        error = "text delta callback is empty";
        return false;
    }
    if (config_.endpoint.empty() || config_.model.empty()
        || config_.max_tokens <= 0 || !std::isfinite(config_.temperature)
        || config_.temperature < 0.0F || !std::isfinite(config_.top_p)
        || config_.top_p <= 0.0F || config_.top_p > 1.0F) {
        error = "invalid llama.cpp text generation configuration";
        return false;
    }
    if (request.user_text.empty()) {
        error = "llama.cpp user message is empty";
        return false;
    }

    network::HttpRequest http_request;
    http_request.url = config_.endpoint;
    http_request.body = BuildRequestBody(config_, request);
    http_request.connect_timeout_ms = config_.connect_timeout_ms;
    http_request.request_timeout_ms = config_.request_timeout_ms;
    AddHeaders(config_, http_request.headers);

    std::string received;
    std::string line_buffer;
    bool saw_sse = false;
    bool saw_content = false;
    auto consume_sse_line = [&](std::string line) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("data:", 0) != 0) return;
        saw_sse = true;
        const std::string payload = Trim(line.substr(5));
        if (payload.empty() || payload == "[DONE]") return;
        std::string delta;
        if (network::json::ExtractString(payload, "content", delta)
            && !delta.empty()) {
            saw_content = true;
            on_delta(delta);
        }
    };

    auto on_data = [&](const uint8_t* data, size_t size) {
        if (cancelled && cancelled()) return false;
        received.append(reinterpret_cast<const char*>(data), size);
        if (!config_.stream) return true;
        line_buffer.append(reinterpret_cast<const char*>(data), size);
        size_t newline = 0;
        while ((newline = line_buffer.find('\n')) != std::string::npos) {
            consume_sse_line(line_buffer.substr(0, newline));
            line_buffer.erase(0, newline + 1);
        }
        return true;
    };

    network::HttpResponseInfo response;
    if (!http_.Post(http_request, on_data, cancelled, response, error)) {
        std::string message;
        if (network::json::ExtractString(received, "message", message)
            && !message.empty()) {
            error += ": " + message;
        }
        return false;
    }

    if (config_.stream) {
        if (!line_buffer.empty()) consume_sse_line(line_buffer);
        if (saw_sse && saw_content) return true;
        error = saw_sse
            ? "llama.cpp stream completed without assistant content"
            : "llama.cpp response is not an SSE stream";
        return false;
    }

    std::string reply;
    if (network::json::ExtractString(received, "content", reply)
        && !reply.empty()) {
        on_delta(reply);
        return true;
    }
    error = "llama.cpp response has no assistant content";
    return false;
}

}  // namespace dialog
}  // namespace digital_human
