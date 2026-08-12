#include "dialog/text_generation_client.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include "network/json_utils.h"

namespace digital_human {
namespace dialog {
namespace {

std::string BuildRequestBody(const GenerateRequest& request, bool stream) {
    using network::json::Escape;
    std::string body = "{\"session_id\":\"" + Escape(request.session_id)
        + "\",\"system_prompt\":\"" + Escape(request.system_prompt)
        + "\",\"user_text\":\"" + Escape(request.user_text)
        + "\",\"history\":[";
    for (size_t i = 0; i < request.history.size(); ++i) {
        if (i > 0) body += ',';
        body += "{\"role\":\"" + Escape(request.history[i].role)
             + "\",\"content\":\"" + Escape(request.history[i].content)
             + "\"}";
    }
    body += "],\"stream\":";
    body += stream ? "true}" : "false}";
    return body;
}

void AddHeaders(const HttpTextGenerationConfig& config,
                std::vector<std::string>& headers) {
    headers = config.headers;
    headers.emplace_back("Content-Type: application/json");
    headers.emplace_back("Accept: application/json, text/event-stream");
    if (!config.api_key.empty()) {
        headers.emplace_back("Authorization: Bearer " + config.api_key);
    }
}

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

}  // namespace

HttpTextGenerationClient::HttpTextGenerationClient(
    HttpTextGenerationConfig config)
    : config_(std::move(config)) {}

bool HttpTextGenerationClient::Generate(
    const GenerateRequest& request,
    const TextDeltaCallback& on_delta,
    const CancelCheck& cancelled,
    std::string& error) {
    if (!on_delta) {
        error = "text delta callback is empty";
        return false;
    }
    const bool request_stream = config_.response_mode != TextResponseMode::JSON;
    network::HttpRequest http_request;
    http_request.url = config_.endpoint;
    http_request.body = BuildRequestBody(request, request_stream);
    http_request.connect_timeout_ms = config_.connect_timeout_ms;
    http_request.request_timeout_ms = config_.request_timeout_ms;
    AddHeaders(config_, http_request.headers);

    std::string received;
    std::string line_buffer;
    bool saw_sse_event = false;
    auto consume_sse_line = [&](std::string line) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("data:", 0) != 0) return;
        saw_sse_event = true;
        std::string payload = Trim(line.substr(5));
        if (payload.empty() || payload == "[DONE]") return;
        std::string delta;
        if (network::json::ExtractString(payload, "delta", delta)
            || network::json::ExtractString(payload, "reply", delta)) {
            if (!delta.empty()) on_delta(delta);
        }
    };

    auto on_data = [&](const uint8_t* data, size_t size) {
        if (cancelled && cancelled()) return false;
        received.append(reinterpret_cast<const char*>(data), size);
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
        return false;
    }
    if (!line_buffer.empty()) consume_sse_line(line_buffer);
    if (saw_sse_event) return true;

    std::string reply;
    if (network::json::ExtractString(received, "reply", reply)
        || network::json::ExtractString(received, "text", reply)) {
        if (!reply.empty()) on_delta(reply);
        return true;
    }

    reply = Trim(received);
    if (!reply.empty() && reply.front() != '{' && reply.front() != '[') {
        on_delta(reply);
        return true;
    }
    error = "text service response has no reply/delta field";
    return false;
}

}  // namespace dialog
}  // namespace digital_human
