#include "network/http_client.h"

#include <mutex>
#include <utility>

#ifdef DIGITAL_HUMAN_HAS_CURL
#include <curl/curl.h>
#endif

namespace digital_human {
namespace network {

#ifdef DIGITAL_HUMAN_HAS_CURL
namespace {

struct TransferContext {
    const DataCallback* on_data = nullptr;
    const CancelCheck* cancelled = nullptr;
    bool callback_failed = false;
};

size_t WriteData(char* data, size_t size, size_t count, void* user_data) {
    auto* ctx = static_cast<TransferContext*>(user_data);
    const size_t bytes = size * count;
    if (ctx->cancelled && *ctx->cancelled && (*ctx->cancelled)()) {
        return 0;
    }
    if (ctx->on_data && *ctx->on_data
        && !(*ctx->on_data)(reinterpret_cast<const uint8_t*>(data), bytes)) {
        ctx->callback_failed = true;
        return 0;
    }
    return bytes;
}

int TransferProgress(void* user_data,
                     curl_off_t,
                     curl_off_t,
                     curl_off_t,
                     curl_off_t) {
    auto* ctx = static_cast<TransferContext*>(user_data);
    return (ctx->cancelled && *ctx->cancelled && (*ctx->cancelled)()) ? 1 : 0;
}

void EnsureCurlInitialized() {
    static std::once_flag flag;
    std::call_once(flag, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

}  // namespace
#endif

bool HttpClient::IsAvailable() {
#ifdef DIGITAL_HUMAN_HAS_CURL
    return true;
#else
    return false;
#endif
}

bool HttpClient::Post(const HttpRequest& request,
                      const DataCallback& on_data,
                      const CancelCheck& cancelled,
                      HttpResponseInfo& response,
                      std::string& error) const {
#ifndef DIGITAL_HUMAN_HAS_CURL
    (void)request;
    (void)on_data;
    (void)cancelled;
    (void)response;
    error = "HTTP client is unavailable: rebuild with libcurl development files";
    return false;
#else
    if (request.url.empty()) {
        error = "HTTP endpoint is empty";
        return false;
    }
    EnsureCurlInitialized();
    CURL* curl = curl_easy_init();
    if (!curl) {
        error = "curl_easy_init failed";
        return false;
    }

    curl_slist* headers = nullptr;
    for (const auto& header : request.headers) {
        headers = curl_slist_append(headers, header.c_str());
    }
    TransferContext ctx{&on_data, &cancelled, false};
    char curl_error[CURL_ERROR_SIZE] = {};

    curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.body.data());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE,
                     static_cast<curl_off_t>(request.body.size()));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteData);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, TransferProgress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(request.connect_timeout_ms));
    if (request.request_timeout_ms > 0) {
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS,
                         static_cast<long>(request.request_timeout_ms));
    }
    // P0 HTTP 资源限制：响应体上限、低速超时、重定向策略。
    if (request.max_response_bytes > 0) {
        curl_easy_setopt(curl, CURLOPT_MAXFILESIZE_LARGE,
                         static_cast<curl_off_t>(request.max_response_bytes));
    }
    if (request.low_speed_limit_bytes_per_s > 0
        && request.low_speed_time_ms > 0) {
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT,
                         request.low_speed_limit_bytes_per_s);
        // libcurl 低速窗口以秒为单位，向上取整避免 0。
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME,
                         static_cast<long>(
                             (request.low_speed_time_ms + 999) / 1000));
    }
    // 默认不跟随重定向，收紧 SSRF 边界；仅当显式开启时按 max_redirects 限制。
    if (request.follow_redirects && request.max_redirects > 0) {
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS,
                         static_cast<long>(request.max_redirects));
    } else {
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);
    }
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_error);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode result = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status_code);
    char* content_type = nullptr;
    curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &content_type);
    if (content_type) response.content_type = content_type;

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (result != CURLE_OK) {
        if (cancelled && cancelled()) {
            error = "HTTP request cancelled";
        } else if (ctx.callback_failed) {
            error = "HTTP response callback rejected data";
        } else if (curl_error[0] != '\0') {
            error = curl_error;
        } else {
            error = curl_easy_strerror(result);
        }
        return false;
    }
    if (response.status_code < 200 || response.status_code >= 300) {
        error = "HTTP status " + std::to_string(response.status_code);
        return false;
    }
    return true;
#endif
}

}  // namespace network
}  // namespace digital_human
