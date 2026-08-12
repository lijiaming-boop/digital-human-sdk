#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace digital_human {
namespace network {

struct HttpRequest {
    std::string url;
    std::vector<std::string> headers;
    std::string body;
    int connect_timeout_ms = 2000;
    int request_timeout_ms = 30000;
    /// 资源限制（P0 HTTP 边界）：
    /// max_response_bytes > 0 时由 libcurl 强制响应体上限，超过即中止传输。
    int64_t max_response_bytes = 0;  // 0: 不限制
    /// 低速超时：在 low_speed_time_ms 内平均速率低于 low_speed_limit_bytes_per_s
    /// 即视为卡死并中止，避免慢消费者长时间占用连接。
    long low_speed_limit_bytes_per_s = 0;   // 0: 不启用
    int low_speed_time_ms = 0;              // 0: 不启用
    /// 重定向策略：follow_redirects=true 时跟随，最多 max_redirects 次（默认拒绝
    /// 重定向以收紧 SSRF 边界，调用方需显式开启）。
    bool follow_redirects = false;
    int max_redirects = 0;
};

struct HttpResponseInfo {
    long status_code = 0;
    std::string content_type;
};

using DataCallback =
    std::function<bool(const uint8_t* data, size_t size)>;
using CancelCheck = std::function<bool()>;

/// Thin synchronous HTTP transport. The implementation uses libcurl when the
/// optional HTTP dependency is available; no curl types leak into the ABI.
class HttpClient {
public:
    static bool IsAvailable();

    bool Post(const HttpRequest& request,
              const DataCallback& on_data,
              const CancelCheck& cancelled,
              HttpResponseInfo& response,
              std::string& error) const;
};

}  // namespace network
}  // namespace digital_human
