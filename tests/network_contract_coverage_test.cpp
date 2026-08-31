#include "test_support.h"

#include <cstdint>
#include <string>

#include "dialog/http_text_generation_client.h"
#include "network/http_client.h"
#include "network/json_utils.h"
#include "tts/http_tts_client.h"

namespace {

void TestJsonHelpers(TestSuite& test) {
    using namespace digital_human::network::json;

    test.Equal(Escape("a\"b\\c\n\t"), std::string("a\\\"b\\\\c\\n\\t"),
               "JSON escape handles quotes, slashes and whitespace controls");
    test.Equal(Escape(std::string(1, '\x01')), std::string("\\u0001"),
               "JSON escape encodes other control characters");

    std::string value = "unchanged";
    test.Check(ExtractString(R"({"text":"line\nquote:\""})", "text", value),
               "JSON string field is extracted");
    test.Equal(value, std::string("line\nquote:\""),
               "JSON string escapes are decoded");
    test.Check(ExtractString(R"({"emoji":"\ud83d\ude00"})", "emoji", value),
               "JSON surrogate pair is extracted");
    test.Equal(value, std::string(u8"😀"),
               "JSON surrogate pair is converted to UTF-8");
    test.Check(!ExtractString(R"({"text":12})", "text", value),
               "non-string JSON value is rejected");
    test.Check(!ExtractString(R"({"text":"bad\q"})", "text", value),
               "invalid JSON escape is rejected");
    test.Check(!ExtractString("{}", "missing", value),
               "missing JSON field is rejected");

    bool boolean = false;
    test.Check(ExtractBool(R"({"done":true})", "done", boolean) && boolean,
               "true JSON field is extracted");
    test.Check(ExtractBool(R"({"done":false})", "done", boolean) && !boolean,
               "false JSON field is extracted");
    test.Check(!ExtractBool(R"({"done":null})", "done", boolean),
               "non-boolean JSON field is rejected");
}

void TestHttpBoundary(TestSuite& test) {
    using namespace digital_human::network;

    test.Check(HttpClient::IsAvailable(),
               "HTTP transport is available in network-enabled build");
    HttpClient client;
    HttpRequest request;
    HttpResponseInfo response;
    std::string error;
    test.Check(!client.Post(request, [](const uint8_t*, size_t) { return true; },
                            {}, response, error),
               "HTTP client rejects an empty URL without network access");
    test.Check(!error.empty(), "HTTP client reports empty URL failure");
    request.url = "http://127.0.0.1:1/";
    test.Check(!client.Post(request, {}, {}, response, error),
               "HTTP client rejects an empty data callback");
}

void TestServiceAdapters(TestSuite& test) {
    using namespace digital_human;

    dialog::HttpTextGenerationConfig text_config;
    text_config.endpoint.clear();
    text_config.connect_timeout_ms = 1;
    text_config.request_timeout_ms = 1;
    dialog::HttpTextGenerationClient text_client(text_config);
    dialog::GenerateRequest generate;
    generate.session_id = "session";
    generate.user_text = "hello";
    std::string error;
    test.Check(!text_client.Generate(generate, {}, {}, error),
               "text adapter rejects an empty callback");
    test.Equal(error, std::string("text delta callback is empty"),
               "text adapter returns a precise callback error");
    test.Check(!text_client.Generate(generate, [](const std::string&) {},
                                     {}, error),
               "text adapter rejects an empty endpoint offline");
    test.Check(!error.empty(), "text adapter propagates transport error");

    tts::HttpTTSConfig tts_config;
    tts_config.endpoint.clear();
    tts_config.connect_timeout_ms = 1;
    tts_config.request_timeout_ms = 1;
    tts::HttpTTSClient tts_client(tts_config);
    test.Check(!tts_client.Synthesize("", [](tts::PCMChunk) { return true; },
                                     {}, error),
               "TTS adapter rejects empty text");
    test.Check(!tts_client.Synthesize("hello", {}, {}, error),
               "TTS adapter rejects an empty callback");

    tts_config.sample_rate = 0;
    tts::HttpTTSClient invalid_tts(tts_config);
    test.Check(!invalid_tts.Synthesize("hello",
                                      [](tts::PCMChunk) { return true; },
                                      {}, error),
               "TTS adapter rejects invalid audio configuration");
    test.Equal(error, std::string("invalid TTS audio configuration"),
               "TTS adapter returns a precise configuration error");
    test.Check(!tts_client.Synthesize("hello",
                                     [](tts::PCMChunk) { return true; },
                                     {}, error),
               "TTS adapter rejects an empty endpoint offline");
    test.Check(!error.empty(), "TTS adapter propagates transport error");
}

}  // namespace

int main() {
    TestSuite test;
    TestJsonHelpers(test);
    TestHttpBoundary(test);
    TestServiceAdapters(test);
    return test.Finish("network_contract_coverage_test");
}
