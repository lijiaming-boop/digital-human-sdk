#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "dialog/text_generation_client.h"
#include "network/http_client.h"
#include "tts/tts_client.h"

using namespace digital_human;

int main(int argc, char** argv) {
    if (!network::HttpClient::IsAvailable()) {
        std::cout << "[SKIP] libcurl HTTP client is not available\n";
        return 0;
    }
    if (argc != 3) {
        std::cerr << "Usage: http_service_client_test <text_url> <tts_url>\n";
        return 2;
    }

    dialog::HttpTextGenerationConfig text_config;
    text_config.endpoint = argv[1];
    text_config.response_mode = dialog::TextResponseMode::AUTO;
    dialog::HttpTextGenerationClient text_client(text_config);

    dialog::GenerateRequest request;
    request.session_id = "http-test";
    request.user_text = "测试网络文本服务";
    std::string reply;
    std::string error;
    if (!text_client.Generate(
            request,
            [&](const std::string& delta) { reply += delta; },
            []() { return false; }, error)) {
        std::cerr << "[FAIL] text service: " << error << '\n';
        return 1;
    }
    if (reply != "网络服务正常。") {
        std::cerr << "[FAIL] unexpected reply: " << reply << '\n';
        return 1;
    }

    tts::HttpTTSConfig tts_config;
    tts_config.endpoint = argv[2];
    tts_config.response_format = tts::TTSAudioFormat::PCM_S16LE;
    tts::HttpTTSClient tts_client(tts_config);
    size_t sample_count = 0;
    if (!tts_client.Synthesize(
            reply,
            [&](tts::PCMChunk chunk) {
                sample_count += chunk.samples.size();
                return true;
            },
            []() { return false; }, error)) {
        std::cerr << "[FAIL] TTS service: " << error << '\n';
        return 1;
    }
    if (sample_count != 3200) {
        std::cerr << "[FAIL] unexpected PCM sample count: " << sample_count
                  << '\n';
        return 1;
    }
    std::cout << "[PASS] HTTP text and TTS service clients\n";
    return 0;
}
