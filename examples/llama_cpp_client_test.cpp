#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "dialog/conversation_session.h"
#include "dialog/llama_cpp_text_generation_client.h"
#include "network/http_client.h"
#include "tts/tts_client.h"

using namespace digital_human;

namespace {

class TestTTSClient final : public tts::ITTSClient {
public:
    bool Synthesize(const std::string&,
                    const tts::PCMCallback& on_audio,
                    const tts::CancelCheck& cancelled,
                    std::string&) override {
        if (cancelled && cancelled()) return false;
        tts::PCMChunk chunk;
        chunk.sample_rate = 16000;
        chunk.channels = 1;
        chunk.samples.assign(3200, 0.01F);
        ++calls;
        return on_audio(std::move(chunk));
    }

    std::atomic<int> calls{0};
};

class CountingSink final : public dialog::IDigitalHumanSink {
public:
    bool PushAudio(const std::vector<float>& samples,
                   int64_t,
                   std::string&) override {
        audio_samples.fetch_add(samples.size());
        return true;
    }

    bool PushVideo(const cv::Mat&, int64_t, std::string&) override {
        video_frames.fetch_add(1);
        return true;
    }

    void Finish() override { finished.store(true); }

    std::atomic<size_t> audio_samples{0};
    std::atomic<int> video_frames{0};
    std::atomic<bool> finished{false};
};

}  // namespace

int main(int argc, char** argv) {
    if (!network::HttpClient::IsAvailable()) {
        std::cout << "[SKIP] libcurl HTTP client is not available\n";
        return 0;
    }

    dialog::LlamaCppTextGenerationConfig config;
    if (argc > 1) config.endpoint = argv[1];
    if (argc > 2) config.model = argv[2];
    config.temperature = 0.0F;
    config.max_tokens = 32;
    config.stream = true;
    config.enable_thinking = false;

    dialog::GenerateRequest request;
    request.session_id = "llama-cpp-test";
    request.system_prompt =
        u8"\u4e25\u683c\u9075\u5b88\u7528\u6237\u7684\u8f93\u51fa\u683c\u5f0f\uff0c\u4e0d\u8981\u89e3\u91ca\uff0c\u4e0d\u8981\u6dfb\u52a0\u5176\u4ed6\u6587\u5b57\u3002";
    request.history = {
        {"user", u8"\u53ea\u56de\u590d\uff1aHISTORY_OK"},
        {"assistant", "HISTORY_OK"},
    };
    request.user_text = u8"\u53ea\u56de\u590d\uff1aLLAMA_CPP_OK";

    std::string error;
    for (const bool stream : {true, false}) {
        config.stream = stream;
        dialog::LlamaCppTextGenerationClient client(config);
        std::string reply;
        if (!client.Generate(
                request,
                [&](const std::string& delta) { reply += delta; },
                []() { return false; }, error)) {
            std::cerr << "[FAIL] llama.cpp client: " << error << '\n';
            return 1;
        }
        if (reply.find("LLAMA_CPP_OK") == std::string::npos) {
            std::cerr << "[FAIL] unexpected llama.cpp reply: " << reply << '\n';
            return 1;
        }
        std::cout << "[PASS] llama.cpp OpenAI-compatible "
                  << (stream ? "SSE: " : "JSON: ") << reply << '\n';
    }

    config.stream = true;
    dialog::LlamaCppTextGenerationClient conversation_client(config);
    TestTTSClient tts_client;
    CountingSink sink;
    dialog::ConversationSession session(
        conversation_client, tts_client, sink);
    dialog::ConversationConfig conversation_config;
    conversation_config.session_id = "llama-cpp-conversation-test";
    conversation_config.system_prompt = request.system_prompt;
    conversation_config.min_tts_clause_chars = 1;
    const cv::Mat avatar(64, 64, CV_8UC3, cv::Scalar(20, 40, 80));
    if (!session.Start(conversation_config, avatar)
        || session.SubmitUserText(request.user_text) == 0
        || !session.WaitUntilIdle(std::chrono::seconds(30))) {
        std::cerr << "[FAIL] llama.cpp ConversationSession did not finish\n";
        session.Stop(false);
        return 1;
    }
    session.Stop(true);
    if (tts_client.calls.load() == 0 || sink.audio_samples.load() == 0
        || sink.video_frames.load() == 0 || !sink.finished.load()) {
        std::cerr << "[FAIL] llama.cpp conversation media feed is incomplete\n";
        return 1;
    }
    std::cout << "[PASS] llama.cpp -> ConversationSession -> TTS/media sink"
              << " audio=" << sink.audio_samples.load()
              << " video=" << sink.video_frames.load() << '\n';
    return 0;
}
