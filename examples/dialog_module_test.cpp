#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "dialog/conversation_session.h"
#include "dialog/sentence_segmenter.h"

using namespace digital_human;

namespace {

class FakeTextClient final : public dialog::ITextGenerationClient {
public:
    bool Generate(const dialog::GenerateRequest& request,
                  const dialog::TextDeltaCallback& on_delta,
                  const dialog::CancelCheck& cancelled,
                  std::string& error) override {
        if (request.user_text.empty()) {
            error = "missing user text";
            return false;
        }
        const std::vector<std::string> deltas{
            "您好，", "欢迎使用数字人。", "我们开始吧！"};
        for (const auto& delta : deltas) {
            if (cancelled && cancelled()) return false;
            on_delta(delta);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return true;
    }
};

class SlowTextClient final : public dialog::ITextGenerationClient {
public:
    bool Generate(const dialog::GenerateRequest&,
                  const dialog::TextDeltaCallback& on_delta,
                  const dialog::CancelCheck& cancelled,
                  std::string&) override {
        for (int i = 0; i < 100; ++i) {
            if (cancelled && cancelled()) return false;
            on_delta("处理中");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return true;
    }
};

class FakeTTSClient final : public tts::ITTSClient {
public:
    std::atomic<int> calls{0};

    bool Synthesize(const std::string& text,
                    const tts::PCMCallback& on_audio,
                    const tts::CancelCheck& cancelled,
                    std::string& error) override {
        if (text.empty()) {
            error = "empty TTS text";
            return false;
        }
        calls.fetch_add(1);
        constexpr int sample_rate = 16000;
        constexpr int samples_per_clause = 2560;  // 160 ms
        tts::PCMChunk chunk;
        chunk.sample_rate = sample_rate;
        chunk.channels = 1;
        chunk.samples.resize(samples_per_clause);
        for (int i = 0; i < samples_per_clause; ++i) {
            if (cancelled && cancelled()) return false;
            chunk.samples[static_cast<size_t>(i)] =
                0.05f * std::sin(2.0 * 3.141592653589793 * 220.0 * i
                                 / sample_rate);
        }
        return on_audio(std::move(chunk));
    }
};

class RecordingSink final : public dialog::IDigitalHumanSink {
public:
    bool PushAudio(const std::vector<float>& samples,
                   int64_t pts_ms,
                   std::string&) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!audio_pts.empty() && pts_ms < audio_pts.back()) monotonic = false;
        audio_pts.push_back(pts_ms);
        total_audio_samples += samples.size();
        return true;
    }

    bool PushVideo(const cv::Mat& frame,
                   int64_t pts_ms,
                   std::string&) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (frame.empty()) return false;
        if (!video_pts.empty() && pts_ms <= video_pts.back()) monotonic = false;
        video_pts.push_back(pts_ms);
        return true;
    }

    void Finish() override { finished.store(true); }

    std::mutex mutex;
    std::vector<int64_t> audio_pts;
    std::vector<int64_t> video_pts;
    size_t total_audio_samples = 0;
    bool monotonic = true;
    std::atomic<bool> finished{false};
};

bool Check(bool condition, const std::string& message) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << message << '\n';
    return condition;
}

}  // namespace

int main() {
    bool ok = true;

    dialog::SentenceSegmenter segmenter({8});
    auto first = segmenter.Push("您好，");
    ok &= Check(first.empty(), "短逗号片段不会过早送入 TTS");
    auto second = segmenter.Push("欢迎使用数字人。下一句");
    ok &= Check(second.size() == 1
                    && second.front() == "您好，欢迎使用数字人。",
                "UTF-8 增量文本按强标点正确分句");
    ok &= Check(segmenter.Flush() == "下一句", "Flush 返回尾部文本");

    FakeTextClient text_client;
    FakeTTSClient tts_client;
    RecordingSink sink;
    dialog::ConversationSession session(text_client, tts_client, sink);

    dialog::ConversationConfig config;
    config.session_id = "module-test";
    config.reply_tail_silence_ms = 200;
    config.mel_lookahead_ms = 160;
    cv::Mat avatar(96, 96, CV_8UC3, cv::Scalar(20, 40, 60));

    std::string full_reply;
    std::atomic<int> completed{0};
    dialog::ConversationCallbacks callbacks;
    callbacks.on_text_delta = [&](uint64_t, const std::string& delta) {
        full_reply += delta;
    };
    callbacks.on_turn_complete = [&](uint64_t) { completed.fetch_add(1); };
    callbacks.on_error = [&](uint64_t, const std::string& error) {
        std::cerr << "[ERROR] " << error << '\n';
    };

    ok &= Check(session.Start(config, avatar, callbacks), "会话控制器启动");
    const uint64_t task_id = session.SubmitUserText("你好");
    ok &= Check(task_id != 0, "提交用户文本任务");
    ok &= Check(session.SubmitUserText("并发任务") == 0,
                "第一阶段明确拒绝同会话并发轮次");
    ok &= Check(session.WaitUntilIdle(std::chrono::seconds(5)),
                "文本→分句→TTS→音视频供料闭环完成");
    session.Stop(true);

    ok &= Check(full_reply == "您好，欢迎使用数字人。我们开始吧！",
                "完整拼接文本服务增量回复");
    ok &= Check(tts_client.calls.load() == 2, "两段回复按顺序调用 TTS");
    {
        std::lock_guard<std::mutex> lock(sink.mutex);
        ok &= Check(sink.total_audio_samples == 8320,
                    "TTS 音频与 200ms 尾静音全部提交");
        ok &= Check(!sink.video_pts.empty(), "按音频水位生成数字人视频帧");
        ok &= Check(sink.monotonic, "音频和视频 PTS 单调递增");
    }
    ok &= Check(completed.load() == 1, "会话轮次完成回调仅触发一次");
    ok &= Check(sink.finished.load(), "停止时向数字人输入发送 EOS");

    SlowTextClient slow_text;
    FakeTTSClient interrupted_tts;
    RecordingSink interrupted_sink;
    dialog::ConversationSession interrupted_session(
        slow_text, interrupted_tts, interrupted_sink);
    ok &= Check(interrupted_session.Start(config, avatar), "打断测试会话启动");
    ok &= Check(interrupted_session.SubmitUserText("取消本轮") != 0,
                "提交可取消文本任务");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    interrupted_session.Interrupt();
    ok &= Check(interrupted_session.WaitUntilIdle(std::chrono::seconds(2)),
                "打断会取消文本/TTS待处理任务并回到空闲");
    interrupted_session.Stop(false);

    std::cout << (ok ? "\nALL DIALOG MODULE TESTS PASSED\n"
                     : "\nDIALOG MODULE TEST FAILED\n");
    return ok ? 0 : 1;
}
