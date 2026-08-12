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
#include "audio/streaming_vad.h"
#include "digital_human/metrics.h"

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

class FailingTextClient final : public dialog::ITextGenerationClient {
public:
    bool Generate(const dialog::GenerateRequest&,
                  const dialog::TextDeltaCallback& on_delta,
                  const dialog::CancelCheck&,
                  std::string& error) override {
        on_delta("不完整响应");
        error = "injected generation failure";
        return false;
    }
};

class UncooperativeTextClient final : public dialog::ITextGenerationClient {
public:
    bool Generate(const dialog::GenerateRequest&,
                  const dialog::TextDeltaCallback&,
                  const dialog::CancelCheck&,
                  std::string&) override {
        entered.store(true);
        while (!release.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }

    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
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

    audio::StreamingVADConfig invalid_vad_config;
    invalid_vad_config.frame_ms = 0;
    ok &= Check(!audio::create_streaming_vad(invalid_vad_config),
                "流式 VAD 拒绝零帧长配置");
    auto vad = audio::create_streaming_vad(audio::StreamingVADConfig{});
    ok &= Check(static_cast<bool>(vad), "流式 VAD 接受默认配置");
    ok &= Check(vad && vad->Start([](audio::VADEvent) {}),
                "流式 VAD 使用有效回调启动");
    ok &= Check(vad && !vad->PushAudio(nullptr, 1),
                "流式 VAD 拒绝空样本指针");
    if (vad) vad->Stop();

    PercentileTracker tracker(64);
    std::thread metric_writer_a([&]() {
        for (int i = 0; i < 1000; ++i) tracker.record(i);
    });
    std::thread metric_writer_b([&]() {
        for (int i = 0; i < 1000; ++i) tracker.record(i + 1000);
    });
    for (int i = 0; i < 100; ++i) (void)tracker.p95();
    metric_writer_a.join();
    metric_writer_b.join();
    ok &= Check(tracker.count() == 2000,
                "指标统计支持并发写入与快照读取");

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

    FailingTextClient failing_text;
    FakeTTSClient failure_tts;
    RecordingSink failure_sink;
    dialog::ConversationSession failure_session(
        failing_text, failure_tts, failure_sink);
    std::atomic<int> failure_errors{0};
    std::atomic<int> failure_completions{0};
    dialog::ConversationCallbacks failure_callbacks;
    failure_callbacks.on_error = [&](uint64_t, const std::string&) {
        failure_errors.fetch_add(1);
    };
    failure_callbacks.on_turn_complete = [&](uint64_t) {
        failure_completions.fetch_add(1);
    };
    ok &= Check(failure_session.Start(config, avatar, failure_callbacks),
                "失败终态测试会话启动");
    ok &= Check(failure_session.SubmitUserText("触发失败") != 0,
                "提交失败注入任务");
    ok &= Check(failure_session.WaitUntilIdle(std::chrono::seconds(2)),
                "生成失败后会话退出忙状态");
    ok &= Check(failure_session.State() == dialog::SessionState::FAILED,
                "生成失败保持 FAILED 终态");
    ok &= Check(failure_errors.load() == 1,
                "生成失败只触发一次 on_error");
    ok &= Check(failure_completions.load() == 0,
                "失败 turn 不触发完成回调");
    ok &= Check(failure_tts.calls.load() == 0,
                "失败的部分 LLM 响应不会进入 TTS");
    failure_session.Stop(false);

    UncooperativeTextClient uncooperative_text;
    FakeTTSClient timeout_tts;
    RecordingSink timeout_sink;
    dialog::ConversationSession timeout_session(
        uncooperative_text, timeout_tts, timeout_sink);
    ok &= Check(timeout_session.Start(config, avatar), "停止超时测试会话启动");
    ok &= Check(timeout_session.SubmitUserText("阻塞生成") != 0,
                "提交不响应取消的生成任务");
    for (int i = 0; i < 100 && !uncooperative_text.entered.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto stop_started = std::chrono::steady_clock::now();
    const auto timeout_result = timeout_session.Stop(
        false, std::chrono::milliseconds(20));
    const auto stop_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - stop_started);
    ok &= Check(timeout_result == dialog::StopResult::Timeout,
                "不合作的生成客户端触发 Stop 超时");
    ok &= Check(stop_elapsed < std::chrono::milliseconds(250),
                "Stop 在 deadline 后有界返回");
    uncooperative_text.release.store(true);
    ok &= Check(timeout_session.Stop(false, std::chrono::seconds(2))
                    == dialog::StopResult::Stopped,
                "客户端退出后可重试 Stop 并回收线程");

    std::cout << (ok ? "\nALL DIALOG MODULE TESTS PASSED\n"
                     : "\nDIALOG MODULE TEST FAILED\n");
    return ok ? 0 : 1;
}
