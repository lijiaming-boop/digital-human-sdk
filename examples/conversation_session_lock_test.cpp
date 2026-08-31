/**
 * @file conversation_session_lock_test.cpp
 * @brief ConversationSession 注入客户端锁行为专项测试
 *
 * 回归覆盖（P0 修复）：
 * - SetStreamingVAD / SetASRClient 在锁外调用 Start/Stop：
 *   客户端在 Start 内同步触发回调（VoiceStart → OnBargeIn、cancelled）
 *   不得自死锁（修复前持锁调用 Start，回调重取会话互斥量即死锁，
 *   本测试将挂起直至 CTest 超时）。
 * - PushUserAudio 锁内快照：音频转发到已注入的 ASR/VAD，
 *   未注入时安全丢弃；替换客户端时旧的被 Stop。
 * - barge-in 中途打断：VoiceStart 触发后回合并发出 on_turn_complete。
 * - 失败回合之后下一回合可正常提交并完成（编排器恢复路径）。
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "dialog/asr_client.h"
#include "dialog/conversation_session.h"
#include "audio/streaming_vad.h"

using namespace digital_human;

namespace {

int g_passed = 0;
int g_failed = 0;

void Check(bool ok, const std::string& desc) {
    std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << desc << std::endl;
    ok ? ++g_passed : ++g_failed;
}

// ---- 会话桩 ----

class InstantTextClient final : public dialog::ITextGenerationClient {
public:
    bool Generate(const dialog::GenerateRequest& request,
                  const dialog::TextDeltaCallback& on_delta,
                  const dialog::CancelCheck&,
                  std::string& error) override {
        if (request.user_text.empty()) {
            error = "missing user text";
            return false;
        }
        on_delta("你好。");
        return true;
    }
};

class SlowTTSClient final : public tts::ITTSClient {
public:
    bool Synthesize(const std::string& text,
                    const tts::PCMCallback& on_audio,
                    const tts::CancelCheck& cancelled,
                    std::string& error) override {
        if (text.empty()) {
            error = "empty TTS text";
            return false;
        }
        if (fail_next.exchange(false)) {
            error = "injected TTS failure";
            return false;
        }
        calls.fetch_add(1);
        for (int c = 0; c < 20; ++c) {
            if (cancelled && cancelled()) return false;
            tts::PCMChunk chunk;
            chunk.sample_rate = 16000;
            chunk.channels = 1;
            chunk.samples.resize(320);
            on_audio(std::move(chunk));
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
        return true;
    }

    std::atomic<int> calls{0};
    std::atomic<bool> fail_next{false};
};

class RecordingSink final : public dialog::IDigitalHumanSink {
public:
    bool PushAudio(const std::vector<float>& samples,
                   int64_t pts_ms, std::string&) override {
        std::lock_guard<std::mutex> lock(mutex);
        audio_chunks += 1;
        total_samples += samples.size();
        (void)pts_ms;
        return true;
    }
    bool PushVideo(const cv::Mat& frame, int64_t, std::string&) override {
        std::lock_guard<std::mutex> lock(mutex);
        video_frames += 1;
        return !frame.empty();
    }
    void Finish() override { finished.store(true); }

    std::mutex mutex;
    int audio_chunks = 0;
    size_t total_samples = 0;
    int video_frames = 0;
    std::atomic<bool> finished{false};
};

// ---- 注入桩：Start 内同步触发回调（死锁回归的关键） ----

class SyncCallbackVAD final : public audio::IStreamingVAD {
public:
    explicit SyncCallbackVAD(bool fire_on_start = false)
        : fire_on_start_(fire_on_start) {}

    bool Start(const audio::VADCallback& on_event) override {
        callback = on_event;
        started.store(true);
        if (fire_on_start_) {
            // 修复前：SetStreamingVAD 持会话锁调用 Start，
            // 回调 → OnBargeIn → 再次取锁 → 自死锁。
            callback(audio::VADEvent::VoiceStart);
        }
        return true;
    }

    bool PushAudio(const float* samples, size_t sample_count) override {
        pushed_samples.fetch_add(static_cast<int>(sample_count));
        return true;
    }

    void Stop() override { stop_calls.fetch_add(1); }

    void FireVoiceStart() {
        if (callback) callback(audio::VADEvent::VoiceStart);
    }

    audio::VADCallback callback;
    bool fire_on_start_ = false;
    std::atomic<bool> started{false};
    std::atomic<int> pushed_samples{0};
    std::atomic<int> stop_calls{0};
};

class SyncCancelASR final : public dialog::IASRClient {
public:
    bool Start(const dialog::TranscriptCallback& on_transcript,
               const dialog::ASRCancelCheck& cancelled,
               std::string&) override {
        transcript_cb = on_transcript;
        // 修复前：cancelled() 取会话锁 → 自死锁。
        if (cancelled) (void)cancelled();
        started.store(true);
        return true;
    }

    bool PushAudio(const float* samples, size_t sample_count,
                   std::string&) override {
        pushed_samples.fetch_add(static_cast<int>(sample_count));
        return true;
    }

    bool Stop(std::string&) override {
        stop_calls.fetch_add(1);
        return true;
    }

    dialog::TranscriptCallback transcript_cb;
    dialog::ASRCancelCheck cancelled_cb;
    std::atomic<bool> started{false};
    std::atomic<int> pushed_samples{0};
    std::atomic<int> stop_calls{0};
};

dialog::ConversationConfig BaseConfig() {
    dialog::ConversationConfig config;
    config.session_id = "lock-test";
    config.audio_sample_rate = 16000;
    config.enable_barge_in = true;
    return config;
}

}  // namespace

int main() {
    std::cout << "====== ConversationSession 注入客户端锁行为测试 ======" << std::endl;
    const cv::Mat avatar(16, 16, CV_8UC3, cv::Scalar(64, 96, 128));

    // ---- Test 1: VAD Start 同步触发 VoiceStart 不死锁 ----
    {
        std::cout << "\n--- Test 1: VAD Start 同步回调不死锁 ---" << std::endl;
        InstantTextClient text;
        SlowTTSClient tts;
        RecordingSink sink;
        dialog::ConversationSession session(text, tts, sink);
        Check(session.Start(BaseConfig(), avatar), "会话启动");
        SyncCallbackVAD vad(true);
        session.SetStreamingVAD(&vad);
        Check(vad.started.load(), "VAD 已启动且同步回调返回");
        session.Stop(false);
    }

    // ---- Test 2: ASR Start 同步调用 cancelled 不死锁 ----
    {
        std::cout << "\n--- Test 2: ASR Start 同步 cancelled 不死锁 ---" << std::endl;
        InstantTextClient text;
        SlowTTSClient tts;
        RecordingSink sink;
        dialog::ConversationSession session(text, tts, sink);
        Check(session.Start(BaseConfig(), avatar), "会话启动");
        SyncCancelASR asr;
        session.SetASRClient(&asr);
        Check(asr.started.load(), "ASR 已启动且同步 cancelled 返回");
        session.Stop(false);
    }

    // ---- Test 3: PushUserAudio 转发到注入的 ASR/VAD ----
    {
        std::cout << "\n--- Test 3: PushUserAudio 快照转发 ---" << std::endl;
        InstantTextClient text;
        SlowTTSClient tts;
        RecordingSink sink;
        dialog::ConversationSession session(text, tts, sink);
        Check(session.Start(BaseConfig(), avatar), "会话启动");
        SyncCallbackVAD vad;
        SyncCancelASR asr;
        session.SetStreamingVAD(&vad);
        session.SetASRClient(&asr);
        std::vector<float> pcm(160, 0.01f);
        std::string error;
        Check(session.PushUserAudio(pcm.data(), pcm.size(), error),
              "PushUserAudio 返回 true");
        Check(asr.pushed_samples.load() == 160, "ASR 收到 160 样本");
        Check(vad.pushed_samples.load() == 160, "VAD 收到 160 样本");

        // ---- Test 4: 未注入客户端时安全丢弃 ----
        session.SetASRClient(nullptr);
        session.SetStreamingVAD(nullptr);
        Check(asr.stop_calls.load() == 1, "解绑时旧 ASR 被 Stop");
        Check(vad.stop_calls.load() == 1, "解绑时旧 VAD 被 Stop");
        Check(session.PushUserAudio(pcm.data(), pcm.size(), error),
              "未注入时 PushUserAudio 安全返回 true");
        Check(asr.pushed_samples.load() == 160, "解绑后 ASR 不再收到样本");
        session.Stop(false);
    }

    // ---- Test 5: 替换 VAD 时旧实例被 Stop ----
    {
        std::cout << "\n--- Test 5: 替换 VAD 停止旧实例 ---" << std::endl;
        InstantTextClient text;
        SlowTTSClient tts;
        RecordingSink sink;
        dialog::ConversationSession session(text, tts, sink);
        Check(session.Start(BaseConfig(), avatar), "会话启动");
        SyncCallbackVAD vad_a;
        SyncCallbackVAD vad_b;
        session.SetStreamingVAD(&vad_a);
        session.SetStreamingVAD(&vad_b);
        Check(vad_a.stop_calls.load() == 1, "旧 VAD 被 Stop 一次");
        Check(vad_b.started.load(), "新 VAD 已启动");
        session.Stop(false);
    }

    // ---- Test 6: barge-in 中途打断并完成回合 ----
    {
        std::cout << "\n--- Test 6: barge-in 触发 OnBargeIn 并完成回合 ---" << std::endl;
        InstantTextClient text;
        SlowTTSClient tts;
        RecordingSink sink;
        dialog::ConversationSession session(text, tts, sink);
        std::atomic<int> completed{0};
        dialog::ConversationCallbacks callbacks;
        callbacks.on_turn_complete = [&](uint64_t) { completed.fetch_add(1); };
        Check(session.Start(BaseConfig(), avatar, callbacks), "会话启动");
        SyncCallbackVAD vad;
        session.SetStreamingVAD(&vad);
        Check(session.SubmitUserText("请播报一段较长的内容") != 0, "提交回合");
        // 等待回合进入 TTS/播放阶段
        bool active = false;
        for (int i = 0; i < 200; ++i) {
            const auto state = session.State();
            if (state == dialog::SessionState::SYNTHESIZING
                || state == dialog::SessionState::PLAYING) {
                active = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        Check(active, "回合进入 SYNTHESIZING/PLAYING");
        vad.FireVoiceStart();
        Check(session.WaitUntilIdle(std::chrono::seconds(3)), "打断后回到空闲");
        Check(completed.load() == 1, "打断回合补发一次 on_turn_complete");
        session.Stop(false);
    }

    // ---- Test 7: 失败回合后下一回合可提交并完成 ----
    {
        std::cout << "\n--- Test 7: 失败回合后编排器恢复 ---" << std::endl;
        InstantTextClient text;
        SlowTTSClient tts;
        RecordingSink sink;
        dialog::ConversationSession session(text, tts, sink);
        std::atomic<int> completed{0};
        std::atomic<int> errors{0};
        dialog::ConversationCallbacks callbacks;
        callbacks.on_turn_complete = [&](uint64_t) { completed.fetch_add(1); };
        callbacks.on_error = [&](uint64_t, const std::string&) {
            errors.fetch_add(1);
        };
        Check(session.Start(BaseConfig(), avatar, callbacks), "会话启动");
        tts.fail_next.store(true);
        Check(session.SubmitUserText("触发失败") != 0, "提交失败回合");
        Check(session.WaitUntilIdle(std::chrono::seconds(3)), "失败回合退出忙状态");
        Check(completed.load() == 1 && errors.load() == 1,
              "失败回合发出 on_error 与 on_turn_complete");
        Check(session.SubmitUserText("再来一次") != 0, "失败后可再次提交");
        Check(session.WaitUntilIdle(std::chrono::seconds(3)), "第二回合完成");
        Check(completed.load() == 2, "两回合各补发一次 on_turn_complete");
        session.Stop(false);
    }

    std::cout << "\n====== 汇总: 通过 " << g_passed << " / 失败 "
              << g_failed << " ======" << std::endl;
    return g_failed == 0 ? 0 : 1;
}
