#include "dialog/conversation_session.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "audio/streaming_vad.h"
#include "dialog/asr_client.h"
#include "dialog/sentence_segmenter.h"
#include "digital_human/log_context.h"
#include "digital_human/log_macros.h"
#include "digital_human/metrics.h"
#include "digital_human_sdk.h"

namespace digital_human {
namespace dialog {
namespace {

bool NormalizeAvatarFrame(const cv::Mat& input, cv::Mat& bgr) {
    if (input.empty() || input.depth() != CV_8U) return false;
    switch (input.channels()) {
        case 1:
            cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
            break;
        case 3:
            bgr = input.clone();
            break;
        case 4:
            cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
            break;
        default:
            return false;
    }
    if (!bgr.isContinuous()) bgr = bgr.clone();
    return !bgr.empty();
}

/// 将 BGR 头像适配到固定画布尺寸（P0 头像热更新画布契约）。
/// - 尺寸已一致：直接 clone 返回。
/// - Reject：尺寸不一致时返回 false。
/// - Fit：等比缩放（取较小比例）并居中，多余区域黑色填充。
/// - Cover：等比缩放（取较大比例）填满画布并居中裁剪。
/// 画布宽高在调用前已强制为偶数，保证 H.264 编码兼容。
bool ApplyCanvasPolicy(const cv::Mat& bgr, int canvas_w, int canvas_h,
                       AvatarUpdatePolicy policy, cv::Mat& out,
                       std::string& error) {
    if (bgr.cols == canvas_w && bgr.rows == canvas_h) {
        out = bgr.clone();
        return true;
    }
    if (policy == AvatarUpdatePolicy::Reject) {
        error = "avatar dimensions (" + std::to_string(bgr.cols) + "x"
              + std::to_string(bgr.rows)
              + ") do not match the fixed canvas ("
              + std::to_string(canvas_w) + "x" + std::to_string(canvas_h) + ")";
        return false;
    }
    const double sx = static_cast<double>(canvas_w) / bgr.cols;
    const double sy = static_cast<double>(canvas_h) / bgr.rows;
    const double scale = policy == AvatarUpdatePolicy::Cover
        ? std::max(sx, sy) : std::min(sx, sy);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    if (policy == AvatarUpdatePolicy::Fit) {
        out = cv::Mat::zeros(canvas_h, canvas_w, bgr.type());
        const int x = (canvas_w - resized.cols) / 2;
        const int y = (canvas_h - resized.rows) / 2;
        if (resized.cols > 0 && resized.rows > 0) {
            cv::Rect roi(std::max(0, x), std::max(0, y),
                         std::min(resized.cols, canvas_w),
                         std::min(resized.rows, canvas_h));
            resized.copyTo(out(roi));
        }
    } else {  // Cover
        const int x = (resized.cols - canvas_w) / 2;
        const int y = (resized.rows - canvas_h) / 2;
        out = resized(
            cv::Rect(std::max(0, x), std::max(0, y),
                     std::min(canvas_w, resized.cols),
                     std::min(canvas_h, resized.rows))).clone();
    }
    return true;
}

}  // namespace

SDKDigitalHumanSink::SDKDigitalHumanSink(DigitalHumanSDK& sdk) : sdk_(sdk) {}

bool SDKDigitalHumanSink::PushAudio(const std::vector<float>& samples,
                                    int64_t pts_ms,
                                    std::string& error) {
    const auto result = sdk_.PushAudio(samples, pts_ms);
    if (result == SDKError::OK) return true;
    error = std::string("PushAudio failed: ") + SDKErrorToString(result)
          + ": " + sdk_.GetLastError();
    return false;
}

bool SDKDigitalHumanSink::PushVideo(const cv::Mat& frame,
                                    int64_t pts_ms,
                                    std::string& error) {
    const auto result = sdk_.PushVideo(frame, pts_ms);
    if (result == SDKError::OK) return true;
    error = std::string("PushVideo failed: ") + SDKErrorToString(result)
          + ": " + sdk_.GetLastError();
    return false;
}

void SDKDigitalHumanSink::Finish() {
    sdk_.MarkAudioEOS();
    sdk_.MarkVideoEOS();
}

struct ConversationSession::Impl {
    struct UserTask {
        uint64_t id = 0;
        std::string text;
    };
    struct SentenceJob {
        uint64_t task_id = 0;
        std::string text;
        bool end_of_reply = false;
    };

    ITextGenerationClient& text_client;
    tts::ITTSClient& tts_client;
    IDigitalHumanSink& media_sink;

    ConversationConfig config;
    ConversationCallbacks callbacks;
    cv::Mat avatar_frame;
    /// 固定画布尺寸（P0 热更新）：Start 时确定，UpdateAvatar 必须服从。
    int avatar_canvas_width = 0;
    int avatar_canvas_height = 0;

    /// Barge-in（P1-3.10c）：可选的 ASR 与流式 VAD，由调用方注入。
    /// 生命周期由调用方管理，Stop 前须保持有效。
    IASRClient*           asr_client = nullptr;
    audio::IStreamingVAD* streaming_vad = nullptr;
    /// ASR 是否已 Start（避免重复 Start，并在 Stop 时正确回收）。
    bool asr_started = false;
    bool vad_started = false;

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<UserTask> user_tasks;
    std::deque<SentenceJob> sentence_jobs;
    std::deque<tts::PCMChunk> audio_chunks;
    std::vector<ChatMessage> history;

    std::thread generation_thread;
    std::thread tts_thread;
    std::thread audio_thread;
    std::thread video_thread;
    bool generation_exited = true;
    bool tts_exited = true;
    bool audio_exited = true;
    bool video_exited = true;

    bool started = false;
    bool stopping = false;
    bool busy = false;
    bool cancel_current = false;
    bool generation_active = false;
    bool tts_active = false;
    bool audio_active = false;
    bool generation_done = true;
    bool tts_done = true;
    bool audio_done = true;
    /// 标记当前 turn 已进入失败终态，使后续错误回调被去重（只产生一次终态事件）。
    bool turn_failed = false;
    bool failure_callback_pending = false;
    uint64_t current_task_id = 0;
    uint64_t next_task_id = 1;

    /// 显式会话状态机，反映当前 turn 所处阶段。
    SessionState state = SessionState::IDLE;

    /// Stop() 共享截止时间：阻塞 Push/Pop 在超过该时间后立即放弃，避免无界等待。
    /// 为 steady_clock::time_point::max() 时表示未设置 deadline。
    std::chrono::steady_clock::time_point stop_deadline =
        std::chrono::steady_clock::time_point::max();

    int64_t audio_sample_cursor = 0;
    int64_t audio_submitted_until_ms = 0;
    int64_t turn_audio_start_ms = 0;
    int64_t video_frame_cursor = 0;
    int64_t next_video_pts_ms = 0;

    Impl(ITextGenerationClient& text,
         tts::ITTSClient& tts,
         IDigitalHumanSink& sink)
        : text_client(text), tts_client(tts), media_sink(sink) {}

    void SetState(SessionState next) { state = next; }

    bool StopDeadlinePassed() const {
        return stop_deadline != std::chrono::steady_clock::time_point::max()
            && std::chrono::steady_clock::now() >= stop_deadline;
    }

    bool IsCancelled(uint64_t task_id) const {
        std::lock_guard<std::mutex> lock(mutex);
        return stopping || cancel_current || current_task_id != task_id;
    }

    /// 将当前 turn 置入失败终态：取消 LLM/TTS/剩余 PCM，清除队列并尝试完成 turn。
    /// 所有不可恢复错误都走同一路径，同一 turn 只产生一次 on_error。
    void FailTurn(uint64_t task_id, const std::string& error) {
        std::function<void(uint64_t, const std::string&)> callback;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (turn_failed) return;
            turn_failed = true;
            failure_callback_pending = true;
            cancel_current = true;
            user_tasks.clear();
            sentence_jobs.clear();
            audio_chunks.clear();
            generation_done = !generation_active;
            tts_done = !tts_active;
            audio_done = !audio_active;
            SetState(SessionState::FAILED);
            callback = callbacks.on_error;
        }
        MetricsRegistry::instance().record_turn_complete(false, false);
        if (callback) callback(task_id, error);
        {
            std::unique_lock<std::mutex> lock(mutex);
            failure_callback_pending = false;
            MaybeCompleteTurn(lock);
        }
        cv.notify_all();
    }

    /// History 预算截断（P1-3.10a）：按轮次/字符/token 三重限制裁剪历史，
    /// 防止多轮对话导致 LLM 上下文溢出和无界内存增长。
    /// 在持有 mutex 时调用。
    void TrimHistory() {
        // 1. 按轮次截断：1 轮 = 1 user + 1 assistant = 2 条消息
        const int max_messages = config.max_history_turns * 2;
        if (max_messages > 0
            && static_cast<int>(history.size()) > max_messages) {
            history.erase(history.begin(),
                          history.begin() + (history.size() - max_messages));
        }
        // 2. 按字符截断：从最旧消息开始删除
        size_t total_chars = 0;
        for (const auto& msg : history) total_chars += msg.content.size();
        while (total_chars > static_cast<size_t>(config.max_history_chars)
               && history.size() > 2) {
            total_chars -= history.front().content.size();
            history.erase(history.begin());
        }
        // 3. token 估算（粗略：chars / 3.5 for CJK, chars / 4 for Latin）
        if (config.max_history_tokens_estimate > 0) {
            int estimated_tokens = static_cast<int>(total_chars / 3.5);
            while (estimated_tokens > config.max_history_tokens_estimate
                   && history.size() > 2) {
                total_chars -= history.front().content.size();
                history.erase(history.begin());
                estimated_tokens = static_cast<int>(total_chars / 3.5);
            }
        }
        MetricsRegistry::instance().record_history_length(history.size());
    }

    /// Barge-in 处理（P1-3.10c）：VAD 检测到用户语音时自动打断当前回复。
    /// 流程：取消 LLM/TTS → 清理 PCM → 重对齐时间轴 → 进入 INTERRUPTING →
    ///       尝试完成当前 turn → 上报指标。
    /// 仅在 enable_barge_in 且处于 PLAYING/SYNTHESIZING 时触发，
    /// 避免在 IDLE 或已中断状态下重复触发。
    void OnBargeIn() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!config.enable_barge_in || !busy || turn_failed) return;
        if (state != SessionState::PLAYING
            && state != SessionState::SYNTHESIZING
            && state != SessionState::GENERATING) {
            return;
        }
        DH_LOG_INFO("dialog.session")
            << "barge-in triggered (state="
            << static_cast<int>(state) << ")";

        // 1. 取消当前 LLM/TTS（复用 Interrupt 逻辑）
        cancel_current = true;
        sentence_jobs.clear();
        audio_chunks.clear();
        tts_done = !tts_active;
        audio_done = !audio_active;

        // 2. 重对齐音频时间轴：清除已提交音频的 PTS 基准，
        //    使下一个 turn 从零开始，避免旧 turn 的音频残留影响同步。
        audio_sample_cursor = 0;
        audio_submitted_until_ms = 0;
        turn_audio_start_ms = 0;
        video_frame_cursor = 0;
        next_video_pts_ms = 0;

        // 3. 进入 INTERRUPTING 状态，尝试完成当前 turn
        SetState(SessionState::INTERRUPTING);
        MetricsRegistry::instance().record_barge_in();
        MaybeCompleteTurn(lock);
        cv.notify_all();
    }

    void EnqueueSentence(uint64_t task_id,
                         std::string text,
                         bool end_of_reply) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping || cancel_current || current_task_id != task_id) return;
            sentence_jobs.push_back(
                SentenceJob{task_id, std::move(text), end_of_reply});
            tts_done = false;
            audio_done = false;
        }
        cv.notify_all();
    }

    bool EnqueueAudio(uint64_t task_id, tts::PCMChunk chunk) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() {
            return stopping || cancel_current || current_task_id != task_id
                || audio_chunks.size() < config.max_pending_audio_chunks
                || StopDeadlinePassed();
        });
        if (stopping || cancel_current || current_task_id != task_id
            || StopDeadlinePassed()) {
            return false;
        }
        audio_chunks.push_back(std::move(chunk));
        audio_done = false;
        lock.unlock();
        cv.notify_all();
        return true;
    }

    void MaybeCompleteTurn(std::unique_lock<std::mutex>& lock) {
        if (!busy || failure_callback_pending
            || generation_active || tts_active || audio_active
            || !generation_done || !tts_done || !audio_done
            || !user_tasks.empty() || !sentence_jobs.empty()
            || !audio_chunks.empty()) {
            return;
        }
        if (audio_submitted_until_ms > turn_audio_start_ms) {
            const int64_t video_ready_until =
                audio_submitted_until_ms - config.mel_lookahead_ms;
            if (video_ready_until >= turn_audio_start_ms
                && next_video_pts_ms <= video_ready_until) {
                return;
            }
        }

        const uint64_t completed_id = current_task_id;
        if (turn_failed) {
            busy = false;
            current_task_id = 0;
            SetState(SessionState::FAILED);
            lock.unlock();
            cv.notify_all();
            lock.lock();
            return;
        }
        const bool was_interrupted = state == SessionState::INTERRUPTING;
        busy = false;
        cancel_current = false;
        current_task_id = 0;
        SetState(SessionState::IDLE);
        auto callback = callbacks.on_turn_complete;
        lock.unlock();
        MetricsRegistry::instance().record_turn_complete(!was_interrupted,
                                                         was_interrupted);
        if (callback) callback(completed_id);
        lock.lock();
        cv.notify_all();
    }

    void GenerationLoop() {
        while (true) {
            UserTask task;
            GenerateRequest request;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]() { return stopping || !user_tasks.empty(); });
                if (stopping && user_tasks.empty()) break;
                task = std::move(user_tasks.front());
                user_tasks.pop_front();
                generation_active = true;
                generation_done = false;
                request.session_id = config.session_id;
                request.system_prompt = config.system_prompt;
                request.user_text = task.text;
                request.history = history;
            }

            SentenceSegmenter segmenter(
                SentenceSegmenterConfig{config.min_tts_clause_chars});
            std::string full_reply;
            LogContext log_context(config.session_id, task.id);
            auto& metrics = MetricsRegistry::instance();
            metrics.record_llm_request_start();
            const auto llm_started = std::chrono::steady_clock::now();
            bool first_token_recorded = false;
            auto on_delta = [&](const std::string& delta) {
                if (IsCancelled(task.id)) return;
                if (!first_token_recorded && !delta.empty()) {
                    first_token_recorded = true;
                    metrics.record_llm_first_token(
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - llm_started).count());
                }
                full_reply += delta;
                auto callback = callbacks.on_text_delta;
                if (callback) callback(task.id, delta);
                for (auto& clause : segmenter.Push(delta)) {
                    EnqueueSentence(task.id, std::move(clause), false);
                }
            };
            auto cancelled = [&]() { return IsCancelled(task.id); };
            std::string error;
            const bool ok = text_client.Generate(
                request, on_delta, cancelled, error);
            const bool was_cancelled = cancelled();
            const double llm_total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - llm_started).count();

            if (ok && !was_cancelled) {
                auto tail = segmenter.Flush();
                if (!tail.empty()) {
                    EnqueueSentence(task.id, std::move(tail), false);
                }
                EnqueueSentence(task.id, {}, true);
            }

            {
                std::unique_lock<std::mutex> lock(mutex);
                generation_active = false;
                generation_done = true;
                if (ok && !was_cancelled) {
                    history.push_back(ChatMessage{"user", task.text});
                    history.push_back(ChatMessage{"assistant", full_reply});
                    TrimHistory();
                } else if (was_cancelled) {
                    sentence_jobs.clear();
                    tts_done = !tts_active;
                    audio_done = audio_chunks.empty() && !audio_active;
                }
                if (ok || was_cancelled) MaybeCompleteTurn(lock);
            }
            if (!ok && !was_cancelled) {
                metrics.record_llm_failed();
                FailTurn(task.id,
                    error.empty() ? "text generation failed" : error);
            } else if (was_cancelled) {
                metrics.record_llm_cancelled(llm_total_ms);
            } else if (ok && !was_cancelled && callbacks.on_reply_ready) {
                const double seconds = llm_total_ms / 1000.0;
                const double estimated_tokens = full_reply.size() / 4.0;
                metrics.record_llm_complete(
                    llm_total_ms, seconds > 0.0 ? estimated_tokens / seconds : 0.0, 0);
                callbacks.on_reply_ready(task.id, full_reply);
            } else if (ok && !was_cancelled) {
                const double seconds = llm_total_ms / 1000.0;
                metrics.record_llm_complete(
                    llm_total_ms,
                    seconds > 0.0 ? (full_reply.size() / 4.0) / seconds : 0.0,
                    0);
            }
            cv.notify_all();
        }
    }

    void TTSLoop() {
        while (true) {
            SentenceJob job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]() {
                    return stopping || !sentence_jobs.empty();
                });
                if (stopping && sentence_jobs.empty()) break;
                job = std::move(sentence_jobs.front());
                sentence_jobs.pop_front();
            }
            LogContext log_context(config.session_id, job.task_id);

            if (job.end_of_reply) {
                const int silence_samples = config.audio_sample_rate
                    * config.reply_tail_silence_ms / 1000;
                if (silence_samples > 0) {
                    tts::PCMChunk silence;
                    silence.sample_rate = config.audio_sample_rate;
                    silence.channels = config.audio_channels;
                    silence.samples.assign(
                        static_cast<size_t>(silence_samples)
                            * static_cast<size_t>(config.audio_channels),
                        0.0f);
                    EnqueueAudio(job.task_id, std::move(silence));
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    tts_done = true;
                }
                cv.notify_all();
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                if (cancel_current || current_task_id != job.task_id) continue;
                tts_active = true;
                // 首个句子开始合成 → 由 GENERATING 进入 SYNTHESIZING。
                if (state == SessionState::GENERATING) {
                    SetState(SessionState::SYNTHESIZING);
                }
            }
            auto cancelled = [&]() { return IsCancelled(job.task_id); };
            auto& metrics = MetricsRegistry::instance();
            metrics.record_tts_request_start();
            const auto tts_started = std::chrono::steady_clock::now();
            bool first_pcm_recorded = false;
            uint64_t total_samples = 0;
            auto on_audio = [&](tts::PCMChunk chunk) {
                if (!first_pcm_recorded && !chunk.samples.empty()) {
                    first_pcm_recorded = true;
                    metrics.record_tts_first_pcm(
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - tts_started).count());
                }
                total_samples += chunk.samples.size();
                return EnqueueAudio(job.task_id, std::move(chunk));
            };
            std::string error;
            const bool ok = tts_client.Synthesize(
                job.text, on_audio, cancelled, error);
            {
                std::lock_guard<std::mutex> lock(mutex);
                tts_active = false;
                if ((cancel_current || stopping)
                    && sentence_jobs.empty()) {
                    tts_done = true;
                }
            }
            if (!ok && !cancelled()) {
                metrics.record_tts_failed();
                FailTurn(job.task_id,
                    error.empty() ? "TTS synthesis failed" : error);
            } else if (ok) {
                const double elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tts_started).count();
                const double audio_ms = config.audio_sample_rate > 0
                    ? static_cast<double>(total_samples) * 1000.0
                        / static_cast<double>(config.audio_sample_rate
                                              * config.audio_channels)
                    : 0.0;
                metrics.record_tts_complete(
                    audio_ms > 0.0 ? elapsed_ms / audio_ms : 0.0,
                    total_samples);
            }
            cv.notify_all();
        }
    }

    void AudioLoop() {
        while (true) {
            tts::PCMChunk chunk;
            uint64_t task_id = 0;
            int64_t pts_ms = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]() {
                    return stopping || !audio_chunks.empty()
                        || (busy && !audio_done && tts_done && !tts_active);
                });
                if (stopping && audio_chunks.empty()) break;
                if (audio_chunks.empty()) {
                    audio_done = true;
                    MaybeCompleteTurn(lock);
                    continue;
                }
                chunk = std::move(audio_chunks.front());
                audio_chunks.pop_front();
                task_id = current_task_id;
                pts_ms = audio_sample_cursor * 1000 / config.audio_sample_rate;
                audio_active = true;
            }

            std::string error;
            bool ok = true;
            if (chunk.sample_rate != config.audio_sample_rate
                || chunk.channels != config.audio_channels) {
                error = "TTS PCM format must match ConversationConfig";
                ok = false;
            } else if (chunk.samples.empty()
                       || chunk.samples.size()
                            % static_cast<size_t>(chunk.channels) != 0) {
                error = "TTS returned an empty or unaligned PCM chunk";
                ok = false;
            } else {
                ok = media_sink.PushAudio(chunk.samples, pts_ms, error);
            }

            {
                std::unique_lock<std::mutex> lock(mutex);
                if (ok) {
                    audio_sample_cursor += static_cast<int64_t>(
                        chunk.samples.size()
                        / static_cast<size_t>(chunk.channels));
                    audio_submitted_until_ms =
                        audio_sample_cursor * 1000 / config.audio_sample_rate;
                    // 首个音频样本成功送入 → 进入 PLAYING 阶段。
                    if (state == SessionState::SYNTHESIZING) {
                        SetState(SessionState::PLAYING);
                    }
                }
                audio_active = false;
                if (ok) {
                    audio_done = audio_chunks.empty() && tts_done && !tts_active;
                    MaybeCompleteTurn(lock);
                }
            }
            // 媒体供料失败：进入失败终态，取消当前 turn 的 LLM/TTS/剩余 PCM。
            if (!ok) {
                FailTurn(task_id, error);
            }
            cv.notify_all();
        }
    }

    void VideoLoop() {
        while (true) {
            int64_t pts_ms = 0;
            uint64_t task_id = 0;
            cv::Mat avatar_snapshot;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]() {
                    return stopping
                        || (busy && next_video_pts_ms
                            + config.mel_lookahead_ms
                            <= audio_submitted_until_ms)
                        || (busy && generation_done && tts_done && audio_done);
                });
                if (stopping) break;
                if (next_video_pts_ms + config.mel_lookahead_ms
                    > audio_submitted_until_ms) {
                    MaybeCompleteTurn(lock);
                    continue;
                }
                pts_ms = next_video_pts_ms;
                task_id = current_task_id;
                ++video_frame_cursor;
                next_video_pts_ms = static_cast<int64_t>(std::llround(
                    static_cast<double>(video_frame_cursor) * 1000.0
                    / config.target_fps));
                avatar_snapshot = avatar_frame;
            }

            std::string error;
            if (!media_sink.PushVideo(avatar_snapshot, pts_ms, error)) {
                // 视频供料失败：进入失败终态，停止后续供料并取消当前 turn。
                FailTurn(task_id, error);
            }
            {
                std::unique_lock<std::mutex> lock(mutex);
                MaybeCompleteTurn(lock);
            }
            cv.notify_all();
        }
    }
};

ConversationSession::ConversationSession(ITextGenerationClient& text_client,
                                         tts::ITTSClient& tts_client,
                                         IDigitalHumanSink& media_sink)
    : impl_(std::make_unique<Impl>(text_client, tts_client, media_sink)) {}

ConversationSession::~ConversationSession() {
    Stop(false);
}

bool ConversationSession::Start(const ConversationConfig& config,
                                const cv::Mat& avatar_frame,
                                ConversationCallbacks callbacks) {
    cv::Mat normalized_avatar;
    if (!NormalizeAvatarFrame(avatar_frame, normalized_avatar)
        || config.audio_sample_rate <= 0
        || config.audio_channels != 1 || config.target_fps <= 0.0
        || config.mel_lookahead_ms < 0
        || config.reply_tail_silence_ms < config.mel_lookahead_ms
        || config.max_pending_audio_chunks == 0) {
        return false;
    }
    // 固定画布尺寸（P0 热更新）：未显式指定时以初始头像尺寸为画布，并强制偶数宽高，
    // 保证 H.264 编码流分辨率在 Session 生命周期内不变。
    int canvas_w = config.avatar_canvas_width > 0
        ? config.avatar_canvas_width : normalized_avatar.cols;
    int canvas_h = config.avatar_canvas_height > 0
        ? config.avatar_canvas_height : normalized_avatar.rows;
    if (canvas_w % 2 != 0) canvas_w -= 1;
    if (canvas_h % 2 != 0) canvas_h -= 1;
    if (canvas_w <= 0 || canvas_h <= 0) return false;
    cv::Mat canvas_avatar;
    std::string canvas_error;
    if (!ApplyCanvasPolicy(normalized_avatar, canvas_w, canvas_h,
                           config.avatar_update_policy,
                           canvas_avatar, canvas_error)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->started) return false;
    impl_->config = config;
    impl_->callbacks = std::move(callbacks);
    impl_->avatar_frame = std::move(canvas_avatar);
    impl_->avatar_canvas_width = canvas_w;
    impl_->avatar_canvas_height = canvas_h;
    impl_->started = true;
    impl_->stopping = false;
    impl_->turn_failed = false;
    impl_->failure_callback_pending = false;
    impl_->stop_deadline = std::chrono::steady_clock::time_point::max();
    impl_->generation_exited = false;
    impl_->tts_exited = false;
    impl_->audio_exited = false;
    impl_->video_exited = false;
    impl_->SetState(SessionState::IDLE);
    impl_->generation_thread = std::thread([this]() {
        impl_->GenerationLoop();
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->generation_exited = true;
        }
        impl_->cv.notify_all();
    });
    impl_->tts_thread = std::thread([this]() {
        impl_->TTSLoop();
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->tts_exited = true;
        }
        impl_->cv.notify_all();
    });
    impl_->audio_thread = std::thread([this]() {
        impl_->AudioLoop();
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->audio_exited = true;
        }
        impl_->cv.notify_all();
    });
    impl_->video_thread = std::thread([this]() {
        impl_->VideoLoop();
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->video_exited = true;
        }
        impl_->cv.notify_all();
    });
    MetricsRegistry::instance().record_session_start();
    return true;
}

bool ConversationSession::UpdateAvatar(const cv::Mat& avatar_frame) {
    cv::Mat normalized_avatar;
    if (!NormalizeAvatarFrame(avatar_frame, normalized_avatar)) return false;

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->started || impl_->stopping) return false;
    // 服从固定画布：按配置策略将新头像适配到 Start 时确定的画布尺寸。
    cv::Mat canvas_avatar;
    std::string canvas_error;
    if (!ApplyCanvasPolicy(normalized_avatar,
                           impl_->avatar_canvas_width,
                           impl_->avatar_canvas_height,
                           impl_->config.avatar_update_policy,
                           canvas_avatar, canvas_error)) {
        return false;
    }
    impl_->avatar_frame = std::move(canvas_avatar);
    return true;
}

uint64_t ConversationSession::SubmitUserText(const std::string& text) {
    if (text.empty()) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->started || impl_->stopping || impl_->busy) return 0;
    const uint64_t id = impl_->next_task_id++;
    impl_->current_task_id = id;
    impl_->busy = true;
    impl_->cancel_current = false;
    impl_->turn_failed = false;
    impl_->failure_callback_pending = false;
    impl_->generation_done = false;
    impl_->tts_done = false;
    impl_->audio_done = false;
    impl_->turn_audio_start_ms = impl_->audio_submitted_until_ms;
    impl_->SetState(SessionState::GENERATING);
    impl_->user_tasks.push_back(Impl::UserTask{id, text});
    impl_->cv.notify_all();
    return id;
}

void ConversationSession::Interrupt() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->busy) return;
    impl_->cancel_current = true;
    impl_->sentence_jobs.clear();
    impl_->audio_chunks.clear();
    impl_->tts_done = !impl_->tts_active;
    impl_->audio_done = !impl_->audio_active;
    impl_->SetState(SessionState::INTERRUPTING);
    impl_->cv.notify_all();
}

bool ConversationSession::WaitUntilIdle(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    return impl_->cv.wait_for(lock, timeout, [&]() {
        return !impl_->busy || !impl_->started;
    });
}

StopResult ConversationSession::Stop(bool drain,
                                     std::chrono::milliseconds timeout) {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->started) return StopResult::Stopped;
        // 设置共享截止时间，使所有阻塞 Push/Pop 在超过 deadline 后立即放弃。
        impl_->stop_deadline = std::chrono::steady_clock::now() + timeout;
    }

    StopResult result = StopResult::Stopped;
    if (drain) {
        // drain 阶段：在剩余 timeout 内等待当前 turn 完成，超时则强制 Interrupt。
        if (!WaitUntilIdle(timeout)) {
            Interrupt();
            result = StopResult::Timeout;
        }
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stopping = true;
        impl_->cancel_current = true;
        impl_->SetState(SessionState::STOPPING);
        if (!drain) {
            impl_->user_tasks.clear();
            impl_->sentence_jobs.clear();
            impl_->audio_chunks.clear();
        }
    }
    impl_->cv.notify_all();
    // 先等待线程报告退出，再 join；超时时保留 thread 所有权供后续 Stop 重试。
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        const bool workers_exited = impl_->cv.wait_until(
            lock, impl_->stop_deadline, [&]() {
                return impl_->generation_exited && impl_->tts_exited
                    && impl_->audio_exited && impl_->video_exited;
            });
        if (!workers_exited) return StopResult::Timeout;
    }
    if (impl_->generation_thread.joinable()) impl_->generation_thread.join();
    if (impl_->tts_thread.joinable()) impl_->tts_thread.join();
    if (impl_->audio_thread.joinable()) impl_->audio_thread.join();
    if (impl_->video_thread.joinable()) impl_->video_thread.join();
    impl_->media_sink.Finish();
    // 回收 ASR/VAD（P1-3.10c）：工作线程退出后再停止，避免回调竞争
    if (impl_->streaming_vad && impl_->vad_started) {
        impl_->streaming_vad->Stop();
        impl_->vad_started = false;
    }
    if (impl_->asr_client && impl_->asr_started) {
        std::string asr_err;
        impl_->asr_client->Stop(asr_err);
        impl_->asr_started = false;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->started = false;
        impl_->busy = false;
        impl_->stop_deadline = std::chrono::steady_clock::time_point::max();
        impl_->SetState(SessionState::STOPPED);
    }
    MetricsRegistry::instance().record_session_end();
    impl_->cv.notify_all();
    return result;
}

bool ConversationSession::IsBusy() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->busy;
}

SessionState ConversationSession::State() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->state;
}

void ConversationSession::SetASRClient(IASRClient* client) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    // 解除旧绑定：若已 Start 则先 Stop
    if (impl_->asr_client && impl_->asr_started) {
        std::string err;
        impl_->asr_client->Stop(err);
        impl_->asr_started = false;
    }
    impl_->asr_client = client;
    if (client && impl_->started && !impl_->stopping) {
        auto on_transcript = [this](const Transcript& t) {
            // ASR 最终结果可作为新 turn 的用户输入
            if (t.is_final && !t.text.empty()) {
                // 通过 callbacks 通知上层，由上层决定是否 SubmitUserText
                // 此处不直接调用 SubmitUserText，避免在 ASR 回调线程中重入
                auto cb = impl_->callbacks.on_text_delta;
                // ASR 结果暂不自动提交，留给上层处理
                (void)cb;
            }
        };
        auto cancelled = [this]() {
            std::lock_guard<std::mutex> lk(impl_->mutex);
            return impl_->stopping;
        };
        std::string err;
        impl_->asr_started = client->Start(on_transcript, cancelled, err);
        if (!impl_->asr_started) {
            DH_LOG_WARN("dialog.session")
                << "ASR Start failed: " << err;
        }
    }
}

void ConversationSession::SetStreamingVAD(audio::IStreamingVAD* vad) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    // 解除旧绑定：若已 Start 则先 Stop
    if (impl_->streaming_vad && impl_->vad_started) {
        impl_->streaming_vad->Stop();
        impl_->vad_started = false;
    }
    impl_->streaming_vad = vad;
    if (vad && impl_->started && !impl_->stopping) {
        auto on_event = [this](audio::VADEvent event) {
            if (event == audio::VADEvent::VoiceStart) {
                impl_->OnBargeIn();
            }
        };
        impl_->vad_started = vad->Start(on_event);
        if (!impl_->vad_started) {
            DH_LOG_WARN("dialog.session") << "VAD Start failed";
        }
    }
}

bool ConversationSession::PushUserAudio(const float* samples,
                                        size_t sample_count,
                                        std::string& error) {
    if (!samples || sample_count == 0) {
        error = "PushUserAudio: null samples or zero count";
        return false;
    }
    // 不持锁地转发音频，避免 ASR/VAD 内部处理阻塞会话线程
    if (impl_->asr_client && impl_->asr_started) {
        std::string asr_err;
        if (!impl_->asr_client->PushAudio(samples, sample_count, asr_err)) {
            DH_LOG_WARN("dialog.session")
                << "ASR PushAudio failed: " << asr_err;
        }
    }
    if (impl_->streaming_vad && impl_->vad_started) {
        impl_->streaming_vad->PushAudio(samples, sample_count);
    }
    return true;
}

}  // namespace dialog
}  // namespace digital_human
