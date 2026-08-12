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

#include "dialog/sentence_segmenter.h"
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

    /// 不可恢复错误回调（去重）：同一 turn 只产生一次 on_error 终态事件。
    void ReportError(uint64_t task_id, const std::string& error) {
        auto callback = callbacks.on_error;
        if (!callback) return;
        bool should_fire = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!turn_failed) {
                turn_failed = true;
                should_fire = true;
            }
        }
        if (should_fire) callback(task_id, error);
    }

    /// 将当前 turn 置入失败终态：取消 LLM/TTS/剩余 PCM，清除队列并尝试完成 turn。
    /// 由媒体供料失败（PushAudio/PushVideo）路径调用。已失败时去重。
    void FailTurn(uint64_t task_id, const std::string& error) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (turn_failed) return;
            turn_failed = true;
            cancel_current = true;
            sentence_jobs.clear();
            audio_chunks.clear();
            tts_done = !tts_active;
            audio_done = true;
            SetState(SessionState::FAILED);
            MaybeCompleteTurn(lock);
        }
        ReportError(task_id, error);
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
        if (!busy || generation_active || tts_active || audio_active
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
        // 注意：turn_failed 不在此处重置。它由 SubmitUserText 在新 turn 开始时重置，
        // 以保证同一 turn 内多次媒体错误只产生一次 on_error 终态事件（去重）。
        busy = false;
        cancel_current = false;
        current_task_id = 0;
        SetState(SessionState::IDLE);
        auto callback = callbacks.on_turn_complete;
        lock.unlock();
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
            auto on_delta = [&](const std::string& delta) {
                if (IsCancelled(task.id)) return;
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

            if (!was_cancelled) {
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
                } else if (was_cancelled) {
                    sentence_jobs.clear();
                    tts_done = !tts_active;
                    audio_done = audio_chunks.empty() && !audio_active;
                }
                MaybeCompleteTurn(lock);
            }
            if (!ok && !was_cancelled) {
                ReportError(task.id,
                    error.empty() ? "text generation failed" : error);
            } else if (ok && !was_cancelled && callbacks.on_reply_ready) {
                callbacks.on_reply_ready(task.id, full_reply);
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
            auto on_audio = [&](tts::PCMChunk chunk) {
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
                ReportError(job.task_id,
                    error.empty() ? "TTS synthesis failed" : error);
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
    impl_->stop_deadline = std::chrono::steady_clock::time_point::max();
    impl_->SetState(SessionState::IDLE);
    impl_->generation_thread = std::thread([this]() {
        impl_->GenerationLoop();
    });
    impl_->tts_thread = std::thread([this]() { impl_->TTSLoop(); });
    impl_->audio_thread = std::thread([this]() { impl_->AudioLoop(); });
    impl_->video_thread = std::thread([this]() { impl_->VideoLoop(); });
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
    // 工作线程的阻塞等待均检查 stopping / stop_deadline，可在 deadline 内退出。
    if (impl_->generation_thread.joinable()) impl_->generation_thread.join();
    if (impl_->tts_thread.joinable()) impl_->tts_thread.join();
    if (impl_->audio_thread.joinable()) impl_->audio_thread.join();
    if (impl_->video_thread.joinable()) impl_->video_thread.join();
    impl_->media_sink.Finish();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->started = false;
        impl_->busy = false;
        impl_->stop_deadline = std::chrono::steady_clock::time_point::max();
        impl_->SetState(SessionState::STOPPED);
    }
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

}  // namespace dialog
}  // namespace digital_human
