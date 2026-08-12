// ============================================================================
// Digital Human SDK — 统一入口实现
//
// 封装 Pipeline + ModelInferencer + AudioLoader + ImageLoader，
// 提供简化生命周期与文件级便捷接口。
// ============================================================================

#include "digital_human_sdk.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include "core/pipeline.h"
#include "audio/audio_loader.h"
#include "core/image_loader.h"

namespace digital_human {

// ============================================================================
// Impl
// ============================================================================

struct DigitalHumanSDK::Impl {
    // ---- 核心模块 ----
    std::unique_ptr<core::Pipeline> pipeline;
    audio::AudioLoader              audio_loader;
    core::ImageLoader               image_loader;

    // ---- 配置缓存 ----
    SDKConfig config;
    bool      lipsync_model_loaded = false;
    bool      face_model_loaded    = false;
    bool      gpu_enabled          = false;

    // ---- 状态机 ----
    std::atomic<SDKState> state{SDKState::UNINITIALIZED};
    std::atomic<int64_t> lifecycle_transition_count{0};
    mutable std::mutex lifecycle_mutex;

    // ---- 最后错误 ----
    // 简单互斥保护即可，错误路径非热点
    mutable std::mutex   last_error_mtx;
    std::string          last_error;

    void SetLastError(const std::string& msg) {
        std::lock_guard<std::mutex> lk(last_error_mtx);
        last_error = msg;
    }

    static bool IsTerminalState(SDKState value) {
        return value == SDKState::STOPPING || value == SDKState::STOPPED;
    }

    void TransitionStateUnlocked(SDKState next) {
        const SDKState previous = state.load(std::memory_order_relaxed);
        if (previous == next) {
            return;
        }
        state.store(next, std::memory_order_release);
        lifecycle_transition_count.fetch_add(1, std::memory_order_relaxed);
    }

    SDKError Fail(SDKError error, const std::string& message) {
        SetLastError(message);
        return error;
    }

    SDKError ValidateMutationUnlocked(const char* operation) {
        if (!pipeline) {
            return Fail(SDKError::NOT_INITIALIZED,
                        std::string(operation) + ": Pipeline is not initialized");
        }
        const SDKState current = state.load(std::memory_order_acquire);
        if (IsTerminalState(current)) {
            return Fail(SDKError::ALREADY_TERMINATED,
                        std::string(operation) + ": SDK is terminal");
        }
        if (current == SDKState::RUNNING || current == SDKState::PAUSED) {
            return Fail(SDKError::ALREADY_RUNNING,
                        std::string(operation) + ": SDK is running");
        }
        return SDKError::OK;
    }

    SDKError LoadLipSyncModelUnlocked(const std::string& model_dir) {
        const SDKError validation = ValidateMutationUnlocked("LoadLipSyncModel");
        if (validation != SDKError::OK) {
            return validation;
        }
        if (!pipeline->InitModelInferencer(model_dir)) {
            lipsync_model_loaded = false;
            return Fail(SDKError::MODEL_LOAD_FAILED,
                        "LoadLipSyncModel: failed to load model: " + model_dir);
        }
        lipsync_model_loaded = true;
        SetLastError("");
        return SDKError::OK;
    }

    SDKError LoadFaceModelUnlocked(const std::string& model_dir) {
        const SDKError validation = ValidateMutationUnlocked("LoadFaceModel");
        if (validation != SDKError::OK) {
            return validation;
        }
        if (!pipeline->LoadFaceModel(model_dir)) {
            face_model_loaded = false;
            return Fail(SDKError::FACE_MODEL_LOAD_FAILED,
                        "LoadFaceModel: failed to load SCRFD and 2D106 models: "
                            + model_dir);
        }
        face_model_loaded = true;
        SetLastError("");
        return SDKError::OK;
    }

    SDKError EnableGPUUnlocked(bool enable) {
        const SDKError validation = ValidateMutationUnlocked("EnableGPU");
        if (validation != SDKError::OK) {
            return validation;
        }
        if (!pipeline->EnableGPU(enable)) {
            return Fail(enable ? SDKError::GPU_NOT_AVAILABLE : SDKError::UNKNOWN,
                        enable
                            ? "EnableGPU: Vulkan is unavailable or model reload failed"
                            : "EnableGPU: failed to disable GPU");
        }
        gpu_enabled = enable;
        SetLastError("");
        return SDKError::OK;
    }

    SDKError SetInferenceThreadsUnlocked(int count) {
        if (count < 0) {
            return Fail(SDKError::INVALID_CONFIG,
                        "SetInferenceThreads: thread count cannot be negative");
        }
        const SDKError validation = ValidateMutationUnlocked("SetInferenceThreads");
        if (validation != SDKError::OK) {
            return validation;
        }
        pipeline->SetInferenceThreads(count);
        SetLastError("");
        return SDKError::OK;
    }

    SDKError StartUnlocked() {
        const SDKState current = state.load(std::memory_order_acquire);
        if (current == SDKState::UNINITIALIZED || !pipeline) {
            return Fail(SDKError::NOT_INITIALIZED, "Start: SDK is not initialized");
        }
        if (IsTerminalState(current)) {
            return Fail(SDKError::ALREADY_TERMINATED,
                        "Start: SDK is terminal and cannot be restarted");
        }
        if (current == SDKState::RUNNING) {
            SetLastError("");
            return SDKError::OK;
        }
        if (current == SDKState::PAUSED) {
            pipeline->Resume();
            TransitionStateUnlocked(SDKState::RUNNING);
            SetLastError("");
            return SDKError::OK;
        }
        if (!lipsync_model_loaded) {
            return Fail(SDKError::MODEL_LOAD_FAILED,
                        "Start: Wav2Lip model is not loaded");
        }
        if (!face_model_loaded || !pipeline->IsFaceModelLoaded()) {
            return Fail(SDKError::FACE_MODEL_LOAD_FAILED,
                        "Start: SCRFD and 2D106 face models are not loaded");
        }
        if (!pipeline->Start()) {
            return Fail(SDKError::PIPELINE_START_FAILED,
                        "Start: Pipeline failed to start");
        }
        TransitionStateUnlocked(SDKState::RUNNING);
        SetLastError("");
        return SDKError::OK;
    }

    SDKError StopUnlocked() {
        const SDKState current = state.load(std::memory_order_acquire);
        if (current == SDKState::UNINITIALIZED || current == SDKState::STOPPED) {
            return SDKError::OK;
        }
        if (current != SDKState::STOPPING) {
            TransitionStateUnlocked(SDKState::STOPPING);
        }
        if (pipeline && !pipeline->Stop()) {
            return Fail(SDKError::SHUTDOWN_TIMEOUT,
                        "Stop: shutdown timeout; call Stop again to retry");
        }
        TransitionStateUnlocked(SDKState::STOPPED);
        SetLastError("");
        return SDKError::OK;
    }

    // ------------------------------------------------------------------
    // 配置转换：SDKConfig → PipelineConfig
    // ------------------------------------------------------------------
    core::PipelineConfig ToPipelineConfig(const SDKConfig& cfg) const {
        core::PipelineConfig p;
        p.audio_sample_rate       = cfg.audio_sample_rate;
        p.audio_channels          = cfg.audio_channels;
        p.audio_frame_size        = cfg.audio_frame_size;
        p.audio_hop_size          = cfg.audio_hop_size;
        p.target_fps              = cfg.target_fps;
        p.face_size               = cfg.face_size;
        p.sync_threshold_ms       = cfg.sync_threshold_ms;
        p.max_drift_ms            = cfg.max_drift_ms;
        p.av_match_threshold_ms   = cfg.av_match_threshold_ms;
        p.mel_window_frames       = cfg.mel_window_frames;
        p.mel_context_frames      = cfg.mel_context_frames;
        p.enable_frame_pacing     = cfg.enable_frame_pacing;
        p.opencv_num_threads      = cfg.opencv_num_threads;
        p.audio_raw_queue_size    = cfg.audio_raw_queue_size;
        p.mel_queue_size          = cfg.mel_queue_size;
        p.video_raw_queue_size    = cfg.video_raw_queue_size;
        p.face_queue_size         = cfg.face_queue_size;
        p.infer_queue_size        = cfg.infer_queue_size;
        p.output_queue_size       = cfg.output_queue_size;
        p.pop_timeout_ms          = cfg.pop_timeout_ms;
        p.shutdown_timeout_ms     = cfg.shutdown_timeout_ms;
        return p;
    }
};

// ============================================================================
// 构造与析构
// ============================================================================

DigitalHumanSDK::DigitalHumanSDK()
    : impl_(std::make_unique<Impl>()) {}

DigitalHumanSDK::~DigitalHumanSDK() {
    Stop();
}

// ============================================================================
// 生命周期
// ============================================================================

SDKError DigitalHumanSDK::Init(const SDKConfig& config) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    const SDKState current = impl_->state.load(std::memory_order_acquire);
    if (Impl::IsTerminalState(current)) {
        return impl_->Fail(SDKError::ALREADY_TERMINATED,
                           "Init: SDK is terminal and cannot be reused");
    }
    if (current != SDKState::UNINITIALIZED) {
        return impl_->Fail(SDKError::ALREADY_RUNNING,
                           "Init: SDK is already initialized");
    }

    if (config.audio_sample_rate <= 0 || config.audio_channels <= 0
        || config.audio_frame_size <= 0 || config.audio_hop_size <= 0
        || config.audio_hop_size > config.audio_frame_size
        || config.face_size <= 0 || config.target_fps <= 0.0
        || config.sync_threshold_ms < 0.0 || config.max_drift_ms < 0.0
        || config.max_drift_ms < config.sync_threshold_ms
        || config.av_match_threshold_ms < 0.0
        || config.mel_window_frames <= 0
        || config.mel_context_frames < config.mel_window_frames
        || config.audio_raw_queue_size < 0 || config.mel_queue_size < 0
        || config.video_raw_queue_size < 0 || config.face_queue_size < 0
        || config.infer_queue_size < 0 || config.output_queue_size < 0
        || config.pop_timeout_ms < 0 || config.shutdown_timeout_ms < 0
        || config.file_audio_lead_ms < 0 || config.file_stall_timeout_ms <= 0) {
        return impl_->Fail(SDKError::INVALID_CONFIG,
                           "Init: invalid configuration");
    }

    impl_->config = config;
    impl_->lipsync_model_loaded = false;
    impl_->face_model_loaded = false;
    impl_->gpu_enabled = false;
    impl_->pipeline = std::make_unique<core::Pipeline>();
    if (!impl_->pipeline->Init(impl_->ToPipelineConfig(config))) {
        impl_->pipeline.reset();
        return impl_->Fail(SDKError::UNKNOWN,
                           "Init: Pipeline initialization failed");
    }

    if (!config.lipsync_model_dir.empty()) {
        const SDKError error = impl_->LoadLipSyncModelUnlocked(
            config.lipsync_model_dir);
        if (error != SDKError::OK) {
            return error;
        }
    }
    if (!config.face_model_dir.empty()) {
        const SDKError error = impl_->LoadFaceModelUnlocked(config.face_model_dir);
        if (error != SDKError::OK) {
            return error;
        }
    }
    if (config.enable_gpu) {
        // GPU unavailability is non-fatal during initialization; CPU remains active.
        impl_->EnableGPUUnlocked(true);
    }
    if (config.inference_threads > 0) {
        const SDKError error = impl_->SetInferenceThreadsUnlocked(
            config.inference_threads);
        if (error != SDKError::OK) {
            return error;
        }
    }

    impl_->TransitionStateUnlocked(SDKState::INITIALIZED);
    impl_->SetLastError("");
    return SDKError::OK;
}

SDKError DigitalHumanSDK::Start() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return impl_->StartUnlocked();
}

SDKError DigitalHumanSDK::Stop() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return impl_->StopUnlocked();
}

SDKError DigitalHumanSDK::Pause() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    if (impl_->state.load(std::memory_order_acquire) != SDKState::RUNNING) {
        return impl_->Fail(SDKError::NOT_RUNNING, "Pause: SDK is not running");
    }
    if (impl_->pipeline) {
        impl_->pipeline->Pause();
    }
    impl_->TransitionStateUnlocked(SDKState::PAUSED);
    impl_->SetLastError("");
    return SDKError::OK;
}

SDKError DigitalHumanSDK::Resume() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    if (impl_->state.load(std::memory_order_acquire) != SDKState::PAUSED) {
        return impl_->Fail(SDKError::NOT_RUNNING, "Resume: SDK is not paused");
    }
    if (impl_->pipeline) {
        impl_->pipeline->Resume();
    }
    impl_->TransitionStateUnlocked(SDKState::RUNNING);
    impl_->SetLastError("");
    return SDKError::OK;
}

// ============================================================================
// 模型管理
// ============================================================================

SDKError DigitalHumanSDK::LoadLipSyncModel(const std::string& model_dir) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return impl_->LoadLipSyncModelUnlocked(model_dir);
}

SDKError DigitalHumanSDK::LoadFaceModel(const std::string& model_dir) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return impl_->LoadFaceModelUnlocked(model_dir);
}

SDKError DigitalHumanSDK::EnableGPU(bool enable) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return impl_->EnableGPUUnlocked(enable);
}

SDKError DigitalHumanSDK::SetInferenceThreads(int n) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
    return impl_->SetInferenceThreadsUnlocked(n);
}

// ============================================================================
// 数据输入
// ============================================================================

SDKError DigitalHumanSDK::PushAudio(const std::vector<float>& pcm,
                                    int64_t pts_ms) {
    if (pcm.empty()) {
        impl_->SetLastError("PushAudio: empty PCM data");
        return SDKError::INVALID_INPUT;
    }

    core::Pipeline* pipeline = nullptr;
    {
        std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
        if (impl_->state.load(std::memory_order_acquire) != SDKState::RUNNING
            || !impl_->pipeline) {
            impl_->SetLastError("PushAudio: SDK is not running");
            return SDKError::NOT_RUNNING;
        }
        pipeline = impl_->pipeline.get();
    }

    if (!pipeline->PushAudio(pcm, pts_ms)) {
        impl_->SetLastError("PushAudio: queue rejected the packet");
        return SDKError::UNKNOWN;
    }
    return SDKError::OK;
}

SDKError DigitalHumanSDK::PushVideo(const cv::Mat& frame, int64_t pts_ms) {
    if (frame.empty()) {
        impl_->SetLastError("PushVideo: empty frame");
        return SDKError::INVALID_INPUT;
    }

    core::Pipeline* pipeline = nullptr;
    {
        std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
        if (impl_->state.load(std::memory_order_acquire) != SDKState::RUNNING
            || !impl_->pipeline) {
            impl_->SetLastError("PushVideo: SDK is not running");
            return SDKError::NOT_RUNNING;
        }
        pipeline = impl_->pipeline.get();
    }

    if (!pipeline->PushVideo(frame, pts_ms)) {
        impl_->SetLastError("PushVideo: queue rejected the frame");
        return SDKError::UNKNOWN;
    }
    return SDKError::OK;
}

SDKError DigitalHumanSDK::MarkAudioEOS() {
    core::Pipeline* pipeline = nullptr;
    {
        std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
        pipeline = impl_->pipeline.get();
    }
    if (!pipeline) return SDKError::NOT_INITIALIZED;
    pipeline->MarkAudioEOS();
    return SDKError::OK;
}

SDKError DigitalHumanSDK::MarkVideoEOS() {
    core::Pipeline* pipeline = nullptr;
    {
        std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
        pipeline = impl_->pipeline.get();
    }
    if (!pipeline) return SDKError::NOT_INITIALIZED;
    pipeline->MarkVideoEOS();
    return SDKError::OK;
}

// ============================================================================
// Data output
// ============================================================================

SDKError DigitalHumanSDK::GetOutputFrame(cv::Mat& frame,
                                         int64_t& pts_ms,
                                         int timeout_ms) {
    core::Pipeline* pipeline = nullptr;
    {
        std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
        const auto s = impl_->state.load(std::memory_order_acquire);
        if (s != SDKState::RUNNING && s != SDKState::PAUSED && s != SDKState::STOPPED) {
            impl_->SetLastError("GetOutputFrame: SDK is not running");
            return SDKError::NOT_RUNNING;
        }
        pipeline = impl_->pipeline.get();
    }
    if (!pipeline) return SDKError::NOT_INITIALIZED;

    core::OutputFramePacket pkt;
    if (!pipeline->GetOutputFrame(pkt, timeout_ms)) {
        if (Impl::IsTerminalState(
                impl_->state.load(std::memory_order_acquire))) {
            return SDKError::NOT_RUNNING;
        }
        return SDKError::TIMEOUT;
    }

    if (pkt.header.IsFatal()) {
        impl_->SetLastError("GetOutputFrame: pipeline reported a fatal error");
        return SDKError::UNKNOWN;
    }
    if (pkt.header.IsEOS()) {
        return SDKError::NOT_RUNNING;
    }

    frame = std::move(pkt.payload);
    pts_ms = pkt.header.pts_ms;
    return SDKError::OK;
}

// ============================================================================
// File convenience API
// ============================================================================

SDKError DigitalHumanSDK::ProcessFile(const std::string& audio_path,
                                      const std::string& image_path,
                                      const FrameCallback& callback) {
    if (!callback) {
        impl_->SetLastError("ProcessFile: 回调为空");
        return SDKError::INVALID_INPUT;
    }

    // ---- 1. 加载音频文件 ----
    audio::AudioData audio_data;
    try {
        audio_data = impl_->audio_loader.load(audio_path);
    } catch (const std::exception& e) {
        impl_->SetLastError(std::string("ProcessFile: 音频加载失败: ") + e.what());
        return SDKError::AUDIO_LOAD_FAILED;
    }
    if (audio_data.samples.empty()) {
        impl_->SetLastError("ProcessFile: 音频数据为空");
        return SDKError::AUDIO_LOAD_FAILED;
    }

    // ---- 2. 加载图片 ----
    cv::Mat face_image = impl_->image_loader.loadImageFromFile(image_path);
    if (face_image.empty()) {
        impl_->SetLastError("ProcessFile: 图片加载失败: " + image_path);
        return SDKError::IMAGE_LOAD_FAILED;
    }

    // ---- 3. Validate state; start automatically when needed. ----
    SDKConfig file_config;
    {
        std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
        const auto s = impl_->state.load(std::memory_order_acquire);
        if (s == SDKState::UNINITIALIZED) {
            impl_->SetLastError("ProcessFile: SDK not initialized");
            return SDKError::NOT_INITIALIZED;
        }
        if (Impl::IsTerminalState(s)) {
            impl_->SetLastError("ProcessFile: SDK already terminated");
            return SDKError::ALREADY_TERMINATED;
        }
        file_config = impl_->config;
        if (s != SDKState::RUNNING) {
            const auto err = impl_->StartUnlocked();
            if (err != SDKError::OK) return err;
        }
    }

    // ---- 4. 并行推送音频与视频 ----
    // 两个独立生产者避免单一生产者先填满 Mel 队列、却尚未提交视频的
    // 循环等待。音频生产者通过 PTS 水位限制领先量，既为 Matcher 保留
    // 足够的 Mel 预读，又避免无界地跑在视频前面。
    const int    sample_rate   = file_config.audio_sample_rate;
    const double target_fps    = file_config.target_fps;
    const int    hop_size      = file_config.audio_hop_size;
    const double frame_ms      = 1000.0 / target_fps;
    const int64_t audio_lead_ms = file_config.file_audio_lead_ms;

    const int64_t total_samples = static_cast<int64_t>(audio_data.samples.size());
    const int64_t total_frames  = static_cast<int64_t>(
        std::ceil(static_cast<double>(total_samples) * target_fps
                  / sample_rate));

    std::atomic<bool> cancel_requested{false};
    std::atomic<bool> video_done{false};
    std::atomic<int64_t> latest_video_pts_ms{0};
    std::mutex lead_mtx;
    std::condition_variable lead_cv;

    std::mutex producer_error_mtx;
    SDKError producer_error = SDKError::OK;
    std::string producer_error_message;

    auto record_producer_error = [&](SDKError error,
                                     const std::string& message) {
        {
            std::lock_guard<std::mutex> lk(producer_error_mtx);
            if (producer_error == SDKError::OK) {
                producer_error = error;
                producer_error_message = message;
            }
        }
        cancel_requested.store(true, std::memory_order_release);
        lead_cv.notify_all();
    };

    // 音频生产者：允许有限预读，但不会无限领先视频时间轴。
    std::thread audio_thread([&]() {
        int64_t sample_offset = 0;
        while (sample_offset < total_samples
               && !cancel_requested.load(std::memory_order_acquire)) {
            const int64_t audio_pts_ms = sample_offset * 1000 / sample_rate;
            {
                std::unique_lock<std::mutex> lk(lead_mtx);
                lead_cv.wait(lk, [&]() {
                    return cancel_requested.load(std::memory_order_acquire)
                        || video_done.load(std::memory_order_acquire)
                        || audio_pts_ms <=
                            latest_video_pts_ms.load(std::memory_order_acquire)
                                + audio_lead_ms;
                });
            }
            if (cancel_requested.load(std::memory_order_acquire)) break;

            int64_t remain = total_samples - sample_offset;
            int64_t chunk  = std::min<int64_t>(hop_size, remain);
            std::vector<float> pcm(
                audio_data.samples.begin() + sample_offset,
                audio_data.samples.begin() + sample_offset + chunk);
            auto err = PushAudio(pcm, audio_pts_ms);
            if (err != SDKError::OK) {
                if (!cancel_requested.load(std::memory_order_acquire)) {
                    record_producer_error(err, "ProcessFile: 音频推送失败");
                }
                break;
            }
            sample_offset += chunk;
        }
        MarkAudioEOS();
        lead_cv.notify_all();
    });

    // 视频生产者：与音频独立运行。即使音频受到 Mel 队列反压，视频仍能
    // 到达 face-driven Matcher，从而释放 Mel 队列。
    std::thread video_thread([&]() {
        for (int64_t f = 0;
             f < total_frames
                && !cancel_requested.load(std::memory_order_acquire);
             ++f) {
            int64_t vpts = static_cast<int64_t>(f * frame_ms);
            auto err = PushVideo(face_image, vpts);
            if (err != SDKError::OK) {
                if (!cancel_requested.load(std::memory_order_acquire)) {
                    record_producer_error(err, "ProcessFile: 视频推送失败");
                }
                break;
            }
            latest_video_pts_ms.store(vpts, std::memory_order_release);
            lead_cv.notify_all();
        }
        MarkVideoEOS();
        video_done.store(true, std::memory_order_release);
        lead_cv.notify_all();
    });

    // ---- 5. 主线程拉取到明确 EOS；一次普通超时不代表流水线已排空 ----
    SDKError result = SDKError::OK;
    std::string result_error_message;
    auto last_progress = std::chrono::steady_clock::now();
    while (true) {
        cv::Mat out_frame;
        int64_t out_pts = 0;
        auto err = GetOutputFrame(out_frame, out_pts, 200);
        if (err == SDKError::OK) {
            last_progress = std::chrono::steady_clock::now();
            if (!out_frame.empty()) {
                try {
                    callback(out_frame, out_pts);
                } catch (const std::exception& e) {
                    result_error_message =
                        std::string("ProcessFile: 帧回调异常: ") + e.what();
                    result = SDKError::UNKNOWN;
                    break;
                } catch (...) {
                    result_error_message = "ProcessFile: 帧回调发生未知异常";
                    result = SDKError::UNKNOWN;
                    break;
                }
            }
        } else if (err == SDKError::TIMEOUT) {
            {
                std::lock_guard<std::mutex> lk(producer_error_mtx);
                if (producer_error != SDKError::OK) {
                    result = producer_error;
                    break;
                }
            }

            const auto stalled_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - last_progress).count();
            if (stalled_ms >= file_config.file_stall_timeout_ms) {
                result_error_message = "ProcessFile: 流水线长时间无输出进展";
                result = SDKError::TIMEOUT;
                break;
            }
        } else {
            // OutputFramePacket::EOS 被 GetOutputFrame 映射为 NOT_RUNNING。
            std::lock_guard<std::mutex> lk(producer_error_mtx);
            if (producer_error != SDKError::OK) {
                result = producer_error;
            } else if (err != SDKError::NOT_RUNNING) {
                result = err;
                result_error_message = GetLastError();
            }
            break;
        }
    }

    // Stop 会停止所有有界队列并唤醒可能阻塞在 Push() 的生产者。
    cancel_requested.store(true, std::memory_order_release);
    lead_cv.notify_all();
    Stop();

    if (audio_thread.joinable()) {
        audio_thread.join();
    }
    if (video_thread.joinable()) {
        video_thread.join();
    }

    {
        std::lock_guard<std::mutex> lk(producer_error_mtx);
        if (result == SDKError::OK && producer_error != SDKError::OK) {
            result = producer_error;
        }
        if (result != SDKError::OK && !producer_error_message.empty()) {
            result_error_message = producer_error_message;
        }
    }
    impl_->SetLastError(result == SDKError::OK ? "" : result_error_message);

    return result;
}

// ============================================================================
// 查询
// ============================================================================

SDKState DigitalHumanSDK::GetState() const {
    return impl_->state.load(std::memory_order_acquire);
}

std::string DigitalHumanSDK::GetLastError() const {
    std::lock_guard<std::mutex> lk(impl_->last_error_mtx);
    return impl_->last_error;
}

SDKMetrics DigitalHumanSDK::GetMetrics() const {
    SDKMetrics metrics{};
    core::Pipeline* pipeline = nullptr;
    {
        std::lock_guard<std::mutex> lifecycle_lock(impl_->lifecycle_mutex);
        pipeline = impl_->pipeline.get();
        metrics.lifecycle_transition_count =
            impl_->lifecycle_transition_count.load(std::memory_order_relaxed);
    }
    if (!pipeline) {
        return metrics;
    }

    const auto pipeline_metrics = pipeline->GetMetrics();
    metrics.total_frames_in = pipeline_metrics.total_frames_in;
    metrics.total_frames_out = pipeline_metrics.total_frames_out;
    metrics.frames_dropped = pipeline_metrics.frames_dropped;
    metrics.frames_skipped = pipeline_metrics.frames_skipped;
    metrics.audio_packets_in = pipeline_metrics.audio_packets_in;
    metrics.video_packets_in = pipeline_metrics.video_packets_in;
    metrics.inference_count = pipeline_metrics.inference_count;
    metrics.avg_audio_process_ms = pipeline_metrics.avg_audio_process_ms;
    metrics.avg_video_process_ms = pipeline_metrics.avg_video_process_ms;
    metrics.avg_inference_ms = pipeline_metrics.avg_inference_ms;
    metrics.avg_output_ms = pipeline_metrics.avg_output_ms;
    metrics.actual_fps = pipeline_metrics.actual_fps;
    metrics.drift_ms = pipeline->GetDriftMs();

    metrics.shutdown_attempt_count = pipeline_metrics.shutdown_attempt_count;
    metrics.shutdown_timeout_count = pipeline_metrics.shutdown_timeout_count;
    metrics.last_shutdown_ms = pipeline_metrics.last_shutdown_ms;
    metrics.max_shutdown_ms = pipeline_metrics.max_shutdown_ms;
    metrics.av_match_count = pipeline_metrics.av_match_count;
    metrics.av_match_miss_count = pipeline_metrics.av_match_miss_count;
    metrics.avg_av_match_error_ms = pipeline_metrics.avg_av_match_error_ms;
    metrics.max_av_match_error_ms = pipeline_metrics.max_av_match_error_ms;

    metrics.queue_depths.audio_raw = pipeline_metrics.queue_depths.audio_raw;
    metrics.queue_depths.mel_features = pipeline_metrics.queue_depths.mel_features;
    metrics.queue_depths.video_raw = pipeline_metrics.queue_depths.video_raw;
    metrics.queue_depths.processed_faces =
        pipeline_metrics.queue_depths.processed_faces;
    metrics.queue_depths.inference_tasks =
        pipeline_metrics.queue_depths.inference_tasks;
    metrics.queue_depths.inference_output =
        pipeline_metrics.queue_depths.inference_output;
    metrics.queue_depths.output_frames = pipeline_metrics.queue_depths.output_frames;

    metrics.queue_peak_depths.audio_raw =
        pipeline_metrics.queue_peak_depths.audio_raw;
    metrics.queue_peak_depths.mel_features =
        pipeline_metrics.queue_peak_depths.mel_features;
    metrics.queue_peak_depths.video_raw =
        pipeline_metrics.queue_peak_depths.video_raw;
    metrics.queue_peak_depths.processed_faces =
        pipeline_metrics.queue_peak_depths.processed_faces;
    metrics.queue_peak_depths.inference_tasks =
        pipeline_metrics.queue_peak_depths.inference_tasks;
    metrics.queue_peak_depths.inference_output =
        pipeline_metrics.queue_peak_depths.inference_output;
    metrics.queue_peak_depths.output_frames =
        pipeline_metrics.queue_peak_depths.output_frames;
    return metrics;
}

}  // namespace digital_human
