// ============================================================================
// Digital Human SDK — 统一入口实现
//
// 封装 Pipeline + ModelInferencer + AudioLoader + ImageLoader，
// 提供简化生命周期与文件级便捷接口。
// ============================================================================

#include "digital_human_sdk.h"

#include <algorithm>
#include <atomic>
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
    std::atomic<bool>     terminated{false};

    // ---- 最后错误 ----
    // 简单互斥保护即可，错误路径非热点
    mutable std::mutex   last_error_mtx;
    std::string          last_error;

    void SetLastError(const std::string& msg) {
        std::lock_guard<std::mutex> lk(last_error_mtx);
        last_error = msg;
    }

    // ------------------------------------------------------------------
    // 配置转换：SDKConfig → PipelineConfig
    // ------------------------------------------------------------------
    core::PipelineConfig ToPipelineConfig(const SDKConfig& cfg) const {
        core::PipelineConfig p;
        p.audio_sample_rate       = cfg.audio_sample_rate;
        p.audio_frame_size        = cfg.audio_frame_size;
        p.audio_hop_size          = cfg.audio_hop_size;
        p.target_fps              = cfg.target_fps;
        p.face_size               = cfg.face_size;
        p.sync_threshold_ms       = cfg.sync_threshold_ms;
        p.max_drift_ms            = cfg.max_drift_ms;
        p.mel_window_frames       = 16;
        p.mel_context_frames      = 300;
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
    if (impl_->state.load() != SDKState::UNINITIALIZED) {
        impl_->SetLastError("Init: SDK 已初始化");
        return SDKError::ALREADY_RUNNING;
    }
    if (impl_->terminated.load()) {
        impl_->SetLastError("Init: SDK 已终止，不可重用");
        return SDKError::ALREADY_TERMINATED;
    }

    // 参数校验
    if (config.audio_sample_rate <= 0) {
        impl_->SetLastError("Init: 无效采样率");
        return SDKError::INVALID_CONFIG;
    }
    if (config.target_fps <= 0) {
        impl_->SetLastError("Init: 无效帧率");
        return SDKError::INVALID_CONFIG;
    }
    if (config.audio_frame_size <= 0 || config.audio_hop_size <= 0) {
        impl_->SetLastError("Init: 无效音频帧/帧移参数");
        return SDKError::INVALID_CONFIG;
    }

    impl_->config = config;

    // 创建并初始化 Pipeline
    impl_->pipeline = std::make_unique<core::Pipeline>();
    auto pipe_cfg = impl_->ToPipelineConfig(config);
    if (!impl_->pipeline->Init(pipe_cfg)) {
        impl_->pipeline.reset();
        impl_->SetLastError("Init: Pipeline 初始化失败");
        return SDKError::UNKNOWN;
    }

    // 加载模型（路径非空时自动加载）
    if (!config.lipsync_model_dir.empty()) {
        auto err = LoadLipSyncModel(config.lipsync_model_dir);
        if (err != SDKError::OK) return err;
    }
    if (!config.face_model_dir.empty()) {
        auto err = LoadFaceModel(config.face_model_dir);
        if (err != SDKError::OK) return err;
    }

    // GPU 开关
    if (config.enable_gpu) {
        auto err = EnableGPU(true);
        if (err != SDKError::OK) {
            // GPU 不可用不致命，回退 CPU 继续
        }
    }

    // 推理线程数
    if (config.inference_threads > 0) {
        impl_->pipeline->SetInferenceThreads(config.inference_threads);
    }

    impl_->state.store(SDKState::INITIALIZED, std::memory_order_release);
    impl_->SetLastError("");
    return SDKError::OK;
}

SDKError DigitalHumanSDK::Start() {
    auto s = impl_->state.load();
    if (s == SDKState::UNINITIALIZED) {
        impl_->SetLastError("Start: 未初始化");
        return SDKError::NOT_INITIALIZED;
    }
    if (s == SDKState::RUNNING) {
        return SDKError::OK;
    }
    if (impl_->terminated.load()) {
        impl_->SetLastError("Start: SDK 已终止，拒绝重启");
        return SDKError::ALREADY_TERMINATED;
    }

    if (!impl_->pipeline->Start()) {
        impl_->SetLastError("Start: Pipeline 启动失败");
        return SDKError::PIPELINE_START_FAILED;
    }

    impl_->state.store(SDKState::RUNNING, std::memory_order_release);
    impl_->SetLastError("");
    return SDKError::OK;
}

SDKError DigitalHumanSDK::Stop() {
    auto s = impl_->state.load();
    if (s == SDKState::UNINITIALIZED || s == SDKState::STOPPED) {
        return SDKError::OK;
    }

    if (impl_->pipeline) {
        impl_->pipeline->Stop();
    }

    impl_->terminated.store(true, std::memory_order_release);
    impl_->state.store(SDKState::STOPPED, std::memory_order_release);
    impl_->SetLastError("");
    return SDKError::OK;
}

SDKError DigitalHumanSDK::Pause() {
    if (impl_->state.load() != SDKState::RUNNING) {
        impl_->SetLastError("Pause: 未运行");
        return SDKError::NOT_RUNNING;
    }
    if (impl_->pipeline) {
        impl_->pipeline->Pause();
    }
    impl_->state.store(SDKState::PAUSED, std::memory_order_release);
    return SDKError::OK;
}

SDKError DigitalHumanSDK::Resume() {
    if (impl_->state.load() != SDKState::PAUSED) {
        impl_->SetLastError("Resume: 未处于暂停状态");
        return SDKError::NOT_RUNNING;
    }
    if (impl_->pipeline) {
        impl_->pipeline->Resume();
    }
    impl_->state.store(SDKState::RUNNING, std::memory_order_release);
    return SDKError::OK;
}

// ============================================================================
// 模型管理
// ============================================================================

SDKError DigitalHumanSDK::LoadLipSyncModel(const std::string& model_dir) {
    if (!impl_->pipeline) {
        impl_->SetLastError("LoadLipSyncModel: Pipeline 未创建");
        return SDKError::NOT_INITIALIZED;
    }
    if (!impl_->pipeline->InitModelInferencer(model_dir)) {
        impl_->SetLastError("LoadLipSyncModel: 加载失败: " + model_dir);
        return SDKError::MODEL_LOAD_FAILED;
    }
    impl_->lipsync_model_loaded = true;
    return SDKError::OK;
}

SDKError DigitalHumanSDK::LoadFaceModel(const std::string& model_dir) {
    if (!impl_->pipeline) {
        impl_->SetLastError("LoadFaceModel: Pipeline 未创建");
        return SDKError::NOT_INITIALIZED;
    }
    impl_->pipeline->SetLandmarkModelPath(model_dir);
    // SetLandmarkModelPath 仅记录路径，实际加载在 VideoProcessor 启动时
    // 这里无法立即验证，标记为已配置
    impl_->face_model_loaded = true;
    return SDKError::OK;
}

SDKError DigitalHumanSDK::EnableGPU(bool enable) {
    if (!impl_->pipeline) {
        impl_->SetLastError("EnableGPU: Pipeline 未创建");
        return SDKError::NOT_INITIALIZED;
    }
    if (!impl_->pipeline->EnableGPU(enable)) {
        impl_->SetLastError(enable
            ? "EnableGPU: Vulkan 不可用或模型加载失败"
            : "EnableGPU: 关闭 GPU 失败");
        return enable ? SDKError::GPU_NOT_AVAILABLE : SDKError::UNKNOWN;
    }
    impl_->gpu_enabled = enable;
    return SDKError::OK;
}

SDKError DigitalHumanSDK::SetInferenceThreads(int n) {
    if (!impl_->pipeline) {
        impl_->SetLastError("SetInferenceThreads: Pipeline 未创建");
        return SDKError::NOT_INITIALIZED;
    }
    if (n < 0) {
        impl_->SetLastError("SetInferenceThreads: 线程数不能为负");
        return SDKError::INVALID_CONFIG;
    }
    impl_->pipeline->SetInferenceThreads(n);
    return SDKError::OK;
}

// ============================================================================
// 数据输入
// ============================================================================

SDKError DigitalHumanSDK::PushAudio(const std::vector<float>& pcm,
                                    int64_t pts_ms) {
    if (impl_->state.load() != SDKState::RUNNING) {
        impl_->SetLastError("PushAudio: 未运行");
        return SDKError::NOT_RUNNING;
    }
    if (pcm.empty()) {
        impl_->SetLastError("PushAudio: 空 PCM 数据");
        return SDKError::INVALID_INPUT;
    }
    if (!impl_->pipeline->PushAudio(pcm, pts_ms)) {
        impl_->SetLastError("PushAudio: 入队失败（队列已停止或满）");
        return SDKError::UNKNOWN;
    }
    return SDKError::OK;
}

SDKError DigitalHumanSDK::PushVideo(const cv::Mat& frame, int64_t pts_ms) {
    if (impl_->state.load() != SDKState::RUNNING) {
        impl_->SetLastError("PushVideo: 未运行");
        return SDKError::NOT_RUNNING;
    }
    if (frame.empty()) {
        impl_->SetLastError("PushVideo: 空图像");
        return SDKError::INVALID_INPUT;
    }
    if (!impl_->pipeline->PushVideo(frame, pts_ms)) {
        impl_->SetLastError("PushVideo: 入队失败（队列已停止或满）");
        return SDKError::UNKNOWN;
    }
    return SDKError::OK;
}

SDKError DigitalHumanSDK::MarkAudioEOS() {
    if (!impl_->pipeline) return SDKError::NOT_INITIALIZED;
    impl_->pipeline->MarkAudioEOS();
    return SDKError::OK;
}

SDKError DigitalHumanSDK::MarkVideoEOS() {
    if (!impl_->pipeline) return SDKError::NOT_INITIALIZED;
    impl_->pipeline->MarkVideoEOS();
    return SDKError::OK;
}

// ============================================================================
// 数据输出
// ============================================================================

SDKError DigitalHumanSDK::GetOutputFrame(cv::Mat& frame,
                                         int64_t& pts_ms,
                                         int timeout_ms) {
    auto s = impl_->state.load();
    if (s != SDKState::RUNNING && s != SDKState::PAUSED) {
        // 流水线已停止：仍允许排空残留帧
        if (s != SDKState::STOPPED) {
            impl_->SetLastError("GetOutputFrame: 未运行");
            return SDKError::NOT_RUNNING;
        }
    }

    core::OutputFramePacket pkt;
    if (!impl_->pipeline->GetOutputFrame(pkt, timeout_ms)) {
        // 区分超时与流结束
        if (impl_->terminated.load()) {
            return SDKError::NOT_RUNNING;
        }
        return SDKError::TIMEOUT;
    }

    if (pkt.header.IsEOS() || pkt.header.IsFatal()) {
        return SDKError::NOT_RUNNING;
    }

    frame  = std::move(pkt.payload);
    pts_ms = pkt.header.pts_ms;
    return SDKError::OK;
}

// ============================================================================
// 文件便捷接口
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

    // ---- 3. 校验状态：必须已 Init，若未 Start 则自动启动 ----
    auto s = impl_->state.load();
    if (s == SDKState::UNINITIALIZED) {
        impl_->SetLastError("ProcessFile: SDK 未初始化");
        return SDKError::NOT_INITIALIZED;
    }
    if (impl_->terminated.load()) {
        impl_->SetLastError("ProcessFile: SDK 已终止");
        return SDKError::ALREADY_TERMINATED;
    }
    if (s != SDKState::RUNNING) {
        auto err = Start();
        if (err != SDKError::OK) return err;
    }

    // ---- 4. 按目标帧率推送数据 ----
    // 音频按 hop_size 分块推送，PTS 按样本数累计
    // 视频按 target_fps 重复推送同一张图片，PTS 按帧序号累计
    const int    sample_rate   = impl_->config.audio_sample_rate;
    const double target_fps    = impl_->config.target_fps;
    const int    hop_size      = impl_->config.audio_hop_size;
    const double frame_ms      = 1000.0 / target_fps;
    const double hop_ms        = 1000.0 * hop_size / sample_rate;

    const int64_t total_samples = static_cast<int64_t>(audio_data.samples.size());
    const int64_t total_frames  = static_cast<int64_t>(
        total_samples / sample_rate * target_fps);

    // 启动一个推送线程 + 主线程拉取输出，避免单线程死锁
    // （Pipeline 队列有界，若推送过快会阻塞，需要同时消费）
    std::atomic<bool> push_done{false};
    std::thread push_thread([&]() {
        // 推送音频
        int64_t sample_offset = 0;
        while (sample_offset < total_samples) {
            int64_t remain = total_samples - sample_offset;
            int64_t chunk  = std::min<int64_t>(hop_size, remain);
            std::vector<float> pcm(
                audio_data.samples.begin() + sample_offset,
                audio_data.samples.begin() + sample_offset + chunk);
            double pts_ms = static_cast<double>(sample_offset)
                          / sample_rate * 1000.0;
            if (PushAudio(pcm, static_cast<int64_t>(pts_ms)) != SDKError::OK) {
                break;
            }
            sample_offset += chunk;
        }
        MarkAudioEOS();

        // 推送视频帧（覆盖音频总时长）
        for (int64_t f = 0; f < total_frames; ++f) {
            int64_t vpts = static_cast<int64_t>(f * frame_ms);
            if (PushVideo(face_image, vpts) != SDKError::OK) {
                break;
            }
        }
        MarkVideoEOS();
        push_done.store(true, std::memory_order_release);
    });

    // 主线程拉取输出帧
    SDKError result = SDKError::OK;
    while (true) {
        cv::Mat out_frame;
        int64_t out_pts = 0;
        auto err = GetOutputFrame(out_frame, out_pts, 200);
        if (err == SDKError::OK) {
            if (!out_frame.empty()) {
                callback(out_frame, out_pts);
            }
        } else if (err == SDKError::TIMEOUT) {
            // 推送完成且拉取超时 → 尝试排空后结束
            if (push_done.load(std::memory_order_acquire)) {
                cv::Mat last;
                int64_t last_pts = 0;
                if (GetOutputFrame(last, last_pts, 100) == SDKError::OK
                    && !last.empty()) {
                    callback(last, last_pts);
                }
                break;
            }
            // 推送未完成，继续等待
        } else {
            // NOT_RUNNING：可能是 EOS 正常结束，也可能是错误停止
            // 若推送已完成，视为正常结束；否则记录错误
            if (push_done.load(std::memory_order_acquire)) {
                break;
            }
            result = err;
            break;
        }
    }

    if (push_thread.joinable()) {
        push_thread.join();
    }

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
    SDKMetrics m{};
    if (!impl_->pipeline) return m;

    auto pm = impl_->pipeline->GetMetrics();
    m.total_frames_in      = pm.total_frames_in;
    m.total_frames_out     = pm.total_frames_out;
    m.frames_dropped       = pm.frames_dropped;
    m.frames_skipped       = pm.frames_skipped;
    m.audio_packets_in     = pm.audio_packets_in;
    m.video_packets_in     = pm.video_packets_in;
    m.inference_count      = pm.inference_count;
    m.avg_audio_process_ms = pm.avg_audio_process_ms;
    m.avg_video_process_ms = pm.avg_video_process_ms;
    m.avg_inference_ms     = pm.avg_inference_ms;
    m.avg_output_ms        = pm.avg_output_ms;
    m.actual_fps           = pm.actual_fps;
    m.drift_ms             = impl_->pipeline->GetDriftMs();
    return m;
}

}  // namespace digital_human
