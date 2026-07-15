#include "core/pipeline.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include "audio/audio_framer.h"
#include "audio/audio_vad.h"
#include "audio/audio_preemphasis.h"
#include "audio/audio_rms_normalize.h"
#include "audio/audio_noise_reduction.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_cmvn.h"
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"
#include "model/model_inferencer.h"
#include "model/output_processor.h"

namespace digital_human {
namespace core {

using audio::AudioFramer;
using audio::VoiceActivityDetector;
using audio::PreEmphasis;
using audio::RMSNormalize;
using audio::NoiseReduction;
using audio::MelFeatureExtract;
using audio::CMVN;

// ============================================================================
// PipelineMetrics 实现
// ============================================================================

std::string PipelineMetrics::ToString() const {
    std::ostringstream oss;
    oss << "PipelineMetrics {"
        << " frames_in=" << total_frames_in
        << " frames_out=" << total_frames_out
        << " dropped=" << frames_dropped
        << " skipped=" << frames_skipped
        << " audio_in=" << audio_packets_in
        << " video_in=" << video_packets_in
        << " inference=" << inference_count
        << " fps=" << actual_fps
        << " }";
    return oss.str();
}

// ============================================================================
// Pipeline::Impl — 内部实现
// ============================================================================

struct Pipeline::Impl {
    // ---- 配置 ----
    PipelineConfig config;
    std::atomic<bool> initialized{false};
    std::atomic<bool> running{false};
    std::atomic<bool> paused{false};

    // ---- 队列 ----
    ThreadSafeQueue<AudioRawPacket>      audio_raw_queue;
    ThreadSafeQueue<MelFeaturePacket>    mel_feature_queue;
    ThreadSafeQueue<VideoFramePacket>    video_raw_queue;
    ThreadSafeQueue<ProcessedFacePacket> processed_face_queue;
    ThreadSafeQueue<InferenceOutputPacket> inference_output_queue;
    ThreadSafeQueue<OutputFramePacket>   output_frame_queue;

    // ---- 同步模块 ----
    AVSync         av_sync;
    FrameScheduler frame_scheduler;

    // ---- 音频处理模块（每个线程拥有独立实例） ----
    NoiseReduction      noise_reduction;
    AudioFramer         audio_framer;
    VoiceActivityDetector vad;
    PreEmphasis         pre_emphasis;
    RMSNormalize        rms_normalize;
    MelFeatureExtract   mel_extract;
    CMVN                cmvn;

    // ---- 视频处理模块（每个线程拥有独立实例） ----
    DigitalHuman::core::FaceDetector     face_detector;
    digital_human::core::FaceAlignigner  face_aligner;
    DigitalHuman::Core::FaceMaskGenerator face_mask_gen;

    // ---- 推理和后处理模块 ----
    model::ModelInferencer  model_inferencer;
    model::OutputProcessor  output_processor;

    // ---- 工作线程 ----
    std::unique_ptr<ThreadBase> audio_producer_thread;
    std::unique_ptr<ThreadBase> audio_processor_thread;
    std::unique_ptr<ThreadBase> video_producer_thread;
    std::unique_ptr<ThreadBase> video_processor_thread;
    std::unique_ptr<ThreadBase> inference_thread;
    std::unique_ptr<ThreadBase> output_thread;
    std::unique_ptr<ThreadBase> render_thread;

    // ---- 指标统计 ----
    std::atomic<int64_t> total_frames_in{0};
    std::atomic<int64_t> total_frames_out{0};
    std::atomic<int64_t> frames_dropped{0};
    std::atomic<int64_t> frames_skipped{0};
    std::atomic<int64_t> audio_packets_in{0};
    std::atomic<int64_t> video_packets_in{0};
    std::atomic<int64_t> inference_count{0};

    // ---- 性能计数 ----
    // 使用微秒累积（int64_t），解决 C++17 中 atomic<double> 无 fetch_add 的问题
    std::atomic<int64_t> total_audio_process_us{0};
    std::atomic<int64_t> audio_process_count{0};
    std::atomic<int64_t> total_video_process_us{0};
    std::atomic<int64_t> video_process_count{0};
    std::atomic<int64_t> total_inference_us{0};
    std::atomic<int64_t> inference_count_val{0};
    std::atomic<int64_t> total_output_us{0};
    std::atomic<int64_t> output_count{0};

    // ---- 输入标记 ----
    std::atomic<bool> audio_eos{false};
    std::atomic<bool> video_eos{false};

    // ---- 启动时间 ----
    std::chrono::steady_clock::time_point start_time;

    // ========================================================================
    // 构造函数：初始化队列
    // ========================================================================

    Impl()
        : audio_raw_queue(0)           // 无界
        , mel_feature_queue(60)
        , video_raw_queue(0)           // 无界
        , processed_face_queue(30)
        , inference_output_queue(30)
        , output_frame_queue(10)
    {}

    // ========================================================================
    // 初始化核心模块
    // ========================================================================

    bool InitCoreModules() {
        // 设置 AVSync
        SyncConfig sync_cfg;
        sync_cfg.audio_sample_rate = config.audio_sample_rate;
        sync_cfg.sync_threshold_ms = config.sync_threshold_ms;
        sync_cfg.max_drift_ms      = config.max_drift_ms;
        av_sync.Init(sync_cfg);

        // 设置 FrameScheduler
        SchedulerConfig sched_cfg;
        sched_cfg.target_fps       = config.target_fps;
        sched_cfg.smoothing_factor = 0.5;
        sched_cfg.enable_smoothing = true;
        frame_scheduler.Init(sched_cfg);

        return true;
    }

    // ========================================================================
    // 配置队列容量
    // ========================================================================

    void ConfigureQueues() {
        // 动态调整队列容量（如果有界）
        // 已在构造函数中设置默认值
    }

    // ========================================================================
    // 创建工作线程
    // ========================================================================

    void CreateWorkers() {
        // ---- AudioProducer ----
        audio_producer_thread = std::make_unique<AudioProducerThread>(*this);

        // ---- AudioProcessor ----
        audio_processor_thread = std::make_unique<AudioProcessorThread>(*this);

        // ---- VideoProducer ----
        video_producer_thread = std::make_unique<VideoProducerThread>(*this);

        // ---- VideoProcessor ----
        video_processor_thread = std::make_unique<VideoProcessorThread>(*this);

        // ---- InferenceWorker ----
        inference_thread = std::make_unique<InferenceThread>(*this);

        // ---- OutputProcessor ----
        output_thread = std::make_unique<OutputThread>(*this);

        // ---- RenderThread ----
        render_thread = std::make_unique<RenderThread>(*this);
    }

    // ========================================================================
    // 工作线程实现（内嵌类）
    // ========================================================================

    // ---------------------------------------------------------------
    // AudioProducer: 从外部 PushAudio 读取数据，转发到 audio_raw_queue
    // 实际由外部调用 PushAudio 驱动，本线程仅管理 EOS 信号
    // ---------------------------------------------------------------
    struct AudioProducerThread : public ThreadBase {
        Impl& ctx;
        AudioProducerThread(Impl& ctx)
            : ThreadBase("AudioProducer"), ctx(ctx) {}
        void Run() override {
            LogInfo("启动");
            // 等待外部 PushAudio 数据，直到 EOS
            while (!IsStopping()) {
                if (ctx.audio_eos.load(std::memory_order_acquire)) {
                    ctx.audio_raw_queue.Push(AudioRawPacket::EOS());
                    LogInfo("音频流结束");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    };

    // ---------------------------------------------------------------
    // AudioProcessor: 音频特征提取
    // PCM → NoiseReduction → AudioFramer → VAD → PreEmphasis
    //   → RMSNormalize → MelFeatureExtract → CMVN
    // ---------------------------------------------------------------
    struct AudioProcessorThread : public ThreadBase {
        Impl& ctx;
        AudioProcessorThread(Impl& ctx)
            : ThreadBase("AudioProcessor"), ctx(ctx) {}

        void Run() override {
            LogInfo("启动");
            while (!IsStopping()) {
                AudioRawPacket pkt;
                if (!ctx.audio_raw_queue.WaitAndPop(pkt, 100)) {
                    continue;
                }

                if (pkt.header.IsEOS()) {
                    ctx.mel_feature_queue.Push(MelFeaturePacket::EOS());
                    break;
                }
                if (pkt.header.IsFatal() || pkt.header.IsSkip()) {
                    continue;
                }

                auto start = std::chrono::steady_clock::now();

                auto result = ProcessOne(pkt);

                auto elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                ctx.total_audio_process_us.fetch_add(
                    static_cast<int64_t>(elapsed * 1000.0),
                    std::memory_order_relaxed);
                ctx.audio_process_count.fetch_add(1, std::memory_order_relaxed);

                if (result.header.IsOK()) {
                    ctx.mel_feature_queue.Push(std::move(result));
                }
                // SKIP 或 ERROR 包直接丢弃
            }
            LogInfo("退出");
        }

        MelFeaturePacket ProcessOne(const AudioRawPacket& pkt) {
            MelFeaturePacket result;
            result.InheritHeader(pkt.header);

            const auto& pcm = pkt.payload;
            int sr = ctx.config.audio_sample_rate;

            try {
                // 1. 降噪
                auto denoised = ctx.noise_reduction.process(pcm, sr);

                // 2. RMS 归一化
                auto normalized = ctx.rms_normalize.process(denoised);

                // 3. 预加重
                auto emphasized = ctx.pre_emphasis.process(normalized);

                // 4. 分帧
                audio::FrameConfig frame_cfg;
                frame_cfg.frameSize = ctx.config.audio_frame_size;
                frame_cfg.hopSize   = ctx.config.audio_hop_size;
                auto frames = ctx.audio_framer.frame(emphasized, frame_cfg);

                if (frames.empty() || frames[0].empty()) {
                    result.header.status = StatusCode::SKIP;
                    return result;
                }

                // 5. VAD 过滤
                auto voiced = ctx.vad.filter(frames);
                if (voiced.empty()) {
                    // 无语音活动，仍然继续处理避免丢失上下文
                }

                // 6. 提取 Mel 频谱
                audio::MelConfig mel_cfg;
                mel_cfg.sampleRate = sr;
                auto mel = ctx.mel_extract.extract(
                    voiced.empty() ? frames : voiced, mel_cfg);

                if (mel.empty()) {
                    result.header.status = StatusCode::SKIP;
                    return result;
                }

                // 7. CMVN 归一化
                result.payload = ctx.cmvn.process(mel);
                result.header.status = StatusCode::OK;

            } catch (const std::exception& e) {
                LogError(std::string("音频处理异常: ") + e.what());
                result.header.status = StatusCode::ERROR;
            }

            return result;
        }
    };

    // ---------------------------------------------------------------
    // VideoProducer: 从外部 PushVideo 读取数据，转发到 video_raw_queue
    // ---------------------------------------------------------------
    struct VideoProducerThread : public ThreadBase {
        Impl& ctx;
        VideoProducerThread(Impl& ctx)
            : ThreadBase("VideoProducer"), ctx(ctx) {}
        void Run() override {
            LogInfo("启动");
            while (!IsStopping()) {
                if (ctx.video_eos.load(std::memory_order_acquire)) {
                    ctx.video_raw_queue.Push(VideoFramePacket::EOS());
                    LogInfo("视频流结束");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    };

    // ---------------------------------------------------------------
    // VideoProcessor: 人脸检测 + 对齐 + 遮罩
    // cv::Mat → FaceDetect → FaceAlign → FaceMask
    // ---------------------------------------------------------------
    struct VideoProcessorThread : public ThreadBase {
        Impl& ctx;
        VideoProcessorThread(Impl& ctx)
            : ThreadBase("VideoProcessor"), ctx(ctx) {}

        void Run() override {
            LogInfo("启动");
            while (!IsStopping()) {
                VideoFramePacket pkt;
                if (!ctx.video_raw_queue.WaitAndPop(pkt, 100)) {
                    continue;
                }

                if (pkt.header.IsEOS()) {
                    ctx.processed_face_queue.Push(ProcessedFacePacket::EOS());
                    break;
                }
                if (pkt.header.IsFatal() || pkt.header.IsSkip()) {
                    continue;
                }

                auto start = std::chrono::steady_clock::now();
                auto result = ProcessOne(pkt);
                auto elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                ctx.total_video_process_us.fetch_add(
                    static_cast<int64_t>(elapsed * 1000.0),
                    std::memory_order_relaxed);
                ctx.video_process_count.fetch_add(1, std::memory_order_relaxed);

                if (result.header.IsOK()) {
                    ctx.processed_face_queue.Push(std::move(result));
                }
            }
            LogInfo("退出");
        }

        ProcessedFacePacket ProcessOne(const VideoFramePacket& pkt) {
            ProcessedFacePacket result;
            result.InheritHeader(pkt.header);

            try {
                const cv::Mat& frame = pkt.payload;

                // 1. 人脸检测
                auto faces = ctx.face_detector.detect(frame);
                if (faces.empty()) {
                    LogInfo("未检测到人脸");
                    result.header.status = StatusCode::SKIP;
                    return result;
                }

                // 取最大的人脸
                auto max_face = std::max_element(faces.begin(), faces.end(),
                    [](const cv::Rect& a, const cv::Rect& b) {
                        return a.area() < b.area();
                    });

                // 2. 获取关键点
                auto landmarks = ctx.face_detector.getLandmarks(frame, *max_face);
                if (landmarks.empty()) {
                    result.header.status = StatusCode::SKIP;
                    return result;
                }

                // 3. 人脸对齐
                std::vector<cv::Point2f> landmarks_f;
                for (const auto& pt : landmarks) {
                    landmarks_f.emplace_back(
                        static_cast<float>(pt.x),
                        static_cast<float>(pt.y));
                }

                auto align_result = ctx.face_aligner.alignByRect(
                    frame, landmarks_f,
                    ctx.config.face_size, *max_face);

                if (!align_result.valid) {
                    result.header.status = StatusCode::SKIP;
                    return result;
                }

                // 4. 生成口唇遮罩
                auto mouth_mask = ctx.face_mask_gen.generateMouthMask(
                    frame.size(), landmarks);

                // 5. 96x96 空间的精细遮罩
                auto precise_mask = ctx.face_mask_gen.generatePreciseMouthAlphaMask96(
                    align_result.landmarks);

                // 填充结果
                result.payload.aligned_face  = align_result.aligned_face;
                result.payload.M_inv         = align_result.M_inv;
                result.payload.face_mask     = precise_mask.empty() ? mouth_mask : precise_mask;
                result.payload.original_face = frame.clone();
                result.payload.face_rect     = align_result.face_rect;
                result.payload.landmarks_96  = align_result.landmarks;
                result.header.status         = StatusCode::OK;

            } catch (const std::exception& e) {
                LogError(std::string("视频处理异常: ") + e.what());
                result.header.status = StatusCode::ERROR;
            }

            return result;
        }
    };

    // ---------------------------------------------------------------
    // InferenceThread: 模型推理
    // 等待 Mel 特征 + 处理后人脸 → 推理 → 输出
    // ---------------------------------------------------------------
    struct InferenceThread : public ThreadBase {
        Impl& ctx;
        InferenceThread(Impl& ctx)
            : ThreadBase("InferenceWorker"), ctx(ctx) {}

        void Run() override {
            LogInfo("启动");
            while (!IsStopping()) {
                // 1. 取 Mel 特征
                MelFeaturePacket mel_pkt;
                if (!ctx.mel_feature_queue.WaitAndPop(mel_pkt, 100)) {
                    continue;
                }
                if (mel_pkt.header.IsEOS()) {
                    ctx.inference_output_queue.Push(
                        InferenceOutputPacket::EOS());
                    break;
                }
                if (mel_pkt.header.IsFatal()) {
                    ctx.inference_output_queue.Push(
                        InferenceOutputPacket::Fatal());
                    break;
                }
                if (mel_pkt.header.IsSkip()) {
                    continue;
                }

                // 2. 取处理后的人脸（时间戳匹配）
                ProcessedFacePacket face_pkt;
                if (!MatchFacePacket(mel_pkt, face_pkt)) {
                    // 匹配失败，丢弃当前 mel
                    continue;
                }
                if (face_pkt.header.IsSkip()) {
                    continue;
                }

                // 3. 准备推理输入
                if (!ctx.model_inferencer.IsInitialized()) {
                    LogError("模型未初始化");
                    continue;
                }

                auto start = std::chrono::steady_clock::now();

                // 将 cv::Mat 转为 ncnn::Mat
                // 注意: Wav2Lip 的输入格式需要匹配模型要求
                cv::Mat face_input;
                if (face_pkt.payload.aligned_face.channels() == 3) {
                    // 单帧: 复制一份作为第 4-6 通道（对口型模型要求 6 通道输入）
                    cv::Mat channels[6];
                    cv::split(face_pkt.payload.aligned_face, channels);
                    // 复制前 3 通道作为后 3 通道
                    channels[3] = channels[0].clone();
                    channels[4] = channels[1].clone();
                    channels[5] = channels[2].clone();
                    cv::merge(channels, 6, face_input);
                } else {
                    face_input = face_pkt.payload.aligned_face.clone();
                }

                // 音频 mel 特征 (cv::Mat) → ncnn::Mat
                // mel.shape = (mel_bins, T, 1)
                cv::Mat mel_mat = mel_pkt.payload;
                if (mel_mat.empty()) continue;

                // 推理
                ncnn::Mat audio_feat_ncnn(mel_mat.cols, mel_mat.rows, 1,
                    (void*)mel_mat.data);
                ncnn::Mat face_input_ncnn(face_input.cols, face_input.rows,
                    face_input.channels(), (void*)face_input.data);

                ncnn::Mat model_output = ctx.model_inferencer.Infer(
                    audio_feat_ncnn, face_input_ncnn);

                auto elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();

                ctx.total_inference_us.fetch_add(
                    static_cast<int64_t>(elapsed * 1000.0),
                    std::memory_order_relaxed);
                ctx.inference_count_val.fetch_add(1, std::memory_order_relaxed);
                ctx.inference_count.fetch_add(1, std::memory_order_relaxed);

                if (model_output.empty()) {
                    continue;
                }

                // 4. 打包推理结果
                InferenceOutputPacket out_pkt;
                out_pkt.InheritHeader(mel_pkt.header);
                out_pkt.payload = model_output;
                // 附加人脸处理数据供下游使用
                // （通过静态变量传递 - 生产环境应使用更健壮的方式）
                ctx.inference_output_queue.Push(std::move(out_pkt));
            }
            LogInfo("退出");
        }

        /// @brief 时间戳匹配：寻找与 mel_pkt 时间戳匹配的人脸包
        bool MatchFacePacket(const MelFeaturePacket& mel_pkt,
                            ProcessedFacePacket& face_pkt) {
            int64_t target_pts = mel_pkt.header.pts_ms;
            double threshold   = ctx.config.av_match_threshold_ms;
            int max_attempts   = 10;

            for (int i = 0; i < max_attempts; ++i) {
                if (!ctx.processed_face_queue.WaitAndPop(face_pkt, 100)) {
                    return false;
                }

                if (face_pkt.header.IsEOS()) {
                    return false;
                }
                if (face_pkt.header.IsFatal()) {
                    return false;
                }
                if (face_pkt.header.IsSkip()) {
                    continue;  // 跳过处理失败的帧
                }

                double drift = std::abs(
                    static_cast<double>(face_pkt.header.pts_ms - target_pts));

                if (drift <= threshold) {
                    return true;  // 匹配成功
                }

                // 视频超前于音频 → 丢弃人脸帧，取下一帧
                // 视频滞后于音频 → 丢弃音频帧（由调用方处理）
                if (face_pkt.header.pts_ms > target_pts) {
                    // 视频超前，丢弃当前 face，继续取
                    ctx.frames_skipped.fetch_add(1, std::memory_order_relaxed);
                    continue;
                } else {
                    // 视频滞后，把 face 放回队列（让调用方丢弃 mel）
                    // 但简化实现：直接返回 SKIP
                    ctx.frames_skipped.fetch_add(1, std::memory_order_relaxed);
                    face_pkt.header.status = StatusCode::SKIP;
                    return true;
                }
            }

            return false;
        }
    };

    // ---------------------------------------------------------------
    // OutputThread: 推理输出后处理
    // ncnn::Mat → OutputToMat → InverseTransform → FaceFusion → PostProcess
    // ---------------------------------------------------------------
    struct OutputThread : public ThreadBase {
        Impl& ctx;
        OutputThread(Impl& ctx)
            : ThreadBase("OutputProcessor"), ctx(ctx) {}

        void Run() override {
            LogInfo("启动");
            while (!IsStopping()) {
                InferenceOutputPacket pkt;
                if (!ctx.inference_output_queue.WaitAndPop(pkt, 100)) {
                    continue;
                }

                if (pkt.header.IsEOS()) {
                    ctx.output_frame_queue.Push(OutputFramePacket::EOS());
                    break;
                }
                if (pkt.header.IsFatal()) {
                    ctx.output_frame_queue.Push(OutputFramePacket::Fatal());
                    break;
                }

                // 注意: 此处需要关联原始人脸数据
                // 简化实现中，我们生成一个空输出帧以便 RenderThread 驱动
                // 完整实现需要 InferenceOutputPacket 携带 ProcessedFaceData

                auto start = std::chrono::steady_clock::now();

                // 模型输出 → cv::Mat
                cv::Mat face_mat = ctx.output_processor.OutputToMat(pkt.payload);
                if (face_mat.empty()) {
                    continue;
                }

                // 简化的后处理（实际需要 M_inv 和 face_mask，此处略）
                OutputFramePacket out_pkt;
                out_pkt.InheritHeader(pkt.header);
                out_pkt.payload = face_mat;
                out_pkt.header.status = StatusCode::OK;

                auto elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                ctx.total_output_us.fetch_add(
                    static_cast<int64_t>(elapsed * 1000.0),
                    std::memory_order_relaxed);
                ctx.output_count.fetch_add(1, std::memory_order_relaxed);

                ctx.output_frame_queue.Push(std::move(out_pkt));
            }
            LogInfo("退出");
        }
    };

    // ---------------------------------------------------------------
    // RenderThread: 帧调度 + 同步判定 + 渲染输出
    // ---------------------------------------------------------------
    struct RenderThread : public ThreadBase {
        Impl& ctx;
        RenderThread(Impl& ctx)
            : ThreadBase("RenderThread"), ctx(ctx) {}

        void Run() override {
            LogInfo("启动");
            ctx.start_time = std::chrono::steady_clock::now();

            int64_t frame_id = 0;

            while (!IsStopping()) {
                if (ctx.paused.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                OutputFramePacket pkt;
                if (!ctx.output_frame_queue.WaitAndPop(pkt, 100)) {
                    continue;
                }

                if (pkt.header.IsEOS()) {
                    break;
                }
                if (pkt.header.IsFatal()) {
                    LogError("下游报告致命错误，停止渲染");
                    break;
                }
                if (pkt.header.IsSkip()) {
                    ctx.frames_skipped.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                // 同步判定 + 帧调度
                ScheduleResult sched = ctx.frame_scheduler.ScheduleFrame(
                    frame_id, static_cast<double>(pkt.header.pts_ms));

                switch (sched.action) {
                    case FrameAction::DISPLAY:
                        ctx.total_frames_out.fetch_add(1,
                            std::memory_order_relaxed);
                        ctx.frame_scheduler.OnFrameDisplayed(
                            static_cast<double>(pkt.header.pts_ms));
                        frame_id++;
                        break;

                    case FrameAction::DROP:
                        ctx.frames_dropped.fetch_add(1,
                            std::memory_order_relaxed);
                        break;

                    case FrameAction::DUPLICATE:
                        // 重复上一帧（输出队列中没有上一帧缓存时，跳过）
                        ctx.total_frames_out.fetch_add(1,
                            std::memory_order_relaxed);
                        ctx.frame_scheduler.OnFrameDisplayed(
                            static_cast<double>(pkt.header.pts_ms));
                        break;

                    case FrameAction::WAIT:
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(
                                static_cast<int>(sched.wait_time_ms)));
                        // 把包放回队列重试
                        // ctx.output_frame_queue.Push(std::move(pkt));  // 简化跳过
                        break;
                }

                // 更新音频时钟（简化：用系统时钟代替 PortAudio 位置）
                // 正式使用时，AudioPlayer::GetConsumedFrames() 提供精确位置
                auto now = std::chrono::steady_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(
                    now - ctx.start_time).count();
                int64_t estimated_samples = static_cast<int64_t>(
                    elapsed_ms / 1000.0 * ctx.config.audio_sample_rate);
                ctx.av_sync.UpdateAudioClock(estimated_samples);
            }

            LogInfo("退出");
        }
    };
};

// ============================================================================
// Pipeline 公有接口实现
// ============================================================================

Pipeline::Pipeline()
    : impl_(std::make_unique<Impl>()) {}

Pipeline::~Pipeline() {
    Stop();
}

bool Pipeline::Init(const PipelineConfig& config) {
    if (impl_->initialized.load()) {
        std::cerr << "[Pipeline] Init: 重复初始化" << std::endl;
        return false;
    }

    // 校验参数
    if (config.audio_sample_rate <= 0) {
        std::cerr << "[Pipeline] Init: 无效采样率" << std::endl;
        return false;
    }
    if (config.target_fps <= 0) {
        std::cerr << "[Pipeline] Init: 无效帧率" << std::endl;
        return false;
    }

    impl_->config = config;
    impl_->ConfigureQueues();

    if (!impl_->InitCoreModules()) {
        return false;
    }

    impl_->CreateWorkers();
    impl_->initialized.store(true, std::memory_order_release);

    std::cout << "[Pipeline] 初始化成功: "
              << config.audio_sample_rate << "Hz, "
              << config.target_fps << "fps" << std::endl;
    return true;
}

bool Pipeline::Start() {
    if (!impl_->initialized.load()) {
        std::cerr << "[Pipeline] Start: 未初始化" << std::endl;
        return false;
    }
    if (impl_->running.load()) {
        return true;
    }

    // 按从下游到上游的顺序启动
    impl_->render_thread->Start();
    impl_->output_thread->Start();
    impl_->inference_thread->Start();
    impl_->video_processor_thread->Start();
    impl_->audio_processor_thread->Start();
    impl_->video_producer_thread->Start();
    impl_->audio_producer_thread->Start();

    impl_->running.store(true, std::memory_order_release);
    std::cout << "[Pipeline] 已启动" << std::endl;
    return true;
}

void Pipeline::Stop() {
    if (!impl_->running.load() && !impl_->initialized.load()) {
        return;
    }

    std::cout << "[Pipeline] 正在停止..." << std::endl;

    // 停止所有队列（唤醒等待线程）
    impl_->audio_raw_queue.Stop();
    impl_->mel_feature_queue.Stop();
    impl_->video_raw_queue.Stop();
    impl_->processed_face_queue.Stop();
    impl_->inference_output_queue.Stop();
    impl_->output_frame_queue.Stop();

    // 按从上游到下游的顺序停止线程
    impl_->audio_producer_thread->Stop();
    impl_->video_producer_thread->Stop();
    impl_->audio_processor_thread->Stop();
    impl_->video_processor_thread->Stop();
    impl_->inference_thread->Stop();
    impl_->output_thread->Stop();
    impl_->render_thread->Stop();

    // 等待所有线程退出
    int timeout = impl_->config.shutdown_timeout_ms;
    impl_->audio_producer_thread->Wait(timeout);
    impl_->video_producer_thread->Wait(timeout);
    impl_->audio_processor_thread->Wait(timeout);
    impl_->video_processor_thread->Wait(timeout);
    impl_->inference_thread->Wait(timeout);
    impl_->output_thread->Wait(timeout);
    impl_->render_thread->Wait(timeout);

    impl_->running.store(false, std::memory_order_release);
    std::cout << "[Pipeline] 已停止" << std::endl;
}

bool Pipeline::IsRunning() const {
    return impl_->running.load(std::memory_order_acquire);
}

// ========================================================================
// 数据输入
// ========================================================================

bool Pipeline::PushAudio(const std::vector<float>& pcm_data, int64_t pts_ms) {
    if (!impl_->running.load()) {
        return false;
    }
    impl_->audio_packets_in.fetch_add(1, std::memory_order_relaxed);
    auto pkt = AudioRawPacket::Make(pcm_data, pts_ms,
                                    impl_->audio_packets_in.load());
    return impl_->audio_raw_queue.Push(std::move(pkt));
}

bool Pipeline::PushVideo(const cv::Mat& frame, int64_t pts_ms) {
    if (!impl_->running.load()) {
        return false;
    }
    impl_->video_packets_in.fetch_add(1, std::memory_order_relaxed);
    impl_->total_frames_in.fetch_add(1, std::memory_order_relaxed);
    auto pkt = VideoFramePacket::Make(frame.clone(), pts_ms,
                                      impl_->video_packets_in.load());
    return impl_->video_raw_queue.Push(std::move(pkt));
}

void Pipeline::MarkAudioEOS() {
    impl_->audio_eos.store(true, std::memory_order_release);
}

void Pipeline::MarkVideoEOS() {
    impl_->video_eos.store(true, std::memory_order_release);
}

// ========================================================================
// 数据输出
// ========================================================================

bool Pipeline::GetOutputFrame(OutputFramePacket& frame, int timeout_ms) {
    return impl_->output_frame_queue.WaitAndPop(frame, timeout_ms);
}

// ========================================================================
// 控制
// ========================================================================

void Pipeline::Pause() {
    impl_->paused.store(true, std::memory_order_release);
    std::cout << "[Pipeline] 已暂停" << std::endl;
}

void Pipeline::Resume() {
    impl_->paused.store(false, std::memory_order_release);
    std::cout << "[Pipeline] 已恢复" << std::endl;
}

bool Pipeline::IsPaused() const {
    return impl_->paused.load(std::memory_order_acquire);
}

// ========================================================================
// 查询
// ========================================================================

PipelineMetrics Pipeline::GetMetrics() const {
    PipelineMetrics m;
    m.total_frames_in   = impl_->total_frames_in.load();
    m.total_frames_out  = impl_->total_frames_out.load();
    m.frames_dropped    = impl_->frames_dropped.load();
    m.frames_skipped    = impl_->frames_skipped.load();
    m.audio_packets_in  = impl_->audio_packets_in.load();
    m.video_packets_in  = impl_->video_packets_in.load();
    m.inference_count   = impl_->inference_count.load();

    auto apc = impl_->audio_process_count.load();
    if (apc > 0)
        m.avg_audio_process_ms = static_cast<double>(
            impl_->total_audio_process_us.load()) / apc / 1000.0;
    auto vpc = impl_->video_process_count.load();
    if (vpc > 0)
        m.avg_video_process_ms = static_cast<double>(
            impl_->total_video_process_us.load()) / vpc / 1000.0;
    auto ic = impl_->inference_count_val.load();
    if (ic > 0)
        m.avg_inference_ms = static_cast<double>(
            impl_->total_inference_us.load()) / ic / 1000.0;
    auto oc = impl_->output_count.load();
    if (oc > 0)
        m.avg_output_ms = static_cast<double>(
            impl_->total_output_us.load()) / oc / 1000.0;

    auto stats = impl_->frame_scheduler.GetStats();
    m.actual_fps = stats.actual_fps;

    return m;
}

FrameStats Pipeline::GetFrameStats() const {
    return impl_->frame_scheduler.GetStats();
}

SyncStatus Pipeline::GetSyncStatus() const {
    return impl_->av_sync.GetSyncStatus(0.0).status;
}

double Pipeline::GetDriftMs() const {
    // 简化：从 AVSync 获取最近一次同步结果
    return 0.0;
}

double Pipeline::GetAudioClockMs() const {
    return impl_->av_sync.GetAudioClockMs();
}

}  // namespace core
}  // namespace digital_human
