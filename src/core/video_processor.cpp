#include "core/video_processor.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"

namespace digital_human {
namespace core {

using digital_human::core::FaceDetector;
using digital_human::core::FaceAligner;
using digital_human::core::FaceMaskGenerator;

// ============================================================================
// Impl 结构体
// ============================================================================

struct VideoProcessor::Impl {
    // ---- 配置 ----
    VideoProcessorConfig config;

    // ---- 队列 ----
    ThreadSafeQueue<VideoFramePacket>* input_queue_  = nullptr;
    ThreadSafeQueue<ProcessedFacePacket>* output_queue_ = nullptr;

    // ---- 视觉处理模块（每个线程独立实例） ----
    FaceDetector     face_detector;
    FaceAligner     face_aligner;
    FaceMaskGenerator face_mask_gen;

    // ---- SCRFD + 2D106 模型目录 ----
    std::string landmark_model_path;

    // ---- 状态 ----
    int64_t           processed_count_ = 0;
    std::atomic<bool> input_eos_{false};

    // ---- 静态人脸缓存（拟合图片/视频用例：同一张人脸重复使用） ----
    // 命中条件：图像尺寸一致 + 像素指纹一致（首末行采样）
    // 命中后跳过人脸检测/对齐/mask 生成，复用上次的
    // aligned_face / M_inv / face_mask / face_rect / landmarks_96
    struct FaceCache {
        cv::Size                  frame_size{0, 0};
        // 像素指纹：首行 + 末行 + 中间一行（每行抽样若干像素）
        std::vector<uchar>        signature;
        cv::Mat                   aligned_face;
        cv::Mat                   M_inv;
        cv::Mat                   face_mask;
        cv::Rect                  face_rect;
        std::vector<cv::Point2f>  landmarks_96;
        bool                      valid = false;
    } face_cache_;

    // ---- 缓存命中/未命中统计 ----
    int64_t cache_hits_   = 0;
    int64_t cache_misses_ = 0;

    // ========================================================================
    // 像素指纹：对帧的若干采样行求和，用于快速判定图像是否变化
    // 仅做轻量哈希，不做密码学强度判等
    // ========================================================================
    std::vector<uchar> ComputeSignature(const cv::Mat& frame) {
        if (frame.empty()) return {};
        const int rows = frame.rows;
        const int cols = frame.cols;
        const int channels = frame.channels();
        // 采样：首行、中间行、末行各取 16 个等距像素
        std::vector<uchar> sig;
        sig.reserve(16 * 3 * channels);
        auto sample_row = [&](int y) {
            const uchar* row = frame.ptr<uchar>(y);
            for (int i = 0; i < 16; ++i) {
                int x = (i * cols) / 16;
                const uchar* p = row + x * channels;
                for (int c = 0; c < channels; ++c) {
                    sig.push_back(p[c]);
                }
            }
        };
        sample_row(0);
        sample_row(rows / 2);
        sample_row(rows - 1);
        return sig;
    }

    bool SignatureEqual(const std::vector<uchar>& a,
                        const std::vector<uchar>& b) {
        if (a.size() != b.size()) return false;
        return std::memcmp(a.data(), b.data(), a.size()) == 0;
    }

    // ========================================================================
    // 处理核心逻辑
    // ========================================================================

    /// @brief 处理一帧视频
    ProcessedFacePacket ProcessOne(const VideoFramePacket& pkt) {
        ProcessedFacePacket result;
        result.InheritHeader(pkt.header);

        try {
            const cv::Mat& frame = pkt.payload;

            // 1. 查缓存：若指纹一致，跳过完整人脸流水线
            //    (拟合图片场景：同一张脸 ×N 帧音频 → 100% 命中)
            auto sig = ComputeSignature(frame);
            if (face_cache_.valid &&
                face_cache_.frame_size == frame.size() &&
                SignatureEqual(face_cache_.signature, sig)) {
                // 复用对齐/mask 结果，仅刷新 original_face 指向当前帧
                result.payload.aligned_face  = face_cache_.aligned_face;
                result.payload.M_inv         = face_cache_.M_inv;
                result.payload.face_mask     = face_cache_.face_mask;
                result.payload.original_face = frame;  // 浅拷贝共享，下游只读
                result.payload.face_rect     = face_cache_.face_rect;
                result.payload.landmarks_96  = face_cache_.landmarks_96;
                result.header.status         = StatusCode::OK;
                ++cache_hits_;
                return result;
            }

            // 2. 人脸检测（缓存未命中时走完整流水线）
            auto faces = face_detector.detect(frame);
            if (faces.empty()) {
                LogWarn("未检测到人脸");
                result.header.status = StatusCode::SKIP;
                // 检测失败时不污染缓存
                return result;
            }

            // 取最大的人脸
            auto max_face = std::max_element(faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b) {
                    return a.area() < b.area();
                });

            // 3. 获取关键点
            auto landmarks = face_detector.getLandmarks(frame, *max_face);
            if (landmarks.empty()) {
                result.header.status = StatusCode::SKIP;
                return result;
            }

            // 4. 人脸对齐
            std::vector<cv::Point2f> landmarks_f;
            landmarks_f.reserve(landmarks.size());
            for (const auto& pt : landmarks) {
                landmarks_f.emplace_back(
                    static_cast<float>(pt.x),
                    static_cast<float>(pt.y));
            }

            auto align_result = face_aligner.alignByRect(
                frame, landmarks_f, config.face_size, *max_face);

            if (!align_result.valid) {
                result.header.status = StatusCode::SKIP;
                return result;
            }

            // 5. 96x96 空间的精细遮罩（优先）
            //    若成功，无需再生成全图 mouth_mask（避免一次全图 fillPoly）
            auto precise_mask = face_mask_gen.generatePreciseMouthAlphaMask96(
                align_result.landmarks);

            cv::Mat final_mask;
            if (!precise_mask.empty()) {
                final_mask = std::move(precise_mask);
            } else {
                // 退化路径：96×96 失败才回退到全图 mask
                final_mask = face_mask_gen.generateMouthMask(
                    frame.size(), landmarks);
            }

            // 填充结果
            result.payload.aligned_face  = align_result.aligned_face;
            result.payload.M_inv         = align_result.M_inv;
            result.payload.face_mask     = std::move(final_mask);
            result.payload.original_face = frame;  // 浅拷贝共享，下游只读
            result.payload.face_rect     = align_result.face_rect;
            result.payload.landmarks_96  = align_result.landmarks;
            result.header.status         = StatusCode::OK;

            // 6. 写入缓存
            face_cache_.frame_size    = frame.size();
            face_cache_.signature     = std::move(sig);
            face_cache_.aligned_face  = result.payload.aligned_face;
            face_cache_.M_inv         = result.payload.M_inv;
            face_cache_.face_mask     = result.payload.face_mask;
            face_cache_.face_rect     = result.payload.face_rect;
            face_cache_.landmarks_96  = result.payload.landmarks_96;
            face_cache_.valid         = true;
            ++cache_misses_;

        } catch (const std::exception& e) {
            LogError(std::string("视频处理异常: ") + e.what());
            result.header.status = StatusCode::ERROR;
        }

        return result;
    }

    void LogError(const std::string& msg) {
        std::cerr << "[VideoProcessor] ERROR: " << msg << std::endl;
    }

    void LogWarn(const std::string& msg) {
        std::cout << "[VideoProcessor] WARN: " << msg << std::endl;
    }

    void LogInfo(const std::string& msg) {
        std::cout << "[VideoProcessor] " << msg << std::endl;
    }
};

// ============================================================================
// 构造 / 析构
// ============================================================================

VideoProcessor::VideoProcessor(const std::string& name)
    : ThreadBase(name)
    , impl_(std::make_unique<Impl>()) {}

VideoProcessor::~VideoProcessor() {
    Shutdown();
}

// ============================================================================
// 配置
// ============================================================================

void VideoProcessor::SetConfig(const VideoProcessorConfig& config) {
    impl_->config = config;
}

const VideoProcessorConfig& VideoProcessor::GetConfig() const {
    return impl_->config;
}

// ============================================================================
// 数据源
// ============================================================================

void VideoProcessor::SetInputQueue(ThreadSafeQueue<VideoFramePacket>* queue) {
    impl_->input_queue_ = queue;
}

void VideoProcessor::SetOutputQueue(ThreadSafeQueue<ProcessedFacePacket>* queue) {
    impl_->output_queue_ = queue;
}

void VideoProcessor::SetLandmarkModelPath(const std::string& path) {
    impl_->landmark_model_path = path;
    // FaceDetector 会在首次检测时按需加载模型
}

bool VideoProcessor::LoadFaceModel(const std::string& path) {
    if (IsRunning()) {
        impl_->LogError("线程运行中，拒绝重新加载人脸模型");
        return false;
    }
    impl_->landmark_model_path = path;
    return impl_->face_detector.loadModel(path);
}

bool VideoProcessor::IsFaceModelLoaded() const {
    return impl_->face_detector.isModelLoaded();
}

// ============================================================================
// 线程主循环
// ============================================================================

void VideoProcessor::Run() {
    impl_->LogInfo("启动");

    if (!impl_->input_queue_) {
        impl_->LogError("输入队列未设置");
        return;
    }
    if (!impl_->output_queue_) {
        impl_->LogError("输出队列未设置");
        return;
    }

    // 加载 SCRFD + 2D106 人脸模型（支持按需加载）
    if (impl_->face_detector.isModelLoaded()) {
        impl_->LogInfo("人脸关键点模型已加载: " +
                       impl_->landmark_model_path);
    } else if (!impl_->landmark_model_path.empty()) {
        if (impl_->face_detector.loadModel(impl_->landmark_model_path)) {
            impl_->LogInfo("人脸关键点模型已加载: " + impl_->landmark_model_path);
        } else {
            impl_->LogError("人脸关键点模型加载失败，终止视频处理");
            impl_->output_queue_->Push(ProcessedFacePacket::Fatal());
            return;
        }
    } else {
        impl_->LogError("未配置人脸模型，终止视频处理");
        impl_->output_queue_->Push(ProcessedFacePacket::Fatal());
        return;
    }

    while (!IsStopping()) {
        VideoFramePacket pkt;
        if (!impl_->input_queue_->WaitAndPop(pkt, impl_->config.pop_timeout_ms)) {
            // 超时：检查是否应退出
            if (IsStopping()) break;
            if (impl_->input_eos_.load(std::memory_order_acquire)
                && impl_->input_queue_->Empty()) {
                impl_->LogInfo("视频帧耗尽，发送 EOS");
                impl_->output_queue_->Push(ProcessedFacePacket::EOS());
                break;
            }
            continue;
        }

        if (pkt.header.IsEOS()) {
            impl_->output_queue_->Push(ProcessedFacePacket::EOS());
            break;
        }
        if (pkt.header.IsFatal()) {
            impl_->output_queue_->Push(ProcessedFacePacket::Fatal());
            break;
        }
        if (pkt.header.IsSkip() || pkt.payload.empty()) {
            continue;
        }

        // 处理一帧
        auto result = impl_->ProcessOne(pkt);
        impl_->processed_count_++;

        if (result.header.IsOK()) {
            impl_->output_queue_->Push(std::move(result));
        }
        // SKIP/ERROR 包直接丢弃

        if (impl_->processed_count_ % 10 == 0) {
            impl_->LogInfo("已处理 " +
                std::to_string(impl_->processed_count_) + " 帧");
        }
    }

    impl_->LogInfo("退出 (处理 " +
                   std::to_string(impl_->processed_count_) + " 帧)");
}

// ============================================================================
// 控制
// ============================================================================

void VideoProcessor::MarkInputEOS() {
    impl_->input_eos_.store(true, std::memory_order_release);
}

void VideoProcessor::Reset() {
    impl_->processed_count_ = 0;
    impl_->input_eos_.store(false, std::memory_order_release);
}

// ============================================================================
// 状态查询
// ============================================================================

int64_t VideoProcessor::GetProcessedCount() const {
    return impl_->processed_count_;
}

bool VideoProcessor::IsInputEOS() const {
    return impl_->input_eos_.load(std::memory_order_acquire);
}

}  // namespace core
}  // namespace digital_human
