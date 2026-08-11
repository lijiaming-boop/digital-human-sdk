#pragma once

#include <memory>
#include <cstdint>

#include "core/thread_base.h"
#include "core/thread_safe_queue.h"
#include "core/packet.h"

namespace digital_human {
namespace core {

// ============================================================================
// 视频处理配置
// ============================================================================

/// @brief 视频处理线程配置
struct VideoProcessorConfig {
    int face_size             = 96;       ///< 对齐人脸尺寸
    double face_detect_scale  = 0.5;      ///< 人脸检测缩放比例
    int    max_faces          = 1;        ///< 最大检测人脸数
    int    pop_timeout_ms     = 100;      ///< 队列弹出超时（毫秒）
};

// ============================================================================
// 视频处理线程
// ============================================================================

/**
 * @brief 视频处理线程
 *
 * 从输入队列获取原始视频帧，经过人脸检测→人脸对齐→口唇遮罩生成流水线，
 * 将处理后的人脸数据推送到输出队列。
 *
 * 处理流程：
 *   cv::Mat → FaceDetect → FaceAlign (96×96) → FaceMask → ProcessedFaceData
 *
 * 线程安全：内部状态通过队列隔离，所有处理模块为独立实例。
 */
class VideoProcessor : public ThreadBase {
public:
    explicit VideoProcessor(const std::string& name = "VideoProcessor");
    ~VideoProcessor() override;

    VideoProcessor(const VideoProcessor&) = delete;
    VideoProcessor& operator=(const VideoProcessor&) = delete;
    VideoProcessor(VideoProcessor&&) = delete;
    VideoProcessor& operator=(VideoProcessor&&) = delete;

    // ========================================================================
    // 配置
    // ========================================================================

    /// @brief 设置处理参数
    void SetConfig(const VideoProcessorConfig& config);

    /// @brief 获取当前配置
    const VideoProcessorConfig& GetConfig() const;

    // ========================================================================
    // 数据源
    // ========================================================================

    /**
     * @brief 设置原始视频帧输入队列
     *
     * @param queue VideoFramePacket 队列指针（外部管理生命周期）
     */
    void SetInputQueue(ThreadSafeQueue<VideoFramePacket>* queue);

    // ========================================================================
    // 输出队列
    // ========================================================================

    /**
     * @brief 设置处理后的人脸数据输出队列
     *
     * @param queue ProcessedFacePacket 队列指针（外部管理生命周期）
     */
    void SetOutputQueue(ThreadSafeQueue<ProcessedFacePacket>* queue);

    // ========================================================================
    // 模型路径（用于 SCRFD + 2D106）
    // ========================================================================

    /// @brief 设置 SCRFD + 2D106 人脸模型目录
    void SetLandmarkModelPath(const std::string& path);

    // ========================================================================
    // 线程主循环
    // ========================================================================

    void Run() override;

    // ========================================================================
    // 控制
    // ========================================================================

    /// @brief 标记输入结束（无更多视频帧）
    void MarkInputEOS();

    /// @brief 重置所有状态
    void Reset();

    // ========================================================================
    // 状态查询
    // ========================================================================

    /// @brief 累计处理帧数
    int64_t GetProcessedCount() const;

    /// @brief 当前 EOS 状态
    bool IsInputEOS() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
