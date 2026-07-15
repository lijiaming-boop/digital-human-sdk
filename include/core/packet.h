#pragma once

#include <cstdint>
#include <string>
#include <opencv2/core.hpp>
#include <ncnn/mat.h>

namespace digital_human {
namespace core {

// ============================================================================
// 状态码
// ============================================================================

enum class StatusCode : int8_t {
    OK      = 0,    ///< 处理成功
    ERROR   = 1,    ///< 可恢复错误
    FATAL   = 2,    ///< 致命错误，触发 Pipeline 停止
    EOS     = 3,    ///< 流结束 (End of Stream)
    SKIP    = 4,    ///< 跳过此帧（同步丢弃或处理失败）
    TIMEOUT = 5,    ///< 等待超时
};

/// @brief 状态码字符串描述
inline const char* StatusCodeToString(StatusCode code) {
    switch (code) {
        case StatusCode::OK:      return "OK";
        case StatusCode::ERROR:   return "ERROR";
        case StatusCode::FATAL:   return "FATAL";
        case StatusCode::EOS:     return "EOS";
        case StatusCode::SKIP:    return "SKIP";
        case StatusCode::TIMEOUT: return "TIMEOUT";
        default:                  return "UNKNOWN";
    }
}

// ============================================================================
// 通用数据包头
// ============================================================================

struct PacketHeader {
    int64_t    pts_ms    = 0;       ///< 呈现时间戳（毫秒）
    int64_t    seq_id    = 0;       ///< 单调递增序列号
    int64_t    source_id = 0;       ///< 数据源标识
    StatusCode status    = StatusCode::OK;  ///< 处理状态
    double     cost_ms   = 0.0;     ///< 本阶段处理耗时（毫秒）

    bool IsOK()    const { return status == StatusCode::OK; }
    bool IsEOS()   const { return status == StatusCode::EOS; }
    bool IsError() const { return status == StatusCode::ERROR; }
    bool IsFatal() const { return status == StatusCode::FATAL; }
    bool IsSkip()  const { return status == StatusCode::SKIP; }
    bool IsTerminal() const { return status == StatusCode::EOS || status == StatusCode::FATAL; }
};

// ============================================================================
// 通用数据包模板
// ============================================================================

template <typename T>
struct Packet {
    PacketHeader header;
    T            payload;

    Packet() = default;

    /// @brief 构造 OK 数据包
    static Packet<T> Make(T data, int64_t pts_ms = 0, int64_t seq_id = 0) {
        Packet<T> pkt;
        pkt.payload      = std::move(data);
        pkt.header.pts_ms = pts_ms;
        pkt.header.seq_id = seq_id;
        pkt.header.status = StatusCode::OK;
        return pkt;
    }

    /// @brief 构造 EOS 终结包
    static Packet<T> EOS() {
        Packet<T> pkt;
        pkt.header.status = StatusCode::EOS;
        return pkt;
    }

    /// @brief 构造 Fatal 错误包
    static Packet<T> Fatal() {
        Packet<T> pkt;
        pkt.header.status = StatusCode::FATAL;
        return pkt;
    }

    /// @brief 构造 Skip 包
    static Packet<T> Skip(int64_t pts_ms = 0, int64_t seq_id = 0) {
        Packet<T> pkt;
        pkt.header.pts_ms = pts_ms;
        pkt.header.seq_id = seq_id;
        pkt.header.status = StatusCode::SKIP;
        return pkt;
    }

    /// @brief 转发包头（下游继承上游的 pts/seq）
    void InheritHeader(const PacketHeader& parent) {
        header.pts_ms    = parent.pts_ms;
        header.seq_id    = parent.seq_id;
        header.source_id = parent.source_id;
    }
};

// ============================================================================
// 人脸处理后数据结构
// ============================================================================

struct ProcessedFaceData {
    cv::Mat                    aligned_face;    ///< 96×96 对齐人脸 (BGR uint8)
    cv::Mat                    M_inv;            ///< 逆仿射变换矩阵 2×3
    cv::Mat                    face_mask;        ///< 口唇遮罩 (CV_32FC1)
    cv::Mat                    original_face;    ///< 原始人脸裁剪 (BGR uint8)
    cv::Rect                   face_rect;        ///< 人脸在原图中的矩形
    std::vector<cv::Point2f>   landmarks_96;     ///< 96x96 空间的关键点

    bool IsValid() const {
        return !aligned_face.empty() && !M_inv.empty();
    }
};

// ============================================================================
// 数据包类型别名
// ============================================================================

using AudioRawPacket        = Packet<std::vector<float>>;       ///< PCM float 音频数据
using MelFeaturePacket      = Packet<cv::Mat>;                  ///< Mel 频谱特征
using VideoFramePacket      = Packet<cv::Mat>;                  ///< 原始视频帧
using ProcessedFacePacket   = Packet<ProcessedFaceData>;        ///< 处理后的人脸数据
using InferenceOutputPacket = Packet<ncnn::Mat>;                ///< 模型推理输出
using OutputFramePacket     = Packet<cv::Mat>;                  ///< 最终输出帧

}  // namespace core
}  // namespace digital_human
