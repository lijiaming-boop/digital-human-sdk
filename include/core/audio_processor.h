#pragma once

#include <memory>
#include <atomic>
#include <vector>
#include <cstdint>

#include "core/thread_base.h"
#include "core/thread_safe_queue.h"
#include "core/packet.h"

namespace digital_human {
namespace audio {
class RingBuffer;
}

namespace core {

// ============================================================================
// 音频处理配置
// ============================================================================

/// @brief 音频处理线程配置
struct AudioProcessorConfig {
    int sample_rate          = 16000;  ///< 采样率（Hz）
    int channels             = 1;       ///< 声道数
    int frame_size           = 400;     ///< 帧长（samples @16kHz = 25ms）
    int hop_size             = 160;     ///< 帧移（samples @16kHz = 10ms）
    int mel_bins             = 80;      ///< Mel 滤波器组数
    int nfft                 = 512;     ///< FFT 点数
    int window_capacity      = 4800;    ///< 滑动窗口最大容量（samples, 300ms）

    /// @brief 根据当前采样率自适应计算帧参数
    void AutoConfigure(int target_sample_rate = 16000) {
        if (target_sample_rate <= 0) return;
        double ratio = static_cast<double>(target_sample_rate) / sample_rate;
        frame_size  = static_cast<int>(400 * ratio);
        hop_size    = static_cast<int>(160 * ratio);
        nfft        = 512;
        sample_rate = target_sample_rate;
    }
};

// ============================================================================
// 音频处理线程
// ============================================================================

/**
 * @brief 音频处理线程
 *
 * 从音频缓冲区读取 PCM 数据，经过完整音频特征提取流水线
 * （降噪→分帧→VAD→预加重→RMS归一化→Mel频谱→CMVN），
 * 将 Mel 特征推送到推理队列。
 *
 * 实时性保证：
 * - 读取 RingBuffer 不阻塞 PortAudio 回调路径
 * - 滑动窗口管理，每 hop=10ms 输出一帧特征
 * - 空闲时休眠，不空转 CPU
 *
 * 两种数据源模式：
 * - 文件模式：SetAudioSource() 指定完整 PCM 缓冲区
 * - 流式模式：SetRingBuffer() 指定 RingBuffer
 */
class AudioProcessor : public ThreadBase {
public:
    /// @brief 构造音频处理器
    /// @param name 线程名称（默认 "AudioProcessor"）
    explicit AudioProcessor(const std::string& name = "AudioProcessor");

    ~AudioProcessor() override;

    AudioProcessor(const AudioProcessor&) = delete;
    AudioProcessor& operator=(const AudioProcessor&) = delete;
    AudioProcessor(AudioProcessor&&) = delete;
    AudioProcessor& operator=(AudioProcessor&&) = delete;

    // ========================================================================
    // 配置
    // ========================================================================

    /// @brief 设置处理参数
    void SetConfig(const AudioProcessorConfig& config);

    /// @brief 获取当前配置
    const AudioProcessorConfig& GetConfig() const;

    // ========================================================================
    // 数据源
    // ========================================================================

    /**
     * @brief 设置固定音频数据源（文件模式）
     *
     * 一次性加载所有音频数据，处理线程按滑动窗口自动推进。
     *
     * @param data    PCM float 样本 (range [-1.0, 1.0])
     * @param samples 样本总数
     * @param rate    采样率
     * @param ch      声道数（仅 mono 处理，立体声自动混合）
     */
    void SetAudioSource(const float* data, size_t samples,
                        int rate, int ch);

    /**
     * @brief 设置 RingBuffer 数据源（流式模式）
     *
     * 从 RingBuffer 持续读取新增音频数据，适用于实时音频流。
     *
     * @param buffer RingBuffer 实例指针（外部管理生命周期）
     */
    void SetRingBuffer(audio::RingBuffer* buffer);

    // ========================================================================
    // 输出队列
    // ========================================================================

    /// @brief 设置 Mel 特征输出队列
    void SetOutputQueue(ThreadSafeQueue<MelFeaturePacket>* queue);

    // ========================================================================
    // 线程主循环
    // ========================================================================

    /// @brief 线程主循环（由 ThreadBase 框架调用）
    void Run() override;

    // ========================================================================
    // 控制
    // ========================================================================

    /// @brief 标记音频数据结束（文件模式下自动，流式模式下需手动调用）
    void MarkEOS();

    /// @brief 重置处理器（清空滑动窗口和所有状态）
    void Reset();

    // ========================================================================
    // 状态查询
    // ========================================================================

    /// @brief 已处理的样本数
    int64_t GetProcessedSamples() const;

    /// @brief 已处理的音频时长（毫秒）
    double GetProcessedDurationMs() const;

    /// @brief 滑动窗口中待处理的帧数
    int64_t GetPendingFrames() const;

    /// @brief 累计输出的 Mel 特征帧数
    int64_t GetOutputCount() const;

    /// @brief 获取当前处理位置相对于源的进度 [0.0, 1.0]
    double GetProgress() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
