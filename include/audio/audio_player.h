#pragma once

#include <memory>
#include <cstdint>
#include <vector>

namespace digital_human {
namespace audio {

/// @brief 音频播放状态枚举
enum class AudioPlayerState {
    IDLE,       ///< 空闲（未初始化或已停止）
    PLAYING,    ///< 播放中
    PAUSED,     ///< 已暂停
    STOPPED,    ///< 已停止
    FINISHED    ///< 播放完毕
};

/**
 * @brief 音频播放器（基于 PortAudio）
 *
 * 封装 PortAudio 的初始化、音频流配置和播放控制。
 * 支持 PCM float 格式的音频数据播放，提供精确的播放位置和 DAC 时间戳。
 *
 * 状态机：
 *   IDLE → PLAYING ↔ PAUSED
 *   PLAYING → STOPPED/FINISHED → IDLE
 *   PAUSED → STOPPED → IDLE
 *
 * 线程安全：PortAudio 回调在独立音频线程运行，通过原子变量保护共享状态。
 */
class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();
    AudioPlayer(const AudioPlayer&) = delete;
    AudioPlayer& operator=(const AudioPlayer&) = delete;
    AudioPlayer(AudioPlayer&&) noexcept;
    AudioPlayer& operator=(AudioPlayer&&) noexcept;

    // ==================== 初始化与资源管理 ====================

    /**
     * @brief 初始化 PortAudio 库并配置音频流参数
     *
     * @param sampleRate        采样率（Hz），默认 48000
     * @param channels          声道数，默认 2
     * @param framesPerBuffer   每缓冲区帧数，默认 512
     * @return true  成功
     * @return false 失败（可调用 GetLastErrorMsg 获取详情）
     */
    bool Init(int sampleRate = 48000, int channels = 2, int framesPerBuffer = 512);

    /// @brief 销毁播放器，释放 PortAudio 资源
    void Destroy();

    /// @brief 检查是否已初始化
    bool IsInitialized() const;

    // ==================== 音频数据加载 ====================

    /**
     * @brief 加载 PCM float 音频数据
     *
     * @param samples    音频样本数据（interleaved，范围 [-1.0, 1.0]）
     * @param numSamples 样本总数
     * @param channels   声道数
     * @return true  成功
     * @return false 失败
     */
    bool LoadAudio(const float* samples, int numSamples, int channels);

    /// @brief 加载音频数据（vector 版本）
    bool LoadAudio(const std::vector<float>& samples, int channels);

    // ==================== 播放控制 ====================

    /// @brief 开始播放
    bool Play();

    /// @brief 暂停播放（保持当前位置）
    bool Pause();

    /// @brief 从暂停位置恢复播放
    bool Resume();

    /**
     * @brief 停止播放
     *
     * 重置播放位置到开头，可再次调用 Play() 重新播放。
     */
    bool Stop();

    // ==================== 状态查询 ====================

    /// @brief 获取当前播放状态
    AudioPlayerState GetState() const;

    /// @brief 是否正在播放
    bool IsPlaying() const;

    /// @brief 是否已暂停
    bool IsPaused() const;

    /// @brief 是否已停止
    bool IsStopped() const;

    /// @brief 是否播放完毕
    bool IsFinished() const;

    // ==================== 位置与时间查询 ====================

    /**
     * @brief 获取已消耗的音频帧数
     *
     * 每帧包含 channels 个样本。该值由 PortAudio 回调线程原子更新，
     * 反映已写入音频输出的总帧数。
     *
     * @return int64_t 已消耗的总帧数
     */
    int64_t GetConsumedFrames() const;

    /**
     * @brief 获取音频播放位置（毫秒）
     *
     * 基于已消耗的帧数计算：position = consumed_frames / sample_rate * 1000
     * 与 AVSync 的 UpdateAudioClock 机制兼容。
     *
     * @return double 播放位置（毫秒）
     */
    double GetPlaybackPositionMs() const;

    /**
     * @brief 获取精确的 DAC 输出时间（毫秒）
     *
     * 基于 PortAudio 内部时钟 Pa_GetStreamTime()，
     * 反映音频硬件实际输出时间，用于高精度音视频同步。
     *
     * @return double DAC 时间（毫秒），失败时返回 -1.0
     */
    double GetDacTimeMs() const;

    /**
     * @brief 获取已加载音频数据的总时长（毫秒）
     *
     * @return double 总时长（毫秒）
     */
    double GetTotalDurationMs() const;

    /// @brief 获取采样率
    int GetSampleRate() const;

    /// @brief 获取声道数
    int GetChannels() const;

    // ==================== 错误处理 ====================

    /// @brief 获取最后一次错误消息
    const char* GetLastErrorMsg() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
