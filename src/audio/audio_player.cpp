#include "audio/audio_player.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <portaudio.h>

namespace digital_human {
namespace audio {

// ============================================================================
// 内部常量
// ============================================================================

/// @brief PortAudio 初始化引用计数
static std::atomic<int> gPaRefCount{0};

/// @brief PortAudio 初始化互斥锁
static std::mutex gPaMutex;

// ============================================================================
// Impl 结构体
// ============================================================================

struct AudioPlayer::Impl {
    // ---- 配置 ----
    int    sample_rate        = 48000;
    int    channels           = 2;
    int    frames_per_buffer  = 512;
    bool   initialized        = false;

    // ---- 音频数据 ----
    std::vector<float> audio_data;      ///< PCM float 样本数据 (interleaved)
    int64_t total_frames     = 0;       ///< 总帧数 (每帧 = channels 个样本)
    bool    data_loaded      = false;

    // ---- 播放状态（原子变量，由回调线程和主线程共享） ----
    std::atomic<AudioPlayerState> state{AudioPlayerState::IDLE};
    std::atomic<int64_t>          consumed_frames{0};  ///< 已消耗的帧数

    // ---- 回调侧的读取位置（仅在回调线程中访问） ----
    int64_t read_frame_pos  = 0;        ///< 回调线程读取位置（帧数）
    bool    callback_active = false;    ///< 回调是否正在运行

    // ---- 时间追踪 ----
    PaStream*       stream             = nullptr;
    PaTime          stream_start_time  = 0.0;   ///< 流启动时的 PaTime
    PaTime          pause_start_time   = 0.0;   ///< 暂停开始时的 PaTime
    std::atomic<double> total_paused_duration{0.0}; ///< 累计暂停时长（秒）

    // ---- 精确 DAC 时间追踪 ----
    std::atomic<double> last_dac_time{0.0};  ///< 上次回调中的 DAC 时间（秒）

    // ---- 错误消息 ----
    std::string last_error_msg;

    // ========================================================================
    // PortAudio 回调函数 (静态)
    // ========================================================================

    /**
     * @brief PortAudio 输出回调
     *
     * @param input             输入缓冲区（本模块未使用）
     * @param output            输出缓冲区（填充音频数据）
     * @param frameCount        本缓冲区的帧数
     * @param timeInfo          时间信息（含 outputBufferDacTime）
     * @param statusFlags       状态标志
     * @param userData          用户数据指针（指向 Impl）
     * @return int paContinue 或 paComplete
     */
    static int paCallback(const void* input, void* output,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void* userData) {
        (void)input;
        (void)statusFlags;

        auto* impl = static_cast<Impl*>(userData);
        float* out = static_cast<float*>(output);

        // 保存 DAC 时间（用于精确时间戳查询）
        impl->last_dac_time.store(
            timeInfo ? timeInfo->outputBufferDacTime : 0.0,
            std::memory_order_relaxed);

        AudioPlayerState currentState = impl->state.load(std::memory_order_acquire);

        // 暂停/停止/空闲/结束 → 输出静音
        if (currentState != AudioPlayerState::PLAYING ||
            !impl->data_loaded ||
            impl->read_frame_pos >= impl->total_frames) {

            std::memset(out, 0, frameCount * impl->channels * sizeof(float));

            // 标记播放完毕
            if (impl->data_loaded && impl->read_frame_pos >= impl->total_frames) {
                impl->state.store(AudioPlayerState::FINISHED, std::memory_order_release);
            }

            return paContinue;
        }

        // 计算要写入的样本数
        int64_t samples_needed  = static_cast<int64_t>(frameCount) * impl->channels;
        int64_t samples_avail   = impl->audio_data.size() - impl->read_frame_pos * impl->channels;
        int64_t samples_to_write = std::min(samples_needed, samples_avail);

        if (samples_to_write > 0) {
            std::memcpy(out,
                        impl->audio_data.data() + impl->read_frame_pos * impl->channels,
                        samples_to_write * sizeof(float));
        }

        // 如果输出缓冲区未填满，补零
        if (samples_to_write < samples_needed) {
            std::memset(out + samples_to_write, 0,
                       (samples_needed - samples_to_write) * sizeof(float));
        }

        // 更新读取位置（帧数）
        int64_t frames_written = samples_to_write / impl->channels;
        impl->read_frame_pos += frames_written;
        impl->consumed_frames.store(
            impl->read_frame_pos,
            std::memory_order_release);

        // 如果所有数据均已消耗
        if (impl->read_frame_pos >= impl->total_frames) {
            impl->state.store(AudioPlayerState::FINISHED, std::memory_order_release);
        }

        return paContinue;
    }

    // ========================================================================
    // 内部工具方法
    // ========================================================================

    /// @brief 设置错误消息
    void setError(const std::string& msg) {
        last_error_msg = msg;
        std::cerr << "[AudioPlayer] " << msg << std::endl;
    }

    /// @brief 获取当前播放状态下允许的 PaTime（考虑暂停）
    double getEffectivePaTime() const {
        if (!stream) return 0.0;

        PaTime now = Pa_GetStreamTime(stream);
        if (now == 0.0) return 0.0;

        double pausedDur = total_paused_duration.load(std::memory_order_acquire);

        // 播放中：流时间 - 启动时间 - 暂停总时长
        return (now - stream_start_time - pausedDur) * 1000.0;  // 毫秒
    }
};

// ============================================================================
// 构造 / 析构
// ============================================================================

AudioPlayer::AudioPlayer()
    : impl_(std::make_unique<Impl>()) {}

AudioPlayer::~AudioPlayer() {
    Destroy();
}

AudioPlayer::AudioPlayer(AudioPlayer&&) noexcept = default;

AudioPlayer& AudioPlayer::operator=(AudioPlayer&&) noexcept = default;

// ============================================================================
// 初始化与资源管理
// ============================================================================

bool AudioPlayer::Init(int sampleRate, int channels, int framesPerBuffer) {
    if (impl_->initialized) {
        impl_->setError("Init: 已初始化，请先调用 Destroy()");
        return false;
    }

    // 参数校验
    if (sampleRate <= 0) {
        impl_->setError("Init: 无效的 sampleRate (" + std::to_string(sampleRate) + ")");
        return false;
    }
    if (channels <= 0 || channels > 2) {
        impl_->setError("Init: 无效的 channels (" + std::to_string(channels) + ")，仅支持 1 或 2");
        return false;
    }
    if (framesPerBuffer <= 0) {
        impl_->setError("Init: 无效的 framesPerBuffer (" + std::to_string(framesPerBuffer) + ")");
        return false;
    }

    // 初始化 PortAudio（引用计数）
    {
        std::lock_guard<std::mutex> lock(gPaMutex);
        if (gPaRefCount.fetch_add(1) == 0) {
            PaError err = Pa_Initialize();
            if (err != paNoError) {
                gPaRefCount.fetch_sub(1);
                impl_->setError("Init: Pa_Initialize 失败 - "
                                + std::string(Pa_GetErrorText(err)));
                return false;
            }
        }
    }

    // 保存配置
    impl_->sample_rate       = sampleRate;
    impl_->channels          = channels;
    impl_->frames_per_buffer = framesPerBuffer;

    // 打开默认输出流
    PaStreamParameters outputParams;
    std::memset(&outputParams, 0, sizeof(outputParams));
    outputParams.device           = Pa_GetDefaultOutputDevice();
    if (outputParams.device == paNoDevice) {
        impl_->setError("Init: 无可用音频输出设备");
        // 回退引用计数
        {
            std::lock_guard<std::mutex> lock(gPaMutex);
            if (--gPaRefCount == 0) {
                Pa_Terminate();
            }
        }
        return false;
    }
    outputParams.channelCount      = channels;
    outputParams.sampleFormat      = paFloat32;
    outputParams.suggestedLatency  = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(
        &impl_->stream,
        nullptr,                        // 无输入
        &outputParams,                  // 输出参数
        sampleRate,
        framesPerBuffer,
        paClipOff,                      // 不裁剪（我们已确保数据在 [-1,1]）
        &Impl::paCallback,              // 回调函数
        impl_.get());                   // 用户数据

    if (err != paNoError) {
        impl_->setError("Init: Pa_OpenStream 失败 - "
                        + std::string(Pa_GetErrorText(err)));
        {
            std::lock_guard<std::mutex> lock(gPaMutex);
            if (--gPaRefCount == 0) {
                Pa_Terminate();
            }
        }
        impl_->stream = nullptr;
        return false;
    }

    impl_->state.store(AudioPlayerState::IDLE, std::memory_order_release);
    impl_->initialized = true;
    return true;
}

void AudioPlayer::Destroy() {
    if (!impl_->initialized) return;

    // 停止流
    if (impl_->stream) {
        if (Pa_IsStreamActive(impl_->stream)) {
            Pa_AbortStream(impl_->stream);
        }
        Pa_CloseStream(impl_->stream);
        impl_->stream = nullptr;
    }

    // 终止 PortAudio（引用计数）
    {
        std::lock_guard<std::mutex> lock(gPaMutex);
        if (--gPaRefCount == 0) {
            Pa_Terminate();
        }
    }

    impl_->state.store(AudioPlayerState::IDLE, std::memory_order_release);
    impl_->consumed_frames.store(0, std::memory_order_release);
    impl_->read_frame_pos     = 0;
    impl_->total_frames       = 0;
    impl_->data_loaded        = false;
    impl_->audio_data.clear();
    impl_->total_paused_duration.store(0.0, std::memory_order_release);
    impl_->last_dac_time.store(0.0, std::memory_order_release);
    impl_->stream_start_time  = 0.0;
    impl_->pause_start_time   = 0.0;
    impl_->initialized        = false;
}

bool AudioPlayer::IsInitialized() const {
    return impl_->initialized;
}

// ============================================================================
// 音频数据加载
// ============================================================================

bool AudioPlayer::LoadAudio(const float* samples, int numSamples, int channels) {
    if (!impl_->initialized) {
        impl_->setError("LoadAudio: 未初始化");
        return false;
    }
    if (!samples || numSamples <= 0 || channels <= 0) {
        impl_->setError("LoadAudio: 参数无效");
        return false;
    }
    if (channels != impl_->channels) {
        impl_->setError("LoadAudio: 声道数不匹配，需要 " +
                        std::to_string(impl_->channels) + "，传入 " +
                        std::to_string(channels));
        return false;
    }

    impl_->audio_data.assign(samples, samples + numSamples);
    impl_->total_frames = numSamples / channels;
    impl_->data_loaded  = true;

    // 重置播放位置
    impl_->read_frame_pos      = 0;
    impl_->consumed_frames.store(0, std::memory_order_release);
    impl_->state.store(AudioPlayerState::STOPPED, std::memory_order_release);

    return true;
}

bool AudioPlayer::LoadAudio(const std::vector<float>& samples, int channels) {
    return LoadAudio(samples.data(), static_cast<int>(samples.size()), channels);
}

// ============================================================================
// 播放控制
// ============================================================================

bool AudioPlayer::Play() {
    if (!impl_->initialized) {
        impl_->setError("Play: 未初始化");
        return false;
    }
    if (!impl_->data_loaded) {
        impl_->setError("Play: 未加载音频数据");
        return false;
    }

    AudioPlayerState curState = impl_->state.load(std::memory_order_acquire);
    if (curState == AudioPlayerState::PLAYING) {
        return true;  // 已经在播放
    }

    // 如果是已停止或播放完毕，重置读取位置
    if (curState == AudioPlayerState::STOPPED ||
        curState == AudioPlayerState::IDLE ||
        curState == AudioPlayerState::FINISHED) {
        impl_->read_frame_pos = 0;
        impl_->consumed_frames.store(0, std::memory_order_release);
        impl_->total_paused_duration.store(0.0, std::memory_order_release);
    }

    impl_->state.store(AudioPlayerState::PLAYING, std::memory_order_release);

    PaError err = Pa_StartStream(impl_->stream);
    if (err != paNoError) {
        impl_->setError("Play: Pa_StartStream 失败 - "
                        + std::string(Pa_GetErrorText(err)));
        impl_->state.store(AudioPlayerState::STOPPED, std::memory_order_release);
        return false;
    }

    impl_->stream_start_time = Pa_GetStreamTime(impl_->stream);

    return true;
}

bool AudioPlayer::Pause() {
    if (!impl_->initialized) {
        impl_->setError("Pause: 未初始化");
        return false;
    }

    AudioPlayerState curState = impl_->state.load(std::memory_order_acquire);
    if (curState != AudioPlayerState::PLAYING) {
        impl_->setError("Pause: 当前不在播放状态");
        return false;
    }

    impl_->state.store(AudioPlayerState::PAUSED, std::memory_order_release);
    impl_->pause_start_time = Pa_GetStreamTime(impl_->stream);

    // 停止流（阻止回调，保持位置）
    PaError err = Pa_StopStream(impl_->stream);
    if (err != paNoError) {
        impl_->setError("Pause: Pa_StopStream 失败 - "
                        + std::string(Pa_GetErrorText(err)));
        // 即使停止失败也保持暂停状态
    }

    return true;
}

bool AudioPlayer::Resume() {
    if (!impl_->initialized) {
        impl_->setError("Resume: 未初始化");
        return false;
    }

    AudioPlayerState curState = impl_->state.load(std::memory_order_acquire);
    if (curState != AudioPlayerState::PAUSED) {
        impl_->setError("Resume: 当前不在暂停状态");
        return false;
    }

    // 记录暂停时长
    PaTime now = Pa_GetStreamTime(impl_->stream);
    double pauseDuration = now - impl_->pause_start_time;
    impl_->total_paused_duration.store(
        impl_->total_paused_duration.load(std::memory_order_acquire) + pauseDuration,
        std::memory_order_release);

    impl_->state.store(AudioPlayerState::PLAYING, std::memory_order_release);
    impl_->stream_start_time = now;

    PaError err = Pa_StartStream(impl_->stream);
    if (err != paNoError) {
        impl_->setError("Resume: Pa_StartStream 失败 - "
                        + std::string(Pa_GetErrorText(err)));
        impl_->state.store(AudioPlayerState::PAUSED, std::memory_order_release);
        return false;
    }

    return true;
}

bool AudioPlayer::Stop() {
    if (!impl_->initialized) {
        impl_->setError("Stop: 未初始化");
        return false;
    }

    AudioPlayerState curState = impl_->state.load(std::memory_order_acquire);
    if (curState == AudioPlayerState::IDLE || curState == AudioPlayerState::STOPPED) {
        return true;
    }

    // 停止流
    if (impl_->stream) {
        if (Pa_IsStreamActive(impl_->stream)) {
            PaError err = Pa_StopStream(impl_->stream);
            if (err != paNoError) {
                impl_->setError("Stop: Pa_StopStream 失败 - "
                                + std::string(Pa_GetErrorText(err)));
            }
        }
    }

    // 重置位置到开头
    impl_->read_frame_pos = 0;
    impl_->consumed_frames.store(0, std::memory_order_release);
    impl_->total_paused_duration.store(0.0, std::memory_order_release);
    impl_->last_dac_time.store(0.0, std::memory_order_release);
    impl_->stream_start_time = 0.0;
    impl_->pause_start_time  = 0.0;

    impl_->state.store(AudioPlayerState::STOPPED, std::memory_order_release);
    return true;
}

// ============================================================================
// 状态查询
// ============================================================================

AudioPlayerState AudioPlayer::GetState() const {
    return impl_->state.load(std::memory_order_acquire);
}

bool AudioPlayer::IsPlaying() const {
    return impl_->state.load(std::memory_order_acquire) == AudioPlayerState::PLAYING;
}

bool AudioPlayer::IsPaused() const {
    return impl_->state.load(std::memory_order_acquire) == AudioPlayerState::PAUSED;
}

bool AudioPlayer::IsStopped() const {
    AudioPlayerState s = impl_->state.load(std::memory_order_acquire);
    return s == AudioPlayerState::STOPPED || s == AudioPlayerState::IDLE;
}

bool AudioPlayer::IsFinished() const {
    return impl_->state.load(std::memory_order_acquire) == AudioPlayerState::FINISHED;
}

// ============================================================================
// 位置与时间查询
// ============================================================================

int64_t AudioPlayer::GetConsumedFrames() const {
    return impl_->consumed_frames.load(std::memory_order_acquire);
}

double AudioPlayer::GetPlaybackPositionMs() const {
    int64_t frames = impl_->consumed_frames.load(std::memory_order_acquire);
    if (impl_->sample_rate <= 0) return 0.0;
    return static_cast<double>(frames) / impl_->sample_rate * 1000.0;
}

double AudioPlayer::GetDacTimeMs() const {
    if (!impl_->stream) return -1.0;

    PaTime dacTime = impl_->last_dac_time.load(std::memory_order_acquire);
    PaTime now     = Pa_GetStreamTime(impl_->stream);

    if (dacTime <= 0.0 || now <= 0.0) return -1.0;

    // DAC 时间相对于当前流的偏差
    double pausedDur = impl_->total_paused_duration.load(std::memory_order_acquire);

    // 有效 DAC 时间 = DAC时间戳 - 流启动时间 - 暂停累计时长
    return (dacTime - impl_->stream_start_time - pausedDur) * 1000.0;
}

double AudioPlayer::GetTotalDurationMs() const {
    if (!impl_->data_loaded || impl_->sample_rate <= 0) return 0.0;
    return static_cast<double>(impl_->total_frames) / impl_->sample_rate * 1000.0;
}

int AudioPlayer::GetSampleRate() const {
    return impl_->sample_rate;
}

int AudioPlayer::GetChannels() const {
    return impl_->channels;
}

// ============================================================================
// 错误处理
// ============================================================================

const char* AudioPlayer::GetLastErrorMsg() const {
    return impl_->last_error_msg.c_str();
}

}  // namespace audio
}  // namespace digital_human
