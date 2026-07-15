#include "core/audio_processor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "audio/audio_framer.h"
#include "audio/audio_vad.h"
#include "audio/audio_preemphasis.h"
#include "audio/audio_rms_normalize.h"
#include "audio/audio_noise_reduction.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_cmvn.h"
#include "audio/audio_ring_buffer.h"

namespace digital_human {
namespace core {

using audio::AudioFramer;
using audio::VoiceActivityDetector;
using audio::PreEmphasis;
using audio::RMSNormalize;
using audio::NoiseReduction;
using audio::MelFeatureExtract;
using audio::CMVN;
using audio::RingBuffer;

// ============================================================================
// Impl 结构体
// ============================================================================

struct AudioProcessor::Impl {
    // ---- 配置 ----
    AudioProcessorConfig config;

    // ---- 数据处理模块（独立实例，线程安全） ----
    NoiseReduction      noise_reduction;
    AudioFramer         framer;
    VoiceActivityDetector vad;
    PreEmphasis         pre_emphasis;
    RMSNormalize        rms_normalize;
    MelFeatureExtract   mel_extract;
    CMVN                cmvn;

    // ---- 数据源 ----
    const float*        fixed_source_   = nullptr;
    size_t              fixed_samples_  = 0;
    bool                has_fixed_source_ = false;
    RingBuffer*         ring_buffer_    = nullptr;
    bool                has_ring_buffer_ = false;

    // ---- 输出 ----
    ThreadSafeQueue<MelFeaturePacket>* output_queue_ = nullptr;

    // ---- 滑动窗口 ----
    std::vector<float>  window_;            ///< 滑动窗口累积数据
    size_t              max_window_size_ = 0;  ///< 窗口最大容量

    // ---- 处理状态 ----
    int64_t             read_cursor_    = 0;   ///< 已处理的 sample 位置
    int64_t             output_count_   = 0;   ///< 累计输出 mel 帧数

    // ---- 文件模式下的顺序推进 ----
    size_t              file_read_pos_  = 0;   ///< 文件读取位置（sample）

    // ---- 结束标记 ----
    bool                eos_marked_     = false;

    // ---- Mel 配置缓存（避免多次构造） ----
    audio::MelConfig    mel_config;
    bool                mel_config_dirty_ = true;

    // ---- 立体声混合缓冲区（避免重复分配） ----
    std::vector<float>  mix_buffer_;

    // ========================================================================
    // 内部方法
    // ========================================================================

    /// @brief 更新 Mel 配置
    void UpdateMelConfig() {
        mel_config.nFFT       = config.nfft;
        mel_config.nMels      = config.mel_bins;
        mel_config.sampleRate = config.sample_rate;
        mel_config.fMin       = 0.0f;
        mel_config.fMax       = 8000.0f;
        mel_config_dirty_     = false;
    }

    /// @brief 初始化滑动窗口
    void InitWindow() {
        max_window_size_ = static_cast<size_t>(config.window_capacity);
        window_.reserve(max_window_size_);
        window_.clear();
    }

    /// @brief 从数据源读取新数据到滑动窗口
    /// @return 本次读取的 sample 数
    size_t FillWindow() {
        if (window_.size() >= max_window_size_) {
            return 0;  // 窗口已满，等待消费
        }

        size_t space = max_window_size_ - window_.size();

        if (has_fixed_source_) {
            // 文件模式：从固定缓冲区拷贝
            size_t remaining = fixed_samples_ - file_read_pos_;
            size_t to_read = std::min(space, remaining);

            if (to_read > 0) {
                window_.insert(window_.end(),
                               fixed_source_ + file_read_pos_,
                               fixed_source_ + file_read_pos_ + to_read);
                file_read_pos_ += to_read;
            }
            return to_read;
        }

        if (has_ring_buffer_) {
            // 流式模式：从 RingBuffer 读取
            // 临时缓冲区避免多次 small read
            std::vector<float> temp(space);
            size_t read = ring_buffer_->read(temp.data(), space);
            if (read > 0) {
                window_.insert(window_.end(), temp.begin(), temp.begin() + read);
            }
            return read;
        }

        return 0;
    }

    /// @brief 从滑动窗口中提取一帧并滑窗
    /// @param[out] frame 输出帧（等长 frame_size 的 vector）
    /// @return true  成功提取一帧
    /// @return false 窗口内数据不足一帧
    bool NextFrame(std::vector<float>& frame) {
        if (window_.size() < static_cast<size_t>(config.frame_size)) {
            return false;
        }

        // 提取前 frame_size 个 sample
        frame.assign(window_.begin(),
                     window_.begin() + config.frame_size);

        // 滑动窗口：丢弃 hop_size 个 sample
        window_.erase(window_.begin(),
                      window_.begin() + config.hop_size);

        read_cursor_ += config.hop_size;
        return true;
    }

    /// @brief 处理一帧音频 → Mel 特征包
    MelFeaturePacket ProcessOneFrame(const std::vector<float>& frame,
                                     int64_t pts_ms, int64_t seq_id) {
        MelFeaturePacket result;
        result.header.pts_ms = pts_ms;
        result.header.seq_id = seq_id;
        result.header.status = StatusCode::OK;

        try {
            // 1. 降噪
            auto denoised = noise_reduction.process(frame, config.sample_rate);

            // 2. RMS 归一化
            auto normalized = rms_normalize.process(denoised);

            // 3. 预加重
            auto emphasized = pre_emphasis.process(normalized);

            // 4. 分帧（本帧已是一个窗口，但 framer 负责加窗和重叠）
            audio::FrameConfig frame_cfg;
            frame_cfg.frameSize = config.frame_size;
            frame_cfg.hopSize   = config.hop_size;
            auto frames = framer.frame(emphasized, frame_cfg);

            if (frames.empty() || frames[0].empty()) {
                result.header.status = StatusCode::SKIP;
                return result;
            }

            // 5. VAD 过滤
            auto voiced = vad.filter(frames);

            // 6. Mel 特征提取
            if (mel_config_dirty_) {
                UpdateMelConfig();
            }
            auto mel = mel_extract.extract(
                voiced.empty() ? frames : voiced, mel_config);

            if (mel.empty()) {
                result.header.status = StatusCode::SKIP;
                return result;
            }

            // 7. CMVN 归一化
            result.payload = cmvn.process(mel);
            result.header.status = StatusCode::OK;

        } catch (const std::exception& e) {
            std::cerr << "[AudioProcessor] 帧处理异常: " << e.what() << std::endl;
            result.header.status = StatusCode::ERROR;
        }

        return result;
    }

    /// @brief 将多声道混合为单声道
    void MixToMono(const float* src, size_t frames,
                   size_t ch, std::vector<float>& mono) {
        if (ch <= 1) {
            mono.assign(src, src + frames);
            return;
        }
        mono.resize(frames);
        for (size_t i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (size_t c = 0; c < ch; ++c) {
                sum += src[i * ch + c];
            }
            mono[i] = sum / static_cast<float>(ch);
        }
    }
};

// ============================================================================
// 构造 / 析构
// ============================================================================

AudioProcessor::AudioProcessor(const std::string& name)
    : ThreadBase(name)
    , impl_(std::make_unique<Impl>()) {}

AudioProcessor::~AudioProcessor() {
    Stop();
}

// ============================================================================
// 配置
// ============================================================================

void AudioProcessor::SetConfig(const AudioProcessorConfig& config) {
    impl_->config = config;
    impl_->max_window_size_ = static_cast<size_t>(config.window_capacity);
    impl_->mel_config_dirty_ = true;
}

const AudioProcessorConfig& AudioProcessor::GetConfig() const {
    return impl_->config;
}

// ============================================================================
// 数据源
// ============================================================================

void AudioProcessor::SetAudioSource(const float* data, size_t samples,
                                     int rate, int ch) {
    if (!data || samples == 0) return;

    // 多声道混合为 mono
    if (ch > 1) {
        impl_->MixToMono(data, samples / ch, ch, impl_->mix_buffer_);
        impl_->fixed_source_  = impl_->mix_buffer_.data();
        impl_->fixed_samples_ = impl_->mix_buffer_.size();
    } else {
        impl_->fixed_source_  = data;
        impl_->fixed_samples_ = samples;
    }

    impl_->has_fixed_source_ = true;
    impl_->config.AutoConfigure(rate);
    impl_->InitWindow();
    impl_->file_read_pos_ = 0;
    impl_->read_cursor_   = 0;
    impl_->output_count_  = 0;
    impl_->eos_marked_    = false;
    impl_->mel_config_dirty_ = true;
}

void AudioProcessor::SetRingBuffer(audio::RingBuffer* buffer) {
    impl_->ring_buffer_    = buffer;
    impl_->has_ring_buffer_ = true;
    impl_->InitWindow();
    impl_->read_cursor_   = 0;
    impl_->output_count_  = 0;
    impl_->eos_marked_    = false;
    impl_->mel_config_dirty_ = true;
}

// ============================================================================
// 输出队列
// ============================================================================

void AudioProcessor::SetOutputQueue(
    ThreadSafeQueue<MelFeaturePacket>* queue) {
    impl_->output_queue_ = queue;
}

// ============================================================================
// 线程主循环
// ============================================================================

void AudioProcessor::Run() {
    LogInfo("[AudioProcessor] 启动");

    if (!impl_->output_queue_) {
        LogError("未设置输出队列");
        return;
    }

    // 初始化 Mel 配置
    if (impl_->mel_config_dirty_) {
        impl_->UpdateMelConfig();
    }

    const int kPopTimeoutMs = 100;  // 空闲检测周期

    while (!IsStopping()) {
        // ---- 步骤 1: 从数据源补充窗口 ----
        size_t filled = impl_->FillWindow();

        // ---- 步骤 2: 检查数据是否耗尽 ----
        bool data_done = false;
        if (impl_->has_fixed_source_) {
            data_done = (impl_->file_read_pos_ >= impl_->fixed_samples_);
        } else if (impl_->has_ring_buffer_) {
            // RingBuffer 模式：无数据且 EOS 标记
            data_done = (filled == 0 && impl_->eos_marked_);
        }

        // ---- 步骤 3: 提取并处理帧 ----
        bool processed_any = false;
        std::vector<float> frame;
        frame.reserve(impl_->config.frame_size);

        while (impl_->NextFrame(frame)) {
            processed_any = true;

            // 计算 PTS（毫秒）= 当前 cursor 位置 / 采样率 * 1000
            int64_t pts_ms = static_cast<int64_t>(
                static_cast<double>(impl_->read_cursor_)
                / impl_->config.sample_rate * 1000.0);
            int64_t seq_id = impl_->output_count_;

            // 执行音频特征提取
            auto mel_pkt = impl_->ProcessOneFrame(frame, pts_ms, seq_id);
            impl_->output_count_++;

            if (mel_pkt.header.IsOK()) {
                mel_pkt.header.cost_ms = mel_pkt.header.cost_ms;
                impl_->output_queue_->Push(std::move(mel_pkt));
            }
            // SKIP/ERROR 包直接丢弃（不影响流水线）

            // 每处理 10 帧输出一次日志
            if (seq_id % 10 == 0) {
                LogInfo("[AudioProcessor] 已处理 " +
                        std::to_string(seq_id) + " 帧, pts=" +
                        std::to_string(pts_ms) + "ms");
            }
        }

        // ---- 步骤 4: 数据耗尽 → 发送 EOS ----
        if (data_done && !processed_any) {
            if (impl_->window_.empty() ||
                impl_->window_.size() < static_cast<size_t>(impl_->config.frame_size)) {
                LogInfo("[AudioProcessor] 音频数据耗尽，发送 EOS");
                impl_->output_queue_->Push(MelFeaturePacket::EOS());
                break;
            }
        }

        // ---- 步骤 5: 无数据时休眠 ----
        if (!processed_any && filled == 0) {
            // RingBuffer 流式模式：等待新数据
            if (impl_->has_ring_buffer_ && !impl_->eos_marked_) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(kPopTimeoutMs));
            } else if (impl_->has_fixed_source_ && data_done) {
                // 文件模式且全部读完 → 退出
                impl_->output_queue_->Push(MelFeaturePacket::EOS());
                break;
            } else if (impl_->has_fixed_source_) {
                // 文件模式窗口未满，继续填充
                continue;
            }
        }
    }

    LogInfo("[AudioProcessor] 退出 (输出 " +
            std::to_string(impl_->output_count_) + " 帧)");
}

// ============================================================================
// 控制
// ============================================================================

void AudioProcessor::MarkEOS() {
    impl_->eos_marked_ = true;
}

void AudioProcessor::Reset() {
    impl_->window_.clear();
    impl_->read_cursor_   = 0;
    impl_->output_count_  = 0;
    impl_->file_read_pos_ = 0;
    impl_->eos_marked_    = false;
}

// ============================================================================
// 状态查询
// ============================================================================

int64_t AudioProcessor::GetProcessedSamples() const {
    return impl_->read_cursor_;
}

double AudioProcessor::GetProcessedDurationMs() const {
    if (impl_->config.sample_rate <= 0) return 0.0;
    return static_cast<double>(impl_->read_cursor_)
           / impl_->config.sample_rate * 1000.0;
}

int64_t AudioProcessor::GetPendingFrames() const {
    if (impl_->window_.size() < static_cast<size_t>(impl_->config.frame_size)) {
        return 0;
    }
    return (impl_->window_.size() - impl_->config.frame_size)
           / impl_->config.hop_size + 1;
}

int64_t AudioProcessor::GetOutputCount() const {
    return impl_->output_count_;
}

double AudioProcessor::GetProgress() const {
    if (!impl_->has_fixed_source_ || impl_->fixed_samples_ == 0) {
        return 0.0;
    }
    return static_cast<double>(impl_->file_read_pos_)
           / static_cast<double>(impl_->fixed_samples_);
}

}  // namespace core
}  // namespace digital_human
