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
    ThreadSafeQueue<AudioRawPacket>* input_queue_ = nullptr;
    bool                has_input_queue_ = false;

    // ---- 输出 ----
    ThreadSafeQueue<MelFeaturePacket>* output_queue_ = nullptr;

    // ---- 环形滑动窗口（O(1) 入队/出队，消除 erase 的 O(N) 数据移动） ----
    std::vector<float>  ring_buf_;          ///< 环形缓冲区（固定大小）
    size_t              ring_capacity_ = 0; ///< 缓冲区总容量
    size_t              ring_head_ = 0;     ///< 有效数据起始索引
    size_t              ring_size_ = 0;     ///< 有效数据长度

    // ---- 处理状态 ----
    int64_t             read_cursor_    = 0;   ///< 已处理的 sample 位置
    int64_t             output_count_   = 0;   ///< 累计输出 mel 帧数

    // ---- 文件模式下的顺序推进 ----
    size_t              file_read_pos_  = 0;   ///< 文件读取位置（sample）

    // ---- 队列模式下的待消费缓冲（防止大包跨 FillWindow 丢样本） ----
    std::vector<float>  pending_audio_;          ///< 当前包尚未消费的样本
    size_t              pending_offset_ = 0;     ///< pending_audio_ 中下一个待消费位置

    // ---- 结束标记 ----
    std::atomic<bool> eos_marked_{false};

    // ---- Mel 配置缓存（避免多次构造） ----
    audio::MelConfig    mel_config;
    bool                mel_config_dirty_ = true;

    // ---- 立体声混合缓冲区（避免重复分配） ----
    std::vector<float>  mix_buffer_;

    // ---- 流式 RMS 归一化状态（utterance 级近似的 EMA 估计） ----
    // 逐帧独立 RMS 归一化会把每一帧都拉到相同响度，摧毁音量动态，
    // 进而破坏口型开合与音频能量的相关性。改为 EMA 估计全局 RMS，
    // 用缓慢收敛的增益做归一化，保留帧间相对响度。
    double              running_rms_    = 0.0;
    bool                has_running_rms_ = false;
    static constexpr double kRmsEmaAlpha = 0.05;   ///< EMA 平滑因子（~20帧收敛）
    static constexpr float  kTargetRms    = 0.056f; ///< 目标 RMS（与参考实现一致）

    // ========================================================================
    // 内部方法
    // ========================================================================

    /// @brief 更新 Mel 配置（对齐 Wav2Lip 官方预处理参数）
    void UpdateMelConfig() {
        mel_config.nFFT       = config.nfft;
        mel_config.nMels      = config.mel_bins;
        mel_config.sampleRate = config.sample_rate;
        // Wav2Lip 官方：fmin=55, fmax=7600（旧值 0/8000 会让模型输入分布偏移）
        mel_config.fMin       = 55.0f;
        mel_config.fMax       = 7600.0f;
        // 窗长与 nFFT 一致（Wav2Lip: win_size=n_fft=800）
        mel_config.winSize    = config.nfft;
        // Wav2Lip symmetric 归一化参数
        mel_config.refLevelDb = 20.0f;
        mel_config.minLevelDb = -100.0f;
        mel_config.maxAbsNorm = 4.0f;
        mel_config_dirty_     = false;
    }

    /// @brief 初始化环形滑动窗口
    void InitWindow() {
        ring_capacity_ = static_cast<size_t>(config.window_capacity);
        ring_buf_.resize(ring_capacity_, 0.0f);
        ring_head_ = 0;
        ring_size_ = 0;
    }

    /// @brief 从数据源读取新数据到环形滑动窗口
    /// @return 本次读取的 sample 数
    size_t FillWindow() {
        if (ring_size_ >= ring_capacity_) {
            return 0;  // 环形窗口已满，等待消费
        }

        size_t space = ring_capacity_ - ring_size_;

        if (has_fixed_source_) {
            // 文件模式：从固定缓冲区拷贝
            size_t remaining = fixed_samples_ - file_read_pos_;
            size_t to_read = std::min(space, remaining);

            if (to_read > 0) {
                AppendToRing(fixed_source_ + file_read_pos_, to_read);
                file_read_pos_ += to_read;
            }
            return to_read;
        }

        if (has_ring_buffer_) {
            // 流式模式：从 RingBuffer 读取
            std::vector<float> temp(space);
            size_t read = ring_buffer_->read(temp.data(), space);
            if (read > 0) {
                AppendToRing(temp.data(), read);
            }
            return read;
        }

        if (has_input_queue_) {
            // 队列模式：优先消费 pending_audio_ 中尚未用完的样本，
            // 避免大包在窗口空间不足时丢失剩余样本
            size_t total_filled = 0;

            while (space > 0) {
                // 1) 先消费 pending_audio_ 中的剩余样本
                if (pending_offset_ < pending_audio_.size()) {
                    size_t avail = pending_audio_.size() - pending_offset_;
                    size_t to_copy = std::min(space, avail);
                    AppendToRing(pending_audio_.data() + pending_offset_,
                                 to_copy);
                    pending_offset_ += to_copy;
                    space           -= to_copy;
                    total_filled    += to_copy;

                    // pending 已全部消费，清空缓冲
                    if (pending_offset_ >= pending_audio_.size()) {
                        pending_audio_.clear();
                        pending_audio_.shrink_to_fit();
                        pending_offset_ = 0;
                    }
                    continue;
                }

                // 2) pending 已空，从队列取下一个包
                AudioRawPacket pkt;
                if (!input_queue_->TryPop(pkt)) {
                    break;  // 队列暂时无数据
                }
                if (pkt.header.IsTerminal() || pkt.payload.empty()) {
                    // 终止包/空包交由 Run() 主循环处理
                    // 注意：终止包不能丢弃，需重新放回队列让 Run 看到
                    // 但 TryPop 已取出，这里改为：标记 EOS 由 MarkEOS 逻辑消费
                    // 实际：终止包由 AudioProcessor::Run 通过队列 Stop 状态感知
                    // 此处仅跳过当前包，不再 push 回去（避免死循环）
                    continue;
                }

                // 取出新包：全部存入 pending，本轮只消费 space 数量
                pending_audio_   = std::move(pkt.payload);
                pending_offset_  = 0;
                // 循环回到步骤 1 继续消费
            }

            return total_filled;
        }

        return 0;
    }

    /// @brief 环形写入：将 data 的前 n 个元素追加到环形缓冲区尾部
    void AppendToRing(const float* data, size_t n) {
        if (n == 0 || ring_capacity_ == 0) return;
        size_t tail = (ring_head_ + ring_size_) % ring_capacity_;
        size_t first = std::min(n, ring_capacity_ - tail);
        std::memcpy(&ring_buf_[tail], data, first * sizeof(float));
        if (first < n) {
            std::memcpy(&ring_buf_[0], data + first, (n - first) * sizeof(float));
        }
        ring_size_ += n;
    }

    /// @brief 从环形窗口中提取一帧并滑窗（O(1)，无数据移动）
    /// @param[out] frame 输出帧（等长 frame_size 的 vector）
    /// @return true  成功提取一帧
    /// @return false 窗口内数据不足一帧
    bool NextFrame(std::vector<float>& frame) {
        if (ring_size_ < static_cast<size_t>(config.frame_size)) {
            return false;
        }

        // 从 ring_head_ 处读取 frame_size 个元素
        size_t frame_size = static_cast<size_t>(config.frame_size);
        frame.resize(frame_size);

        if (ring_head_ + frame_size <= ring_capacity_) {
            // 无需绕环
            std::memcpy(frame.data(), &ring_buf_[ring_head_],
                        frame_size * sizeof(float));
        } else {
            // 分两段（尾部 + 头部）
            size_t first = ring_capacity_ - ring_head_;
            std::memcpy(frame.data(), &ring_buf_[ring_head_],
                        first * sizeof(float));
            std::memcpy(frame.data() + first, &ring_buf_[0],
                        (frame_size - first) * sizeof(float));
        }

        // 滑动窗口：前进 hop_size（O(1)，仅移动索引）
        ring_head_ = (ring_head_ + static_cast<size_t>(config.hop_size)) % ring_capacity_;
        ring_size_ -= static_cast<size_t>(config.hop_size);
        read_cursor_ += config.hop_size;
        return true;
    }

    /// @brief 流式 RMS 归一化：EMA 估计全局 RMS，增益缓慢收敛，保留响度动态
    void RunningRmsNormalize(std::vector<float>& frame) {
        if (frame.empty()) return;
        double sum_sq = 0.0;
        for (float s : frame) sum_sq += static_cast<double>(s) * s;
        double frame_rms = std::sqrt(sum_sq / frame.size());

        if (!has_running_rms_) {
            running_rms_ = frame_rms > 1e-6 ? frame_rms : 1e-6;
            has_running_rms_ = true;
        } else if (frame_rms > 1e-6) {
            // 仅在非静音帧上更新估计，避免静音段把估计值拉向 0
            running_rms_ = kRmsEmaAlpha * frame_rms
                         + (1.0 - kRmsEmaAlpha) * running_rms_;
        }

        double gain = (running_rms_ > 1e-6)
                          ? static_cast<double>(kTargetRms) / running_rms_
                          : 1.0;
        // 限幅防止静音/爆音帧的极端增益
        gain = std::clamp(gain, 0.1, 20.0);
        float g = static_cast<float>(gain);
        for (float& s : frame) s *= g;
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

            // 2. 流式 RMS 归一化（EMA 全局近似，保留响度动态）
            RunningRmsNormalize(denoised);

            // 3. 预加重
            auto emphasized = pre_emphasis.process(denoised);

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

            // 6. Mel 特征提取 — Wav2Lip symmetric 归一化到 [-4,4]：
            //    这是 Wav2Lip 模型推理必需的输入格式。旧实现传 apply_minmax=false
            //    输出原始 dB 值（约 [-100,-20]），导致模型输入分布严重偏移、
            //    对音频几乎无响应（诊断响应比仅 0.4%）。
            if (mel_config_dirty_) {
                UpdateMelConfig();
            }
            auto mel = mel_extract.extract(
                voiced.empty() ? frames : voiced, mel_config,
                /*apply_minmax=*/true);

            if (mel.empty()) {
                result.header.status = StatusCode::SKIP;
                return result;
            }

            // 7. 输出已归一化的 mel 行（1×80，值域 [-4,4]）。
            //    不再做 CMVN —— Wav2Lip symmetric 归一化已是模型期望的最终输入。
            result.payload = mel;
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
    Shutdown();
}

// ============================================================================
// 配置
// ============================================================================

void AudioProcessor::SetConfig(const AudioProcessorConfig& config) {
    impl_->config = config;
    impl_->ring_capacity_ = static_cast<size_t>(config.window_capacity);
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
    impl_->running_rms_      = 0.0;
    impl_->has_running_rms_  = false;
}

void AudioProcessor::SetRingBuffer(audio::RingBuffer* buffer) {
    impl_->ring_buffer_    = buffer;
    impl_->has_ring_buffer_ = true;
    impl_->InitWindow();
    impl_->read_cursor_   = 0;
    impl_->output_count_  = 0;
    impl_->eos_marked_    = false;
    impl_->mel_config_dirty_ = true;
    impl_->running_rms_      = 0.0;
    impl_->has_running_rms_  = false;
}

void AudioProcessor::SetInputQueue(ThreadSafeQueue<AudioRawPacket>* queue) {
    impl_->input_queue_     = queue;
    impl_->has_input_queue_ = true;
    impl_->InitWindow();
    impl_->read_cursor_   = 0;
    impl_->output_count_  = 0;
    impl_->pending_audio_.clear();
    impl_->pending_audio_.shrink_to_fit();
    impl_->pending_offset_ = 0;
    impl_->eos_marked_    = false;
    impl_->mel_config_dirty_ = true;
    impl_->running_rms_      = 0.0;
    impl_->has_running_rms_  = false;
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
            data_done = (filled == 0 && impl_->eos_marked_.load(std::memory_order_acquire));
        } else if (impl_->has_input_queue_) {
            // 队列模式：无数据且 EOS 标记
            data_done = (filled == 0 && impl_->eos_marked_.load(std::memory_order_acquire));
        }

        // ---- 步骤 3: 提取并处理帧 ----
        bool processed_any = false;
        std::vector<float> frame;
        frame.reserve(impl_->config.frame_size);

        while (impl_->NextFrame(frame)) {
            processed_any = true;

            // 计算 PTS（毫秒）= 帧起始时间 = seq × hop。
            // 注意不能用 read_cursor_：NextFrame 已将其前进一个 hop，
            // 直接用会引入 +1hop 的系统性同步偏移。
            int64_t seq_id = impl_->output_count_;
            int64_t pts_ms = static_cast<int64_t>(
                static_cast<double>(seq_id) * impl_->config.hop_size
                / impl_->config.sample_rate * 1000.0);

            // 执行音频特征提取
            auto t0 = std::chrono::steady_clock::now();
            auto mel_pkt = impl_->ProcessOneFrame(frame, pts_ms, seq_id);
            auto t1 = std::chrono::steady_clock::now();
            impl_->output_count_++;

            if (mel_pkt.header.IsOK()) {
                mel_pkt.header.cost_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
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
            if (impl_->ring_size_ < static_cast<size_t>(impl_->config.frame_size)) {
                LogInfo("[AudioProcessor] 音频数据耗尽，发送 EOS");
                impl_->output_queue_->Push(MelFeaturePacket::EOS());
                break;
            }
        }

        // ---- 步骤 5: 无数据时休眠 ----
        if (!processed_any && filled == 0) {
            // RingBuffer 流式模式：等待新数据
            if (impl_->has_ring_buffer_ && !impl_->eos_marked_.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(kPopTimeoutMs));
            } else if (impl_->has_fixed_source_ && data_done) {
                // 文件模式且全部读完 → 退出
                impl_->output_queue_->Push(MelFeaturePacket::EOS());
                break;
            } else if (impl_->has_fixed_source_) {
                // 文件模式窗口未满，继续填充
                continue;
            } else if (impl_->has_input_queue_) {
                if (data_done) {
                    // 队列模式数据耗尽 → 发 EOS 退出
                    LogInfo("[AudioProcessor] 队列数据耗尽，发送 EOS");
                    impl_->output_queue_->Push(MelFeaturePacket::EOS());
                    break;
                }
                // 等待新数据（下轮循环 TryPop）
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(kPopTimeoutMs));
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
    impl_->eos_marked_.store(true, std::memory_order_release);
}

void AudioProcessor::Reset() {
    impl_->ring_head_ = 0;
    impl_->ring_size_ = 0;
    impl_->read_cursor_   = 0;
    impl_->output_count_  = 0;
    impl_->file_read_pos_ = 0;
    impl_->pending_audio_.clear();
    impl_->pending_audio_.shrink_to_fit();
    impl_->pending_offset_ = 0;
    impl_->eos_marked_.store(false, std::memory_order_release);
    impl_->running_rms_      = 0.0;
    impl_->has_running_rms_  = false;
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
    if (impl_->ring_size_ < static_cast<size_t>(impl_->config.frame_size)) {
        return 0;
    }
    return (impl_->ring_size_ - impl_->config.frame_size)
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
