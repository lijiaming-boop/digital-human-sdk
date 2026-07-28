#include "audio/audio_noise_reduction.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

namespace digital_human {
namespace audio {

struct NoiseReduction::Impl {
    int noiseFrames;
    float oversubtraction;

    static constexpr int kFrameSize = 512;
    static constexpr int kHopSize = 128;
    static constexpr float kFloor = 0.01f;

    // ---- 预分配缓冲区（避免每帧堆分配） ----
    int last_padded_len_ = 0;            ///< 上次 pad 长度
    int last_n_freq_bins_ = 0;           ///< 上次频点数
    int last_spectra_size_ = 0;          ///< 噪声估计帧的复数频谱缓冲大小
    std::vector<float> padded_buf_;      ///< 填充后数据
    std::vector<float> output_buf_;      ///< overlap-add 输出
    std::vector<float> weight_sum_buf_;  ///< 重叠权重和
    std::vector<float> noise_mag_buf_;   ///< 噪声估计
    std::vector<float> window_buf_;      ///< 窗函数（只构造一次）

    // 只缓存估计噪声所需的前 noiseFrames 帧。其余帧边 FFT 边合成，
    // 避免内存占用随整段音频时长线性增长。每个 bin 保存 real/imag。
    std::vector<float> initial_spectra_buf_;
    std::vector<float> current_spectrum_buf_;

    // OpenCV DFT 工作矩阵复用，避免每帧重新分配。
    cv::Mat fft_input_;
    cv::Mat fft_result_;
    cv::Mat complex_spec_;
    cv::Mat time_output_;

    /// @brief 确保预分配缓冲区尺寸匹配
    void EnsureBuffers(int padded_len, int cached_frames, int n_freq_bins) {
        // 窗函数与音频内容无关，只构造一次。
        if (window_buf_.empty()) {
            window_buf_.resize(kFrameSize);
            for (int i = 0; i < kFrameSize; i++) {
                window_buf_[i] = 0.54f - 0.46f * std::cos(
                    2.0f * 3.14159265f * i / (kFrameSize - 1));
            }
        }

        if (padded_len != last_padded_len_) {
            padded_buf_.resize(padded_len);
            output_buf_.resize(padded_len);
            weight_sum_buf_.assign(padded_len, 0.0f);

            // overlap-add 归一化权重只取决于长度和窗函数。
            // 同长度的后续调用直接复用，避免每次重复累加。
            const int total_frames =
                (padded_len - kFrameSize) / kHopSize + 1;
            for (int f = 0; f < total_frames; ++f) {
                for (int j = 0; j < kFrameSize; ++j) {
                    const int idx = f * kHopSize + j;
                    const float w = window_buf_[j];
                    weight_sum_buf_[idx] += w * w;
                }
            }
            last_padded_len_ = padded_len;
        }
        if (n_freq_bins != last_n_freq_bins_) {
            noise_mag_buf_.resize(n_freq_bins);
            current_spectrum_buf_.resize(2 * n_freq_bins);
            last_n_freq_bins_ = n_freq_bins;
        }
        const int spectra_size = 2 * cached_frames * n_freq_bins;
        if (spectra_size != last_spectra_size_) {
            initial_spectra_buf_.resize(spectra_size);
            last_spectra_size_ = spectra_size;
        }

        fft_input_.create(1, kFrameSize, CV_32F);
        fft_result_.create(1, kFrameSize, CV_32FC2);
        complex_spec_.create(1, kFrameSize, CV_32FC2);
        time_output_.create(1, kFrameSize, CV_32F);
    }

    float* InitialSpectrum(int frame) {
        return initial_spectra_buf_.data()
            + frame * 2 * last_n_freq_bins_;
    }

    std::vector<float> process(const std::vector<float>& pcm, int sampleRate) {
        (void)sampleRate;
        if (pcm.empty()) return {};

        int N = static_cast<int>(pcm.size());
        int nFreqBins = kFrameSize / 2 + 1;

        // Frame the signal
        int numFrames = std::max(1, (N - kFrameSize + kHopSize - 1) / kHopSize + 1);

        // Pad to full frames
        int paddedLen = (numFrames - 1) * kHopSize + kFrameSize;
        int noiseCount = std::min(noiseFrames, numFrames);
        EnsureBuffers(paddedLen, noiseCount, nFreqBins);

        // 使用预分配缓冲区
        std::copy(pcm.begin(), pcm.end(), padded_buf_.begin());
        std::fill(padded_buf_.begin() + N, padded_buf_.end(), 0.0f);
        const auto& window = window_buf_;

        std::fill(noise_mag_buf_.begin(), noise_mag_buf_.end(), 0.0f);

        auto analyzeFrame = [&](int frame, float* spectrum,
                                bool accumulate_noise) {
            float* row = fft_input_.ptr<float>(0);
            for (int j = 0; j < kFrameSize; j++) {
                int idx = frame * kHopSize + j;
                row[j] = padded_buf_[idx] * window[j];
            }

            cv::dft(fft_input_, fft_result_, cv::DFT_COMPLEX_OUTPUT);

            const float* fftRow = fft_result_.ptr<float>(0);
            for (int k = 0; k < nFreqBins; k++) {
                float real = fftRow[2 * k];
                float imag = fftRow[2 * k + 1];
                spectrum[2 * k] = real;
                spectrum[2 * k + 1] = imag;
                if (accumulate_noise) {
                    noise_mag_buf_[k] +=
                        std::sqrt(real * real + imag * imag);
                }
            }
        };

        // 噪声模型只需要前 noiseCount 帧。
        for (int f = 0; f < noiseCount; f++) {
            analyzeFrame(f, InitialSpectrum(f), true);
        }

        // Estimate noise from first noiseFrames (average magnitude)
        for (int k = 0; k < nFreqBins; k++) {
            noise_mag_buf_[k] /= noiseCount;
        }

        // IFFT and overlap-add（使用预分配缓冲区）
        std::fill(output_buf_.begin(), output_buf_.end(), 0.0f);

        auto synthesizeFrame = [&](int frame, const float* spectrum) {
            float* cRow = complex_spec_.ptr<float>(0);
            for (int k = 0; k < nFreqBins; k++) {
                const float real = spectrum[2 * k];
                const float imag = spectrum[2 * k + 1];
                const float mag = std::sqrt(real * real + imag * imag);
                const float clean_mag = std::max(
                    mag - oversubtraction * noise_mag_buf_[k],
                    kFloor * mag);
                const float scale = mag > 1e-20f ? clean_mag / mag : 0.0f;

                const float clean_real = real * scale;
                const float clean_imag = imag * scale;
                cRow[2 * k] = clean_real;
                cRow[2 * k + 1] = clean_imag;

                // 实信号的负频率是正频率的共轭。DC 和 Nyquist bin
                // 没有独立的镜像 bin。
                if (k > 0 && k < kFrameSize / 2) {
                    const int mirror = kFrameSize - k;
                    cRow[2 * mirror] = clean_real;
                    cRow[2 * mirror + 1] = -clean_imag;
                }
            }

            cv::idft(complex_spec_, time_output_,
                     cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

            const float* tRow = time_output_.ptr<float>(0);
            for (int j = 0; j < kFrameSize; j++) {
                int idx = frame * kHopSize + j;
                output_buf_[idx] += tRow[j] * window[j];
            }
        };

        for (int f = 0; f < noiseCount; f++) {
            synthesizeFrame(f, InitialSpectrum(f));
        }
        for (int f = noiseCount; f < numFrames; f++) {
            analyzeFrame(f, current_spectrum_buf_.data(), false);
            synthesizeFrame(f, current_spectrum_buf_.data());
        }

        // Normalize by window overlap weights
        std::vector<float> result(pcm.size());
        for (size_t i = 0; i < result.size(); i++) {
            if (weight_sum_buf_[i] > 1e-10f) {
                result[i] = output_buf_[i] / weight_sum_buf_[i];
            } else {
                result[i] = output_buf_[i];
            }
        }

        return result;
    }
};

NoiseReduction::NoiseReduction(int noiseFrames, float oversubtraction)
    : impl_(std::make_unique<Impl>()) {
    impl_->noiseFrames = std::max(1, noiseFrames);
    impl_->oversubtraction = std::max(0.0f, oversubtraction);
}

NoiseReduction::~NoiseReduction() = default;

NoiseReduction::NoiseReduction(NoiseReduction&&) noexcept = default;

NoiseReduction& NoiseReduction::operator=(NoiseReduction&&) noexcept = default;

std::vector<float> NoiseReduction::process(const std::vector<float>& pcm, int sampleRate) {
    return impl_->process(pcm, sampleRate);
}

}  // namespace audio
}  // namespace digital_human
