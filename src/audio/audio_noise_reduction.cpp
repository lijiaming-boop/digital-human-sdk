#include "audio/audio_noise_reduction.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

namespace digital_human {
namespace audio {

struct NoiseReduction::Impl {
    int noiseFrames;
    float oversubtraction;
    float floorFactor;

    static constexpr int kFrameSize = 512;
    static constexpr int kHopSize = 128;
    static constexpr float kFloor = 0.01f;

    // ---- 预分配缓冲区（避免每帧堆分配） ----
    int last_padded_len_ = 0;            ///< 上次 pad 长度
    int last_num_frames_ = 0;            ///< 上次帧数
    int last_n_freq_bins_ = 0;           ///< 上次频点数
    int last_flat_size_ = 0;             ///< 上次频谱扁平化缓冲大小
    std::vector<float> padded_buf_;      ///< 填充后数据
    std::vector<float> output_buf_;      ///< overlap-add 输出
    std::vector<float> weight_sum_buf_;  ///< 重叠权重和
    std::vector<float> noise_mag_buf_;   ///< 噪声估计
    std::vector<float> window_buf_;      ///< 窗函数（只构造一次）

    // 扁平化频谱存储（替代 vector<vector<float>> 嵌套分配）
    std::vector<float> mag_flat_;        ///< 幅度谱扁平化存储 [numFrames * nFreqBins]
    std::vector<float> phase_flat_;      ///< 相位谱扁平化存储 [numFrames * nFreqBins]

    /// @brief 确保预分配缓冲区尺寸匹配
    void EnsureBuffers(int padded_len, int num_frames, int n_freq_bins) {
        if (padded_len != last_padded_len_) {
            padded_buf_.resize(padded_len);
            output_buf_.resize(padded_len);
            weight_sum_buf_.resize(padded_len);
            last_padded_len_ = padded_len;
        }
        if (n_freq_bins != last_n_freq_bins_) {
            noise_mag_buf_.resize(n_freq_bins);
            last_n_freq_bins_ = n_freq_bins;
        }
        int flat_size = num_frames * n_freq_bins;
        if (flat_size != last_flat_size_) {
            mag_flat_.resize(flat_size);
            phase_flat_.resize(flat_size);
            last_flat_size_ = flat_size;
        }
        last_num_frames_ = num_frames;
        // 窗函数只构造一次
        if (window_buf_.empty()) {
            window_buf_.resize(kFrameSize);
            for (int i = 0; i < kFrameSize; i++) {
                window_buf_[i] = 0.54f - 0.46f * std::cos(2.0f * 3.14159265f * i / (kFrameSize - 1));
            }
        }
    }

    /// @brief 获取扁平化频谱的行指针（模拟 [f][k] 访问）
    inline float* MagRow(int f) { return &mag_flat_[f * last_n_freq_bins_]; }
    inline float* PhaseRow(int f) { return &phase_flat_[f * last_n_freq_bins_]; }

    std::vector<float> process(const std::vector<float>& pcm, int sampleRate) {
        (void)sampleRate;
        if (pcm.empty()) return {};

        int N = static_cast<int>(pcm.size());
        int nFreqBins = kFrameSize / 2 + 1;

        // Frame the signal
        int numFrames = std::max(1, (N - kFrameSize + kHopSize - 1) / kHopSize + 1);

        // Pad to full frames
        int paddedLen = (numFrames - 1) * kHopSize + kFrameSize;
        EnsureBuffers(paddedLen, numFrames, nFreqBins);

        // 使用预分配缓冲区
        std::fill(padded_buf_.begin(), padded_buf_.end(), 0.0f);
        std::copy(pcm.begin(), pcm.end(), padded_buf_.begin());
        const auto& window = window_buf_;

        cv::Mat fftInput(1, kFrameSize, CV_32F);
        for (int f = 0; f < numFrames; f++) {
            float* mag_f = MagRow(f);
            float* phase_f = PhaseRow(f);
            float* row = fftInput.ptr<float>(0);
            for (int j = 0; j < kFrameSize; j++) {
                int idx = f * kHopSize + j;
                row[j] = (idx < paddedLen) ? padded_buf_[idx] * window[j] : 0.0f;
            }

            cv::Mat fftResult;
            cv::dft(fftInput, fftResult, cv::DFT_COMPLEX_OUTPUT);

            const float* fftRow = fftResult.ptr<float>(0);
            for (int k = 0; k < nFreqBins; k++) {
                float real = fftRow[2 * k];
                float imag = fftRow[2 * k + 1];
                mag_f[k] = std::sqrt(real * real + imag * imag);
                phase_f[k] = std::atan2(imag, real);
            }
        }

        // Estimate noise from first noiseFrames (average magnitude)
        int noiseCount = std::min(noiseFrames, numFrames);
        std::fill(noise_mag_buf_.begin(), noise_mag_buf_.end(), 0.0f);
        for (int f = 0; f < noiseCount; f++) {
            float* mag_f = MagRow(f);
            for (int k = 0; k < nFreqBins; k++) {
                noise_mag_buf_[k] += mag_f[k];
            }
        }
        for (int k = 0; k < nFreqBins; k++) {
            noise_mag_buf_[k] /= noiseCount;
        }

        // Spectral subtraction
        for (int f = 0; f < numFrames; f++) {
            float* mag_f = MagRow(f);
            for (int k = 0; k < nFreqBins; k++) {
                float cleanMag = mag_f[k] - oversubtraction * noise_mag_buf_[k];
                cleanMag = std::max(cleanMag, kFloor * mag_f[k]);
                mag_f[k] = cleanMag;
            }
        }

        // IFFT and overlap-add（使用预分配缓冲区）
        std::fill(output_buf_.begin(), output_buf_.end(), 0.0f);
        std::fill(weight_sum_buf_.begin(), weight_sum_buf_.end(), 0.0f);

        for (int f = 0; f < numFrames; f++) {
            float* mag_f = MagRow(f);
            float* phase_f = PhaseRow(f);

            // Reconstruct complex spectrum
            cv::Mat complexSpec(1, kFrameSize, CV_32FC2);
            float* cRow = complexSpec.ptr<float>(0);
            for (int k = 0; k < kFrameSize; k++) {
                float mag, phase;
                if (k < nFreqBins) {
                    mag = mag_f[k];
                    phase = phase_f[k];
                } else {
                    int mirror = kFrameSize - k;
                    mag = mag_f[mirror];
                    phase = -phase_f[mirror];
                }
                cRow[2 * k] = mag * std::cos(phase);
                cRow[2 * k + 1] = mag * std::sin(phase);
            }

            cv::Mat timeOut;
            cv::idft(complexSpec, timeOut,
                     cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

            float* tRow = timeOut.ptr<float>(0);
            for (int j = 0; j < kFrameSize; j++) {
                int idx = f * kHopSize + j;
                if (idx < paddedLen) {
                    output_buf_[idx] += tRow[j] * window[j];
                    weight_sum_buf_[idx] += window[j] * window[j];
                }
            }
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
    impl_->floorFactor = 0.01f;
}

NoiseReduction::~NoiseReduction() = default;

NoiseReduction::NoiseReduction(NoiseReduction&&) noexcept = default;

NoiseReduction& NoiseReduction::operator=(NoiseReduction&&) noexcept = default;

std::vector<float> NoiseReduction::process(const std::vector<float>& pcm, int sampleRate) {
    return impl_->process(pcm, sampleRate);
}

}  // namespace audio
}  // namespace digital_human
