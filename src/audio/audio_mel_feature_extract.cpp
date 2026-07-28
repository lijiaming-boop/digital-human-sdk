#include "audio/audio_mel_feature_extract.h"

#include <cmath>
#include <algorithm>

namespace digital_human {
namespace audio {

struct MelFeatureExtract::Impl {
    cv::Mat melFilterbank_;
    MelConfig cachedConfig_;

    // ========================================================================
    // Mel 滤波器组（Slaney area-normalized，对齐 librosa.filters.mel）
    // ========================================================================
    cv::Mat createMelFilterbank(const MelConfig& config) {
        int nFreqBins = config.nFFT / 2 + 1;

        // librosa 使用的 mel 公式（与 hzToMel 一致，Slaney 风格）
        auto hzToMel = [](float hz) {
            const float fMin = 0.0f;
            const float fSp  = 200.0f / 3.0f;
            float mel = (hz - fMin) / fSp;
            float minLogHz = 1000.0f;
            float minLogMel = (minLogHz - fMin) / fSp;
            float logstep = std::log(6.4f) / 27.0f;
            if (hz >= minLogHz) {
                mel = minLogMel + std::log(hz / minLogHz) / logstep;
            }
            return mel;
        };
        auto melToHz = [](float mel) {
            const float fMin = 0.0f;
            const float fSp  = 200.0f / 3.0f;
            float hz = fMin + fSp * mel;
            float minLogHz = 1000.0f;
            float minLogMel = (minLogHz - fMin) / fSp;
            float logstep = std::log(6.4f) / 27.0f;
            if (mel >= minLogMel) {
                hz = minLogHz * std::exp(logstep * (mel - minLogMel));
            }
            return hz;
        };

        float melMin = hzToMel(config.fMin);
        float melMax = hzToMel(config.fMax);

        int nPoints = config.nMels + 2;
        std::vector<float> binPoints(nPoints);
        float melStep = (melMax - melMin) / (config.nMels + 1);
        for (int i = 0; i < nPoints; i++) {
            float hz = melToHz(melMin + i * melStep);
            binPoints[i] = hz * config.nFFT / static_cast<float>(config.sampleRate);
        }

        cv::Mat filterbank = cv::Mat::zeros(config.nMels, nFreqBins, CV_32F);
        for (int m = 0; m < config.nMels; m++) {
            float left = binPoints[m];
            float center = binPoints[m + 1];
            float right = binPoints[m + 2];

            if (std::abs(center - left) < 1e-8f || std::abs(right - center) < 1e-8f) {
                continue;
            }

            float denom_left  = center - left;
            float denom_right = right - center;

            // 累加三角形权重，用于 Slaney area-normalization
            float enorm = 0.0f;
            for (int k = 0; k < nFreqBins; k++) {
                float fk = static_cast<float>(k);
                float w = 0.0f;
                if (fk >= left && fk <= center) {
                    w = (fk - left) / denom_left;
                } else if (fk >= center && fk <= right) {
                    w = (right - fk) / denom_right;
                }
                filterbank.at<float>(m, k) = w;
                enorm += w;
            }

            // Slaney area-normalization：每个滤波器除以其面积，
            // 使各 mel 通道能量一致（对齐 librosa 默认 norm='slaney'）
            if (enorm > 1e-8f) {
                float scale = 2.0f / enorm;
                for (int k = 0; k < nFreqBins; k++) {
                    filterbank.at<float>(m, k) *= scale;
                }
            }
        }

        return filterbank;
    }

    // ========================================================================
    // Hann 窗（对齐 scipy.signal.windows.hann / numpy.hanning）
    // ========================================================================
    std::vector<float> createHannWindow(int N) const {
        // numpy.hanning: 0.5 - 0.5*cos(2*pi*n/(N-1)), n=0..N-1
        std::vector<float> w(N);
        for (int n = 0; n < N; ++n) {
            w[n] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * n / (N - 1));
        }
        return w;
    }

    // ========================================================================
    // 主提取流程
    // ========================================================================
    cv::Mat extract(const std::vector<std::vector<float>>& frames,
                    const MelConfig& config,
                    bool apply_minmax) {
        if (frames.empty()) {
            return cv::Mat();
        }

        if (config.nFFT <= 0 || config.nMels <= 0 || config.sampleRate <= 0) {
            return cv::Mat();
        }

        int numFrames = static_cast<int>(frames.size());
        int nFFT = config.nFFT;
        int nFreqBins = nFFT / 2 + 1;
        int winSize = (config.winSize > 0) ? config.winSize : nFFT;

        // 滤波器缓存判定（含新增字段）
        bool rebuildFilter = melFilterbank_.empty()
            || cachedConfig_.nFFT != config.nFFT
            || cachedConfig_.nMels != config.nMels
            || cachedConfig_.sampleRate != config.sampleRate
            || std::abs(cachedConfig_.fMin - config.fMin) > 1e-6f
            || std::abs(cachedConfig_.fMax - config.fMax) > 1e-6f;

        if (rebuildFilter) {
            melFilterbank_ = createMelFilterbank(config);
            cachedConfig_ = config;
        }

        // Hann 窗（窗长 = winSize，作用于每帧前 winSize 个样本，再零填充到 nFFT）
        std::vector<float> hann = createHannWindow(winSize);

        // Step 1: 加窗 + 零填充到 nFFT
        cv::Mat paddedFrames(numFrames, nFFT, CV_32F, cv::Scalar(0.0f));
        for (int i = 0; i < numFrames; i++) {
            float* row = paddedFrames.ptr<float>(i);
            const auto& frame = frames[i];
            int copyLen = std::min(static_cast<int>(frame.size()), winSize);
            for (int j = 0; j < copyLen; j++) {
                row[j] = frame[j] * hann[j];
            }
        }

        // Step 2: FFT (逐行)
        cv::Mat fftResult;
        cv::dft(paddedFrames, fftResult, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);

        // Step 3: 振幅谱 |D|（Wav2Lip 用振幅谱而非功率谱）
        cv::Mat magSpec(numFrames, nFreqBins, CV_32F);
        for (int i = 0; i < numFrames; i++) {
            float* msRow = magSpec.ptr<float>(i);
            const float* fftRow = fftResult.ptr<float>(i);
            for (int k = 0; k < nFreqBins; k++) {
                float real = fftRow[2 * k];
                float imag = fftRow[2 * k + 1];
                msRow[k] = std::sqrt(real * real + imag * imag);
            }
        }

        // Step 4: 应用 Mel 滤波器组
        // melFilterbank_: (nMels x nFreqBins) * magSpec^T: (nFreqBins x numFrames)
        cv::Mat melEnergies = melFilterbank_ * magSpec.t();
        cv::Mat melSpec = melEnergies.t();  // (numFrames x nMels)

        // Step 5: 振幅 → dB（Wav2Lip: 20*log10(max(min_level, |D|)) - ref_level_db）
        // 注意：Wav2Lip 官方先对 |D| 取 dB，min_level_db 作为下限钳位
        for (int i = 0; i < numFrames; i++) {
            float* row = melSpec.ptr<float>(i);
            for (int m = 0; m < config.nMels; m++) {
                float val = std::max(row[m], 1e-10f);
                row[m] = 20.0f * std::log10(val) - config.refLevelDb;
            }
        }

        // Step 6: 归一化
        if (apply_minmax) {
            // Wav2Lip symmetric 归一化：
            //   mel = 10 * (mel_db - ref_level_db - min_level_db) / -min_level_db
            //   mel = clip(mel, -max_abs_norm, +max_abs_norm)
            // 注意：上面 Step 5 已减去 ref_level_db，这里再减 min_level_db 并缩放
            float scale = (2.0f * config.maxAbsNorm) / (-config.minLevelDb);
            for (int i = 0; i < numFrames; i++) {
                float* row = melSpec.ptr<float>(i);
                for (int m = 0; m < config.nMels; m++) {
                    float v = (row[m] - config.minLevelDb) * scale
                            - config.maxAbsNorm;
                    row[m] = std::clamp(v, -config.maxAbsNorm, config.maxAbsNorm);
                }
            }
        }
        // apply_minmax=false: 输出 dB 域 log-mel（未归一化），供下游自定义归一化

        return melSpec;
    }
};

MelFeatureExtract::MelFeatureExtract() : impl_(std::make_unique<Impl>()) {}

MelFeatureExtract::~MelFeatureExtract() = default;

MelFeatureExtract::MelFeatureExtract(MelFeatureExtract&&) noexcept = default;

MelFeatureExtract& MelFeatureExtract::operator=(MelFeatureExtract&&) noexcept = default;

cv::Mat MelFeatureExtract::extract(const std::vector<std::vector<float>>& frames,
                                    const MelConfig& config,
                                    bool apply_minmax) {
    return impl_->extract(frames, config, apply_minmax);
}

}  // namespace audio
}  // namespace digital_human
