#include "audio/audio_mel_feature_extract.h"

#include <cmath>
#include <algorithm>

namespace digital_human {
namespace audio {

struct MelFeatureExtract::Impl {
    cv::Mat melFilterbank_;
    MelConfig cachedConfig_;

    cv::Mat createMelFilterbank(const MelConfig& config) {
        int nFreqBins = config.nFFT / 2 + 1;

        auto hzToMel = [](float hz) {
            return 2595.0f * std::log10(1.0f + hz / 700.0f);
        };
        auto melToHz = [](float mel) {
            return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
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

            // 防除零：相邻频点重合时跳过该滤波器（权重保持为零）
            if (std::abs(center - left) < 1e-8f || std::abs(right - center) < 1e-8f) {
                continue;
            }

            float denom_left  = center - left;
            float denom_right = right - center;

            for (int k = 0; k < nFreqBins; k++) {
                float fk = static_cast<float>(k);
                if (fk >= left && fk <= center) {
                    filterbank.at<float>(m, k) = (fk - left) / denom_left;
                } else if (fk >= center && fk <= right) {
                    filterbank.at<float>(m, k) = (right - fk) / denom_right;
                }
            }
        }

        return filterbank;
    }

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

        // Step 1: Copy all PCM frames to padded matrix (numFrames x nFFT)
        cv::Mat paddedFrames(numFrames, nFFT, CV_32F, cv::Scalar(0.0f));
        for (int i = 0; i < numFrames; i++) {
            float* row = paddedFrames.ptr<float>(i);
            const auto& frame = frames[i];
            int copyLen = std::min(static_cast<int>(frame.size()), nFFT);
            for (int j = 0; j < copyLen; j++) {
                row[j] = frame[j];
            }
        }

        // Step 2: FFT on every row
        cv::Mat fftResult;
        cv::dft(paddedFrames, fftResult, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);

        // Step 3: Power spectrum — keep only bins [0, nFreqBins)
        cv::Mat powerSpec(numFrames, nFreqBins, CV_32F);
        for (int i = 0; i < numFrames; i++) {
            float* psRow = powerSpec.ptr<float>(i);
            const float* fftRow = fftResult.ptr<float>(i);
            for (int k = 0; k < nFreqBins; k++) {
                float real = fftRow[2 * k];
                float imag = fftRow[2 * k + 1];
                psRow[k] = real * real + imag * imag;
            }
        }

        // Step 4: Apply Mel filterbank
        // melFilterbank_: (nMels x nFreqBins)  *  powerSpec^T: (nFreqBins x numFrames)
        // melEnergies: (nMels x numFrames)
        cv::Mat melEnergies = melFilterbank_ * powerSpec.t();
        cv::Mat melSpec = melEnergies.t();  // (numFrames x nMels)

        // Step 5: Log compression (dB scale)
        for (int i = 0; i < numFrames; i++) {
            float* row = melSpec.ptr<float>(i);
            for (int m = 0; m < config.nMels; m++) {
                row[m] = 10.0f * std::log10(std::max(row[m], 1e-10f));
            }
        }

        // Step 6: Normalize to [0, 1]（可选 — 统计范围为本次输入全体帧。
        // 流式单帧调用时必须跳过，否则每帧被独立拉伸，帧间动态被破坏）
        if (apply_minmax) {
            double minVal, maxVal;
            cv::minMaxLoc(melSpec, &minVal, &maxVal);
            if (maxVal > minVal) {
                melSpec = (melSpec - minVal) / (maxVal - minVal);
            }
        }

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
