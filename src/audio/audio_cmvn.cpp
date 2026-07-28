#include "audio/audio_cmvn.h"

#include <cmath>

namespace digital_human {
namespace audio {

struct CMVN::Impl {
    static constexpr float kEps = 1e-10f;

    cv::Mat process(const cv::Mat& melSpectrogram) const {
        if (melSpectrogram.empty()) return cv::Mat();

        int nFrames = melSpectrogram.rows;
        int nMels = melSpectrogram.cols;

        // Per-dimension mean: average across frames (reduce rows → 1 × nMels)
        cv::Mat meanRow;
        cv::reduce(melSpectrogram, meanRow, 0, cv::REDUCE_AVG);

        // Per-dimension std: sqrt of mean of squared deviations
        cv::Mat result(nFrames, nMels, CV_32F);
        for (int j = 0; j < nMels; j++) {
            float mean = meanRow.at<float>(0, j);

            float sumSq = 0.0f;
            for (int i = 0; i < nFrames; i++) {
                float diff = melSpectrogram.at<float>(i, j) - mean;
                sumSq += diff * diff;
            }
            float std = std::sqrt(sumSq / nFrames);
            // 防除零/放大噪声：std 极小时不放大（使用条件赋值而非 max+除）
            float invStd = (std > kEps) ? (1.0f / std) : 1.0f;

            for (int i = 0; i < nFrames; i++) {
                float val = melSpectrogram.at<float>(i, j);
                result.at<float>(i, j) = (val - mean) * invStd;
            }
        }

        return result;
    }
};

CMVN::CMVN() : impl_(std::make_unique<Impl>()) {}

CMVN::~CMVN() = default;

CMVN::CMVN(CMVN&&) noexcept = default;

CMVN& CMVN::operator=(CMVN&&) noexcept = default;

cv::Mat CMVN::process(const cv::Mat& melSpectrogram) const {
    return impl_->process(melSpectrogram);
}

}  // namespace audio
}  // namespace digital_human
