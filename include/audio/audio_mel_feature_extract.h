#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace digital_human {
namespace audio {

struct MelConfig {
    int nFFT = 512;
    int nMels = 80;
    int sampleRate = 16000;
    float fMin = 0.0f;
    float fMax = 8000.0f;
};

class MelFeatureExtract {
public:
    MelFeatureExtract();
    ~MelFeatureExtract();
    MelFeatureExtract(const MelFeatureExtract&) = delete;
    MelFeatureExtract& operator=(const MelFeatureExtract&) = delete;
    MelFeatureExtract(MelFeatureExtract&&) noexcept;
    MelFeatureExtract& operator=(MelFeatureExtract&&) noexcept;

    /**
     * @brief 提取 Mel 频谱特征
     *
     * @param frames        PCM 分帧数据（每帧等长）
     * @param config        Mel 配置
     * @param apply_minmax  是否对输出做 min-max 归一化到 [0,1]。
     *                      注意：归一化统计范围是【本次调用输入的全体帧】，
     *                      单帧输入会被独立拉伸，破坏帧间能量动态。
     *                      流式逐帧场景应传 false 输出 dB 域 log-mel，
     *                      由具备上下文窗口的下游统一归一化。
     * @return cv::Mat (rows=帧数, cols=nMels, CV_32F)
     */
    cv::Mat extract(const std::vector<std::vector<float>>& frames,
                    const MelConfig& config = {},
                    bool apply_minmax = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
