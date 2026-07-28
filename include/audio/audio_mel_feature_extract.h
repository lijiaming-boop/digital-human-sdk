#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace digital_human {
namespace audio {

/// @brief Mel 频谱提取配置
///
/// 默认值对齐 Wav2Lip 官方实现 (hparams.py + audio.py)：
///   n_fft=800, win_size=800, hop 由分帧器决定,
///   n_mels=80, fmin=55, fmax=7600, sample_rate=16000,
///   ref_level_db=20, min_level_db=-100, max_abs_norm=4.0
struct MelConfig {
    int   nFFT        = 800;     ///< FFT 点数（Wav2Lip: 800）
    int   nMels       = 80;      ///< Mel 滤波器组数（Wav2Lip: 80）
    int   sampleRate  = 16000;   ///< 采样率（Hz）
    float fMin        = 55.0f;   ///< 最低频率（Hz，Wav2Lip: 55）
    float fMax        = 7600.0f; ///< 最高频率（Hz，Wav2Lip: 7600）
    int   winSize     = 800;     ///< 窗长（samples，Wav2Lip: 800；≤0 时取 nFFT）
    float refLevelDb  = 20.0f;   ///< 参考电平（dB，Wav2Lip: 20）
    float minLevelDb  = -100.0f; ///< 最低电平（dB，Wav2Lip: -100）
    float maxAbsNorm  = 4.0f;    ///< symmetric 归一化上界（Wav2Lip: 4.0）
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
     * @param config        Mel 配置（默认对齐 Wav2Lip 官方）
     * @param apply_minmax  是否做归一化：
     *                      - true:  做 Wav2Lip symmetric 归一化到 [-maxAbsNorm, maxAbsNorm]，
     *                               这是 Wav2Lip 模型推理必须使用的模式。
     *                      - false: 仅输出 dB 域 log-mel（未归一化），
     *                               供需要自定义归一化的下游使用。
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
