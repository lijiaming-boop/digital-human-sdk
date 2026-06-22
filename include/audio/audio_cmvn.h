#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

namespace digital_human {
namespace audio {

class CMVN {
public:
    CMVN();
    ~CMVN();
    CMVN(const CMVN&) = delete;
    CMVN& operator=(const CMVN&) = delete;
    CMVN(CMVN&&) noexcept;
    CMVN& operator=(CMVN&&) noexcept;

    cv::Mat process(const cv::Mat& melSpectrogram) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
