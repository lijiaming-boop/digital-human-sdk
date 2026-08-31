#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include "digital_human/export.h"

namespace digital_human {
namespace audio {

class DH_API CMVN {
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
