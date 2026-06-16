#pragma once

#include <filesystem>
#include <memory>
#include <vector>
#include <opencv2/core/types.hpp>

namespace cv {
class Mat;
}

namespace DigitalHuman {
namespace core {

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();
    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;
    FaceDetector(FaceDetector&&) noexcept;
    FaceDetector& operator=(FaceDetector&&) noexcept;

    bool loadModel(const std::filesystem::path& model_path);
    bool isModelLoaded();

    std::vector<cv::Rect> detect(const cv::Mat& image);
    std::vector<cv::Point> getLandmarks(const cv::Mat& image, const cv::Rect& face_rect);

private:
    struct FaceDetectorImpl;
    std::unique_ptr<FaceDetectorImpl> impl_;
};

} // namespace core
} // namespace DigitalHuman