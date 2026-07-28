#pragma once
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace digital_human {
namespace core {

struct FaceAlignerResult;

class FaceAligner {
public:
    FaceAligner();
    ~FaceAligner();
    FaceAligner& operator=(const FaceAligner& other) = delete;
    FaceAligner(const FaceAligner& other) = delete;
    FaceAligner& operator=(FaceAligner&& other) noexcept;
    FaceAligner(FaceAligner&& other) noexcept;

    cv::Mat align(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks,
                  int face_size = 96);

    FaceAlignerResult alignByRect(const cv::Mat& image,
                                  const std::vector<cv::Point2f>& landmarks,
                                  int face_size, const cv::Rect& face_rect,
                                  double ratio = 0.15);

    std::vector<cv::Point2f> transform_landmarks(
        const std::vector<cv::Point2f>& landmarks, const cv::Mat& M) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct FaceAlignerResult {
    bool valid = false;
    cv::Mat aligned_face;
    cv::Mat M;
    cv::Mat M_inv;
    cv::Rect face_rect;
    std::vector<cv::Point2f> landmarks;
};

}  // namespace core
}  // namespace digital_human
