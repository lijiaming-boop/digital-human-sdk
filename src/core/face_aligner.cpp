#include "core/face_aligner.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

namespace digital_human {
namespace core {

struct FaceAligner::Impl {
    Impl() = default;
    ~Impl() = default;
    Impl(Impl&&) = default;
    Impl& operator=(Impl&&) = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    cv::Point2f getCenter(const std::vector<cv::Point>& points) {
        cv::Point2f center(0.0f, 0.0f);
        if (points.empty()) {
            return center;
        }
        for (const auto& p : points) {
            center.x += static_cast<float>(p.x);
            center.y += static_cast<float>(p.y);
        }
        center.x /= static_cast<float>(points.size());
        center.y /= static_cast<float>(points.size());
        return center;
    }

    cv::Mat alignByEyes(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks,
                        int face_size = 96) {
        if (image.empty() || landmarks.size() != 68) {
            std::cerr << "alignByEyes: image or landmarks is empty" << std::endl;
            return cv::Mat();
        }

        std::vector<cv::Point> left_eye, right_eye;
        for (int i = 36; i < 42; i++) {
            left_eye.push_back(landmarks[i]);
        }
        for (int i = 42; i < 48; i++) {
            right_eye.push_back(landmarks[i]);
        }

        cv::Point2f left_eye_center = getCenter(left_eye);
        cv::Point2f right_eye_center = getCenter(right_eye);

        double angle = atan2(right_eye_center.y - left_eye_center.y,
                             right_eye_center.x - left_eye_center.x) *
                       180.0 / CV_PI;

        double desired_dist = face_size * 0.4;
        double current_dist =
            sqrt(pow(right_eye_center.x - left_eye_center.x, 2) +
                 pow(right_eye_center.y - left_eye_center.y, 2));
        // 防除零：两眼重合时无法计算仿射变换
        if (current_dist < 1e-6) {
            std::cerr << "alignByEyes: 两眼重合，无法对齐" << std::endl;
            cv::Mat empty;
            return empty;
        }
        double scale = desired_dist / current_dist;

        cv::Point2f center((left_eye_center.x + right_eye_center.x) / 2.0f,
                           (left_eye_center.y + right_eye_center.y) / 2.0f);

        cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);

        double tx = face_size * 0.5;
        double ty = face_size * 0.4;
        M.at<double>(0, 2) += (tx - center.x);
        M.at<double>(1, 2) += (ty - center.y);

        cv::Mat aligned_image;
        cv::warpAffine(image, aligned_image, M, cv::Size(face_size, face_size));

        return aligned_image;
    }

    FaceAlignerResult alignByRect(const cv::Mat& image,
                                  const std::vector<cv::Point2f>& landmarks,
                                  int face_size, const cv::Rect& face_rect,
                                  double ratio) {
        FaceAlignerResult result;
        if (image.empty() || face_rect.width == 0 || face_rect.height == 0) {
            std::cerr << "alignByRect: image or face_rect is empty" << std::endl;
            result.valid = false;
            return result;
        }
        if (landmarks.size() != 68) {
            std::cerr << "alignByRect: need 68 landmarks, got " << landmarks.size()
                      << std::endl;
            result.valid = false;
            return result;
        }

        int dw = static_cast<int>(face_rect.width * ratio);
        int dh = static_cast<int>(face_rect.height * ratio);
        int x1 = std::max(0, face_rect.x - dw);
        int y1 = std::max(0, face_rect.y - dh);
        int x2 = std::min(image.cols, face_rect.x + face_rect.width + dw);
        int y2 = std::min(image.rows, face_rect.y + face_rect.height + dh);

        std::vector<cv::Point2f> local_landmarks;
        local_landmarks.reserve(landmarks.size());
        for (const auto& p : landmarks) {
            local_landmarks.push_back(
                cv::Point2f(p.x - x1, p.y - y1));
        }

        std::vector<cv::Point> left_eye, right_eye;
        for (int i = 36; i < 42; i++) {
            left_eye.push_back(local_landmarks[i]);
        }
        for (int i = 42; i < 48; i++) {
            right_eye.push_back(local_landmarks[i]);
        }

        cv::Point2f left_eye_center = getCenter(left_eye);
        cv::Point2f right_eye_center = getCenter(right_eye);

        double angle = atan2(right_eye_center.y - left_eye_center.y,
                             right_eye_center.x - left_eye_center.x) *
                       180.0 / CV_PI;

        double desired_dist = face_size * 0.4;
        double current_dist =
            sqrt(pow(right_eye_center.x - left_eye_center.x, 2) +
                 pow(right_eye_center.y - left_eye_center.y, 2));
        // 防除零：两眼重合时无法计算仿射变换
        if (current_dist < 1e-6) {
            std::cerr << "alignByRect: 两眼重合，无法对齐" << std::endl;
            result.valid = false;
            return result;
        }
        double scale = desired_dist / current_dist;

        cv::Point2f center((left_eye_center.x + right_eye_center.x) / 2.0f,
                           (left_eye_center.y + right_eye_center.y) / 2.0f);

        cv::Mat M_local = cv::getRotationMatrix2D(center, angle, scale);

        double tx = face_size * 0.5;
        double ty = face_size * 0.4;
        M_local.at<double>(0, 2) += (tx - center.x);
        M_local.at<double>(1, 2) += (ty - center.y);

        cv::Mat crop = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::Mat aligned_face;
        cv::warpAffine(crop, aligned_face, M_local,
                       cv::Size(face_size, face_size));

        cv::Mat M_full = M_local.clone();
        M_full.at<double>(0, 2) -=
            M_local.at<double>(0, 0) * x1 + M_local.at<double>(0, 1) * y1;
        M_full.at<double>(1, 2) -=
            M_local.at<double>(1, 0) * x1 + M_local.at<double>(1, 1) * y1;

        cv::Mat M_inv;
        cv::invertAffineTransform(M_full, M_inv);

        std::vector<cv::Point2f> transformed_landmarks;
        transformed_landmarks.reserve(landmarks.size());
        for (const auto& p : landmarks) {
            double tx_p = M_full.at<double>(0, 0) * p.x +
                          M_full.at<double>(0, 1) * p.y +
                          M_full.at<double>(0, 2);
            double ty_p = M_full.at<double>(1, 0) * p.x +
                          M_full.at<double>(1, 1) * p.y +
                          M_full.at<double>(1, 2);
            transformed_landmarks.push_back(
                cv::Point2f(static_cast<float>(tx_p), static_cast<float>(ty_p)));
        }

        result.valid = true;
        result.aligned_face = aligned_face;
        result.M = M_full;
        result.M_inv = M_inv;
        result.face_rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        result.landmarks = transformed_landmarks;

        return result;
    }
};

FaceAligner::FaceAligner()
    : impl_(std::make_unique<Impl>()) {}

FaceAligner::~FaceAligner() = default;

FaceAligner::FaceAligner(FaceAligner&&) noexcept = default;

FaceAligner& FaceAligner::operator=(FaceAligner&&) noexcept = default;

cv::Mat FaceAligner::align(const cv::Mat& image,
                              const std::vector<cv::Point2f>& landmarks,
                              int face_size) {
    return impl_->alignByEyes(image, landmarks, face_size);
}

FaceAlignerResult FaceAligner::alignByRect(
    const cv::Mat& image, const std::vector<cv::Point2f>& landmarks,
    int face_size, const cv::Rect& face_rect, double ratio) {
    return impl_->alignByRect(image, landmarks, face_size, face_rect, ratio);
}

std::vector<cv::Point2f> FaceAligner::transform_landmarks(
    const std::vector<cv::Point2f>& landmarks, const cv::Mat& M) const {
    std::vector<cv::Point2f> result;
    result.reserve(landmarks.size());
    for (const auto& p : landmarks) {
        double x =
            M.at<double>(0, 0) * p.x + M.at<double>(0, 1) * p.y + M.at<double>(0, 2);
        double y =
            M.at<double>(1, 0) * p.x + M.at<double>(1, 1) * p.y + M.at<double>(1, 2);
        result.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
    }
    return result;
}

}  // namespace core
}  // namespace digital_human
