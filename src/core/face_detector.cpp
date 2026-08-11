#include "core/face_detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <utility>

#include <layer.h>
#include <net.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace digital_human {
namespace core {
namespace {

constexpr int kDetectorInputSize = 640;
constexpr int kLandmarkInputSize = 192;
constexpr float kScoreThreshold = 0.50f;
constexpr float kNmsThreshold = 0.40f;

// 2D106 (JD/InsightFace layout) -> the iBUG 68-point order used downstream.
constexpr std::array<int, 68> kLandmark106To68 = {
    1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17,
    43, 48, 49, 51, 50, 102, 103, 104, 105, 101, 72, 73, 74, 86, 78,
    79, 80, 85, 84, 35, 41, 42, 39, 37, 36, 89, 95, 96, 93, 91, 90,
    52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70,
    69, 57, 60, 54
};

struct Proposal {
    cv::Rect2f rect;
    float score = 0.0f;
};

float IntersectionArea(const Proposal& a, const Proposal& b) {
    return (a.rect & b.rect).area();
}

void Nms(std::vector<Proposal>& proposals) {
    std::sort(proposals.begin(), proposals.end(), [](const Proposal& a, const Proposal& b) {
        return a.score > b.score;
    });

    std::vector<Proposal> kept;
    kept.reserve(proposals.size());
    for (const auto& candidate : proposals) {
        bool keep = true;
        for (const auto& selected : kept) {
            const float intersection = IntersectionArea(candidate, selected);
            const float union_area = candidate.rect.area() + selected.rect.area() - intersection;
            if (union_area > 0.0f && intersection / union_area > kNmsThreshold) {
                keep = false;
                break;
            }
        }
        if (keep) kept.push_back(candidate);
    }
    proposals = std::move(kept);
}

cv::Mat EnsureBgr8(const cv::Mat& image, const char* operation) {
    if (image.empty()) return {};
    if (image.type() == CV_8UC3) return image;

    cv::Mat bgr;
    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.type() == CV_8UC4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else {
        std::cerr << "[FaceDetector] " << operation
                  << ": expected CV_8UC1, CV_8UC3, or CV_8UC4; got type "
                  << image.type() << std::endl;
    }
    return bgr;
}

}  // namespace

struct FaceDetector::Impl {
    ncnn::Net detector_net;
    cv::dnn::Net landmark_net;
    bool model_loaded = false;

    bool loadModel(const std::filesystem::path& model_dir) {
        const auto detector_param = model_dir / "scrfd_2.5g_kps-opt2.param";
        const auto detector_bin = model_dir / "scrfd_2.5g_kps-opt2.bin";
        const auto landmark_onnx = model_dir / "2d106det.onnx";

        if (!std::filesystem::is_directory(model_dir) ||
            !std::filesystem::is_regular_file(detector_param) ||
            !std::filesystem::is_regular_file(detector_bin) ||
            !std::filesystem::is_regular_file(landmark_onnx)) {
            std::cerr << "[FaceDetector] model directory must contain "
                      << "scrfd_2.5g_kps-opt2.param/.bin and 2d106det.onnx: "
                      << model_dir << std::endl;
            model_loaded = false;
            return false;
        }

        detector_net.clear();
        landmark_net = cv::dnn::Net();
        if (detector_net.load_param(detector_param.string().c_str()) != 0 ||
            detector_net.load_model(detector_bin.string().c_str()) != 0) {
            std::cerr << "[FaceDetector] failed to load SCRFD model from " << model_dir << std::endl;
            model_loaded = false;
            return false;
        }

        try {
            landmark_net = cv::dnn::readNetFromONNX(landmark_onnx.string());
        } catch (const cv::Exception& e) {
            std::cerr << "[FaceDetector] failed to load 2D106 model: " << e.what() << std::endl;
            detector_net.clear();
            model_loaded = false;
            return false;
        }

        model_loaded = !landmark_net.empty();
        return model_loaded;
    }

    void GenerateProposals(const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob,
                           int stride, std::vector<Proposal>& proposals) const {
        if (score_blob.empty() || bbox_blob.empty() || score_blob.c < 2 || bbox_blob.c < 8) {
            return;
        }

        for (int anchor = 0; anchor < 2; ++anchor) {
            const ncnn::Mat scores = score_blob.channel(anchor);
            const ncnn::Mat boxes = bbox_blob.channel_range(anchor * 4, 4);
            for (int y = 0; y < scores.h; ++y) {
                for (int x = 0; x < scores.w; ++x) {
                    const int index = y * scores.w + x;
                    const float score = scores[index];
                    if (score < kScoreThreshold) continue;

                    const float cx = static_cast<float>(x * stride);
                    const float cy = static_cast<float>(y * stride);
                    const float left = boxes.channel(0)[index] * stride;
                    const float top = boxes.channel(1)[index] * stride;
                    const float right = boxes.channel(2)[index] * stride;
                    const float bottom = boxes.channel(3)[index] * stride;
                    proposals.push_back({cv::Rect2f(cx - left, cy - top,
                                                     left + right, top + bottom), score});
                }
            }
        }
    }

    std::vector<cv::Rect> detect(const cv::Mat& image) const {
        std::vector<cv::Rect> faces;
        if (!model_loaded) {
            std::cerr << "[FaceDetector] model is not loaded" << std::endl;
            return faces;
        }

        const cv::Mat bgr = EnsureBgr8(image, "detect");
        if (bgr.empty()) return faces;

        const float scale = static_cast<float>(kDetectorInputSize) /
                            static_cast<float>(std::max(bgr.cols, bgr.rows));
        const int resized_w = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
        const int resized_h = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
        ncnn::Mat input = ncnn::Mat::from_pixels_resize(
            bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, resized_w, resized_h);

        const int wpad = (resized_w + 31) / 32 * 32 - resized_w;
        const int hpad = (resized_h + 31) / 32 * 32 - resized_h;
        ncnn::Mat padded;
        ncnn::copy_make_border(input, padded, hpad / 2, hpad - hpad / 2,
                               wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.0f);
        const float mean_values[3] = {127.5f, 127.5f, 127.5f};
        const float norm_values[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
        padded.substract_mean_normalize(mean_values, norm_values);

        ncnn::Extractor extractor = detector_net.create_extractor();
        if (extractor.input("input.1", padded) != 0) {
            std::cerr << "[FaceDetector] SCRFD input blob 'input.1' is unavailable" << std::endl;
            return faces;
        }

        std::vector<Proposal> proposals;
        for (const int stride : {8, 16, 32}) {
            ncnn::Mat scores;
            ncnn::Mat boxes;
            const std::string score_name = "score_" + std::to_string(stride);
            const std::string box_name = "bbox_" + std::to_string(stride);
            if (extractor.extract(score_name.c_str(), scores) != 0 ||
                extractor.extract(box_name.c_str(), boxes) != 0) {
                std::cerr << "[FaceDetector] unsupported SCRFD model outputs; expected score_"
                          << stride << " and bbox_" << stride << std::endl;
                return {};
            }
            GenerateProposals(scores, boxes, stride, proposals);
        }

        Nms(proposals);
        const float pad_x = static_cast<float>(wpad / 2);
        const float pad_y = static_cast<float>(hpad / 2);
        for (const auto& proposal : proposals) {
            const float x0 = std::clamp((proposal.rect.x - pad_x) / scale,
                                        0.0f, static_cast<float>(bgr.cols - 1));
            const float y0 = std::clamp((proposal.rect.y - pad_y) / scale,
                                        0.0f, static_cast<float>(bgr.rows - 1));
            const float x1 = std::clamp((proposal.rect.x + proposal.rect.width - pad_x) / scale,
                                        0.0f, static_cast<float>(bgr.cols));
            const float y1 = std::clamp((proposal.rect.y + proposal.rect.height - pad_y) / scale,
                                        0.0f, static_cast<float>(bgr.rows));
            const int left = static_cast<int>(std::floor(x0));
            const int top = static_cast<int>(std::floor(y0));
            const int right = static_cast<int>(std::ceil(x1));
            const int bottom = static_cast<int>(std::ceil(y1));
            if (right > left && bottom > top) faces.emplace_back(left, top, right - left, bottom - top);
        }
        return faces;
    }

    std::vector<cv::Point> getLandmarks(const cv::Mat& image, const cv::Rect& face_rect) {
        if (!model_loaded || image.empty() || face_rect.width < 2 || face_rect.height < 2) {
            return {};
        }

        const cv::Mat bgr = EnsureBgr8(image, "getLandmarks");
        if (bgr.empty()) return {};
        const cv::Rect image_rect(0, 0, bgr.cols, bgr.rows);
        const cv::Rect bounded_face = face_rect & image_rect;
        if (bounded_face.width < 2 || bounded_face.height < 2) return {};

        const cv::Point2f center(bounded_face.x + bounded_face.width * 0.5f,
                                 bounded_face.y + bounded_face.height * 0.5f);
        const double scale = static_cast<double>(kLandmarkInputSize) /
                             (static_cast<double>(std::max(bounded_face.width, bounded_face.height)) * 1.5);
        cv::Mat transform = cv::getRotationMatrix2D(center, 0.0, scale);
        transform.at<double>(0, 2) += kLandmarkInputSize * 0.5 - center.x;
        transform.at<double>(1, 2) += kLandmarkInputSize * 0.5 - center.y;

        cv::Mat aligned;
        cv::warpAffine(bgr, aligned, transform,
                       cv::Size(kLandmarkInputSize, kLandmarkInputSize),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        const cv::Mat blob = cv::dnn::blobFromImage(
            aligned, 1.0, cv::Size(kLandmarkInputSize, kLandmarkInputSize),
            cv::Scalar(), true, false);

        cv::Mat output;
        try {
            landmark_net.setInput(blob);
            output = landmark_net.forward();
        } catch (const cv::Exception& e) {
            std::cerr << "[FaceDetector] 2D106 inference failed: " << e.what() << std::endl;
            return {};
        }

        if (output.total() != 212) {
            std::cerr << "[FaceDetector] 2D106 output must contain 212 values; got "
                      << output.total() << std::endl;
            return {};
        }
        const cv::Mat flat = output.reshape(1, 1);
        cv::Mat inverse;
        cv::invertAffineTransform(transform, inverse);

        std::array<cv::Point2f, 106> points106;
        for (int i = 0; i < 106; ++i) {
            const float x = (flat.at<float>(0, i * 2) + 1.0f) * (kLandmarkInputSize / 2.0f);
            const float y = (flat.at<float>(0, i * 2 + 1) + 1.0f) * (kLandmarkInputSize / 2.0f);
            points106[i] = cv::Point2f(
                static_cast<float>(inverse.at<double>(0, 0) * x + inverse.at<double>(0, 1) * y + inverse.at<double>(0, 2)),
                static_cast<float>(inverse.at<double>(1, 0) * x + inverse.at<double>(1, 1) * y + inverse.at<double>(1, 2)));
        }

        std::vector<cv::Point> landmarks;
        landmarks.reserve(kLandmark106To68.size());
        for (const int index : kLandmark106To68) {
            landmarks.emplace_back(cvRound(points106[index].x), cvRound(points106[index].y));
        }
        return landmarks;
    }
};

FaceDetector::FaceDetector() : impl_(std::make_unique<Impl>()) {}
FaceDetector::~FaceDetector() = default;
FaceDetector::FaceDetector(FaceDetector&&) noexcept = default;
FaceDetector& FaceDetector::operator=(FaceDetector&&) noexcept = default;

bool FaceDetector::loadModel(const std::filesystem::path& model_path) {
    return impl_->loadModel(model_path);
}

bool FaceDetector::isModelLoaded() const {
    return impl_->model_loaded;
}

std::vector<cv::Point> FaceDetector::getLandmarks(const cv::Mat& image, const cv::Rect& face_rect) {
    return impl_->getLandmarks(image, face_rect);
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& image) {
    return impl_->detect(image);
}

}  // namespace core
}  // namespace digital_human
