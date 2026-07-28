#include <iostream>
#include <vector>
#include <exception>
#include <string>
#include <filesystem>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include "core/face_detector.h"

namespace digital_human {
namespace core {

struct FaceDetector::Impl {
    dlib::frontal_face_detector detector_;
    dlib::shape_predictor predictor_;
    bool is_model_loaded_ = false;

    Impl() {
        detector_ = dlib::get_frontal_face_detector();
    }

    std::vector<cv::Rect> detect(const cv::Mat& image) {
        std::vector<cv::Rect> faces;
        if (image.empty()) {
            std::cerr << "Image is empty" << std::endl;
            return faces;
        }

        // dlib::rgb_pixel 要求 CV_8UC3，非 3 通道图像需要转换
        cv::Mat image_8uc3;
        if (image.type() != CV_8UC3) {
            if (image.channels() == 1) {
                cv::cvtColor(image, image_8uc3, cv::COLOR_GRAY2BGR);
            } else if (image.channels() == 4) {
                cv::cvtColor(image, image_8uc3, cv::COLOR_BGRA2BGR);
            } else {
                std::cerr << "[FaceDetector] 不支持的图像类型: " << image.type() << std::endl;
                return faces;
            }
        } else {
            image_8uc3 = image;
        }

        cv::Mat process_image;
        double scale = 0.5;
        double apply_scale = 1.0;

        const int max_size = 1200;
        const int min_size = 100;
        int image_height = image_8uc3.rows;
        int image_width = image_8uc3.cols;

        if (image_height > max_size || image_width > max_size) {
            cv::resize(image_8uc3, process_image, cv::Size(0, 0), scale, scale);
            apply_scale = scale;
        } else if (image_height < min_size || image_width < min_size) {
            cv::resize(image_8uc3, process_image, cv::Size(0, 0), 1.0 / scale, 1.0 / scale);
            apply_scale = 1.0 / scale;
        } else {
            process_image = image_8uc3;
        }

        try {
            dlib::cv_image<dlib::rgb_pixel> dlib_image(process_image);
            std::vector<dlib::rectangle> dlib_faces = detector_(dlib_image);
            double inv_scale = 1.0 / apply_scale;

            for (const auto& face : dlib_faces) {
                faces.push_back(cv::Rect(
                    static_cast<int>(face.left() * inv_scale),
                    static_cast<int>(face.top() * inv_scale),
                    static_cast<int>(face.right() * inv_scale - face.left() * inv_scale),
                    static_cast<int>(face.bottom() * inv_scale - face.top() * inv_scale)
                ));
            }
        } catch (const dlib::error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        return faces;
    }

    bool loadModel(const std::filesystem::path& model_path) {
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file not found: " << model_path << std::endl;
            is_model_loaded_ = false;
            return false;
        }

        try {
            dlib::deserialize(model_path.string()) >> predictor_;
            is_model_loaded_ = true;
            return true;
        } catch (const dlib::error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<cv::Point> getLandmarks(const cv::Mat& image, const cv::Rect& face_rect) {
        if (!is_model_loaded_) {
            std::cerr << "Model is not loaded" << std::endl;
            return {};
        }
        if (image.empty()) {
            std::cerr << "Image is empty" << std::endl;
            return {};
        }
        if (face_rect.width <= 64 || face_rect.height <= 64) {
            std::cerr << "face rect is too small" << std::endl;
            return {};
        }

        // dlib::rgb_pixel 要求 CV_8UC3
        cv::Mat image_8uc3;
        if (image.type() != CV_8UC3) {
            if (image.channels() == 1) {
                cv::cvtColor(image, image_8uc3, cv::COLOR_GRAY2BGR);
            } else if (image.channels() == 4) {
                cv::cvtColor(image, image_8uc3, cv::COLOR_BGRA2BGR);
            } else {
                std::cerr << "[FaceDetector] getLandmarks: 不支持的图像类型: "
                          << image.type() << std::endl;
                return {};
            }
        } else {
            image_8uc3 = image;
        }

        std::vector<cv::Point> landmarks;
        try {
            dlib::cv_image<dlib::rgb_pixel> dlib_image(image_8uc3);
            dlib::rectangle dlib_rect(
                face_rect.x, face_rect.y,
                face_rect.x + face_rect.width,
                face_rect.y + face_rect.height);
            dlib::full_object_detection shape = predictor_(dlib_image, dlib_rect);

            for (unsigned long i = 0; i < shape.num_parts(); ++i) {
                landmarks.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
            }
        } catch (const dlib::error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        return landmarks;
    }
};

FaceDetector::FaceDetector() : impl_(std::make_unique<Impl>()) {}
FaceDetector::~FaceDetector() = default;

bool FaceDetector::loadModel(const std::filesystem::path& model_path) {
    return impl_->loadModel(model_path);
}

bool FaceDetector::isModelLoaded() {
    return impl_->is_model_loaded_;
}

std::vector<cv::Point> FaceDetector::getLandmarks(const cv::Mat& image, const cv::Rect& face_rect) {
    return impl_->getLandmarks(image, face_rect);
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& image) {
    return impl_->detect(image);
}

}  // namespace core
}  // namespace digital_human