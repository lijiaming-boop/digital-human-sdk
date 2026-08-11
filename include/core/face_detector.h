#pragma once
#include <string>
#include <vector>
#include <memory>
#include <exception>
#include<filesystem>
#include <opencv2/opencv.hpp>

namespace digital_human{
    namespace core{
        class FaceDetector{
        public:
            FaceDetector();
            ~FaceDetector();
            FaceDetector(const FaceDetector& other) = delete;
            FaceDetector& operator=(const FaceDetector& other) = delete;
            FaceDetector(FaceDetector&& other) noexcept;
            FaceDetector& operator=(FaceDetector&& other) noexcept;

            /// 载入 SCRFD + 2D106 模型目录，详见 docs/models/face_models.md。
            bool isModelLoaded() const;
            std::vector<cv::Rect> detect(const cv::Mat& image);
            /// @param model_path 包含 SCRFD ncnn 模型和 2d106det.onnx 的目录
            bool loadModel(const std::filesystem::path& model_path);
            /// 保持旧 68 点接口；内部由 2D106 深度模型映射。
            std::vector<cv::Point> getLandmarks(const cv::Mat& image, const cv::Rect& face_rect);

            
        private:
            struct Impl;
            std::unique_ptr<Impl> impl_;
        };
    }
}
