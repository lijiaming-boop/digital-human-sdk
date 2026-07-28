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

            bool isModelLoaded();
            std::vector<cv::Rect> detect(const cv::Mat& image);
            bool loadModel(const std::filesystem::path& model_path);
            std::vector<cv::Point> getLandmarks(const cv::Mat& image, const cv::Rect& face_rect);

            
        private:
            struct Impl;
            std::unique_ptr<Impl> impl_;
        };
    }
}