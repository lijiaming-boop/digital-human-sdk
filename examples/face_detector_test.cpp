#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "core/face_detector.h"

namespace fs = std::filesystem;
using namespace DigitalHuman::core;

std::string resolvePath(const fs::path& relative) {
    fs::path dir = fs::current_path();
    while (true) {
        fs::path candidate = dir / relative;
        if (fs::exists(candidate)) return candidate.string();
        fs::path parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return relative.string();
}

void testEmptyImage() {
    std::cout << "[Test 1] Empty image..." << std::endl;
    FaceDetector detector;
    cv::Mat empty_img;
    std::vector<cv::Rect> faces = detector.detect(empty_img);
    std::cout << (faces.empty() ? "  PASS: returned empty" : "  FAIL: should be empty") << std::endl;
}

void testRealImage(const std::string& image_path) {
    std::cout << "[Test 2] Real face image..." << std::endl;

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cout << "  FAIL: cannot load " << image_path << std::endl;
        return;
    }
    std::cout << "  Image size: " << img.cols << "x" << img.rows << std::endl;

    FaceDetector detector;
    std::vector<cv::Rect> faces = detector.detect(img);
    std::cout << "  Found " << faces.size() << " face(s)" << std::endl;

    if (faces.empty()) {
        std::cout << "  FAIL: no face detected" << std::endl;
        return;
    }

    for (size_t i = 0; i < faces.size(); ++i) {
        std::cout << "  Face #" << i << ": (" << faces[i].x << "," << faces[i].y << ") "
                  << faces[i].width << "x" << faces[i].height << std::endl;
        cv::rectangle(img, faces[i], cv::Scalar(0, 255, 0), 2);
    }

    std::string output = "face_detector_result.jpg";
    cv::imwrite(output, img);
    std::cout << "  PASS: result saved to " << output << std::endl;
}

void testDownsampleStrategy(const std::string& image_path) {
    std::cout << "[Test 3] Large image downsample..." << std::endl;

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cout << "  FAIL: cannot load image" << std::endl;
        return;
    }

    // Upscale to trigger downsample path (>1200px)
    cv::Mat large_img;
    cv::resize(img, large_img, cv::Size(2000, 1500));
    std::cout << "  Scaled to: " << large_img.cols << "x" << large_img.rows << std::endl;

    FaceDetector detector;
    std::vector<cv::Rect> faces = detector.detect(large_img);
    std::cout << "  Found " << faces.size() << " face(s)" << std::endl;

    if (!faces.empty()) {
        for (size_t i = 0; i < faces.size(); ++i) {
            std::cout << "  Face #" << i << ": (" << faces[i].x << "," << faces[i].y << ") "
                      << faces[i].width << "x" << faces[i].height << std::endl;
            // Verify coordinates are within original (large) image bounds
            bool in_bounds = faces[i].x >= 0 && faces[i].y >= 0 &&
                             faces[i].x + faces[i].width <= large_img.cols &&
                             faces[i].y + faces[i].height <= large_img.rows;
            std::cout << "  Coordinate in bounds: " << (in_bounds ? "YES" : "NO") << std::endl;
            if (!in_bounds) {
                std::cout << "  FAIL: coordinates out of bounds (scale-back bug?)" << std::endl;
            }
        }
    }
    std::cout << "  PASS" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "===== FaceDetector Test =====" << std::endl;

    std::string image_path;
    if (argc >= 2) {
        image_path = argv[1];
    } else {
        image_path = resolvePath("face.jpg");
    }
    std::cout << "Image: " << image_path << std::endl << std::endl;

    testEmptyImage();
    std::cout << std::endl;

    testRealImage(image_path);
    std::cout << std::endl;

    testDownsampleStrategy(image_path);
    std::cout << std::endl;

    std::cout << "===== Done =====" << std::endl;
    return 0;
}
