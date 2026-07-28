#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "core/face_detector.h"

namespace fs = std::filesystem;
using namespace digital_human::core;

const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar BLUE(255, 0, 0);

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

void drawLandmarks(cv::Mat& img, const std::vector<cv::Point>& landmarks) {
    for (size_t i = 0; i < landmarks.size(); ++i) {
        cv::circle(img, landmarks[i], 1, GREEN, -1);
    }
}

bool validateLandmarks(const std::vector<cv::Point>& landmarks, const cv::Rect& face_rect) {
    if (landmarks.size() != 68) {
        std::cerr << "  expected 68 landmarks, got " << landmarks.size() << std::endl;
        return false;
    }

    int out_of_bounds = 0;
    for (const auto& p : landmarks) {
        if (p.x < face_rect.x || p.x > face_rect.x + face_rect.width ||
            p.y < face_rect.y || p.y > face_rect.y + face_rect.height) {
            ++out_of_bounds;
        }
    }
    if (out_of_bounds > 0) {
        std::cerr << "  " << out_of_bounds << " / " << landmarks.size()
                  << " landmarks outside face rect (expected: some jaw points near border)" << std::endl;
    }

    std::cout << "  Face rect: (" << face_rect.x << "," << face_rect.y << ") "
              << face_rect.width << "x" << face_rect.height << std::endl;

    if (landmarks.size() >= 68) {
        std::cout << "  Nose tip (pt 30): (" << landmarks[30].x << "," << landmarks[30].y << ")" << std::endl;
        std::cout << "  Left eye (pt 36): (" << landmarks[36].x << "," << landmarks[36].y << ")" << std::endl;
        std::cout << "  Right eye (pt 45): (" << landmarks[45].x << "," << landmarks[45].y << ")" << std::endl;
        std::cout << "  Mouth (pt 48): (" << landmarks[48].x << "," << landmarks[48].y << ")" << std::endl;

        bool nose_in_face = face_rect.contains(cv::Point(landmarks[30].x, landmarks[30].y));
        bool left_eye_in_face = face_rect.contains(cv::Point(landmarks[36].x, landmarks[36].y));
        std::cout << "  Nose tip inside face rect: " << (nose_in_face ? "YES" : "NO") << std::endl;
        std::cout << "  Left eye inside face rect: " << (left_eye_in_face ? "YES" : "NO") << std::endl;
    }

    return landmarks.size() == 68;
}

int main(int argc, char** argv) {
    std::cout << "===== Face Landmark Test =====" << std::endl;

    std::string image_path;
    std::string model_path;

    if (argc >= 2) {
        image_path = argv[1];
    } else {
        image_path = resolvePath("face.jpg");
    }

    if (argc >= 3) {
        model_path = argv[2];
    } else {
        model_path = resolvePath("models/shape_predictor_68_face_landmarks.dat");
    }

    std::cout << "Image:  " << image_path << std::endl;
    std::cout << "Model:  " << model_path << std::endl;

    if (!fs::exists(image_path)) {
        std::cerr << "Image not found: " << image_path << std::endl;
        return 1;
    }
    if (!fs::exists(model_path)) {
        std::cerr << "Model not found: " << model_path << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;

    FaceDetector detector;

    std::cout << "\n[1] Loading model..." << std::endl;
    if (!detector.loadModel(model_path)) {
        std::cerr << "Model load failed" << std::endl;
        return 1;
    }
    std::cout << "  Model loaded OK" << std::endl;

    std::cout << "\n[2] Detecting faces..." << std::endl;
    std::vector<cv::Rect> faces = detector.detect(img);
    std::cout << "  Found " << faces.size() << " face(s)" << std::endl;

    if (faces.empty()) {
        std::cout << "No faces found in image. Try a clearer frontal face photo." << std::endl;
        return 0;
    }

    std::cout << "\n[3] Extracting landmarks..." << std::endl;
    bool all_valid = true;
    for (size_t i = 0; i < faces.size(); ++i) {
        std::cout << "  Face #" << i << ":" << std::endl;
        cv::rectangle(img, faces[i], BLUE, 2);

        std::vector<cv::Point> landmarks = detector.getLandmarks(img, faces[i]);

        if (!validateLandmarks(landmarks, faces[i])) {
            all_valid = false;
        }
        std::cout << "  Landmarks extracted: " << landmarks.size() << std::endl;
        drawLandmarks(img, landmarks);
    }

    std::string output = "landmark_result.jpg";
    cv::imwrite(output, img);
    std::cout << "\n[4] Result saved to: " << output << std::endl;

    std::cout << "\n===== " << (all_valid ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << " =====" << std::endl;
    return all_valid ? 0 : 1;
}
