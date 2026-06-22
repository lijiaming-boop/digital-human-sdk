#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "core/face_detector.h"
#include "core/face_aligner.h"

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

void drawLandmarks(cv::Mat& img, const std::vector<cv::Point2f>& landmarks,
                   const cv::Scalar& color = GREEN) {
    for (size_t i = 0; i < landmarks.size(); ++i) {
        cv::circle(img, landmarks[i], 1, color, -1);
    }
}

void testAlignByEyes(const cv::Mat& image,
                     const std::vector<cv::Point>& landmarks_pts) {
    std::cout << "\n[Test 1] align (eye-based)..." << std::endl;

    if (landmarks_pts.size() != 68) {
        std::cout << "  SKIP: need 68 landmarks, got " << landmarks_pts.size()
                  << std::endl;
        return;
    }

    std::vector<cv::Point2f> landmarks;
    for (const auto& p : landmarks_pts) {
        landmarks.push_back(cv::Point2f(static_cast<float>(p.x),
                                         static_cast<float>(p.y)));
    }

    FaceAlignigner aligner;

    for (int size : {96, 128, 256}) {
        cv::Mat aligned = aligner.align(image, landmarks, size);
        if (aligned.empty()) {
            std::cout << "  FAIL: aligned image is empty (face_size=" << size
                      << ")" << std::endl;
        } else {
            std::cout << "  PASS: face_size=" << size
                      << " output=" << aligned.cols << "x" << aligned.rows
                      << std::endl;
            std::string filename =
                "aligned_eyes_" + std::to_string(size) + ".jpg";
            cv::imwrite(filename, aligned);
            std::cout << "    Saved: " << filename << std::endl;
        }
    }
}

void testAlignByRect(const cv::Mat& image,
                     const std::vector<cv::Point>& landmarks_pts,
                     const std::vector<cv::Rect>& faces) {
    std::cout << "\n[Test 2] alignByRect..." << std::endl;

    if (landmarks_pts.size() != 68) {
        std::cout << "  SKIP: need 68 landmarks" << std::endl;
        return;
    }
    if (faces.empty()) {
        std::cout << "  SKIP: no face rect" << std::endl;
        return;
    }

    std::vector<cv::Point2f> landmarks;
    for (const auto& p : landmarks_pts) {
        landmarks.push_back(
            cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
    }

    FaceAlignigner aligner;

    for (double ratio : {0.0, 0.15, 0.3}) {
        FaceAlignerResult result =
            aligner.alignByRect(image, landmarks, 128, faces[0], ratio);

        if (!result.valid) {
            std::cout << "  FAIL: result invalid (ratio=" << ratio << ")"
                      << std::endl;
            continue;
        }

        std::cout << "  PASS: ratio=" << ratio
                  << " output=" << result.aligned_face.cols << "x"
                  << result.aligned_face.rows
                  << " expanded_rect=(" << result.face_rect.x << ","
                  << result.face_rect.y << ") " << result.face_rect.width << "x"
                  << result.face_rect.height << std::endl;

        std::string filename =
            "aligned_rect_" + std::to_string(static_cast<int>(ratio * 100)) + ".jpg";
        cv::imwrite(filename, result.aligned_face);
        std::cout << "    Saved: " << filename << std::endl;

        std::cout << "    Transformed landmarks count: "
                  << result.landmarks.size() << std::endl;

        double det = cv::determinant(result.M(cv::Rect(0, 0, 2, 2)));
        std::cout << "    M 2x2 determinant: " << det
                  << " (scale*scale, should be >0)" << std::endl;

        bool invertible = !result.M_inv.empty();
        std::cout << "    M_inv valid: " << (invertible ? "YES" : "NO")
                  << std::endl;
    }
}

void testTransformLandmarks(const std::vector<cv::Point>& landmarks_pts) {
    std::cout << "\n[Test 3] transform_landmarks..." << std::endl;

    if (landmarks_pts.size() != 68) {
        std::cout << "  SKIP: need 68 landmarks" << std::endl;
        return;
    }

    std::vector<cv::Point2f> landmarks;
    for (const auto& p : landmarks_pts) {
        landmarks.push_back(
            cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
    }

    FaceAlignigner aligner;

    cv::Mat identity = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    std::vector<cv::Point2f> transformed =
        aligner.transform_landmarks(landmarks, identity);

    if (transformed.size() != landmarks.size()) {
        std::cout << "  FAIL: count mismatch" << std::endl;
        return;
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        float dx = std::abs(transformed[i].x - landmarks[i].x);
        float dy = std::abs(transformed[i].y - landmarks[i].y);
        max_diff = std::max(max_diff, std::max(dx, dy));
    }
    std::cout << "  Identity transform max error: " << max_diff
              << " (should be ~0)" << std::endl;

    cv::Mat translate = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 50.0, 0.0, 1.0,
                         30.0);
    transformed = aligner.transform_landmarks(landmarks, translate);
    float tx_diff = std::abs(transformed[30].x - (landmarks[30].x + 50.0f));
    float ty_diff = std::abs(transformed[30].y - (landmarks[30].y + 30.0f));
    std::cout << "  Translation error (nose tip): dx=" << tx_diff
              << " dy=" << ty_diff << " (should be ~0)" << std::endl;

    cv::Mat scale2x = (cv::Mat_<double>(2, 3) << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0);
    transformed = aligner.transform_landmarks(landmarks, scale2x);
    float sx_diff = std::abs(transformed[30].x - 2.0f * landmarks[30].x);
    float sy_diff = std::abs(transformed[30].y - 2.0f * landmarks[30].y);
    std::cout << "  Scale 2x error (nose tip): dx=" << sx_diff
              << " dy=" << sy_diff << std::endl;
}

void testEdgeCases() {
    std::cout << "\n[Test 4] Edge cases..." << std::endl;

    FaceAlignigner aligner;
    cv::Mat empty_img;
    std::vector<cv::Point2f> empty_lm;

    cv::Mat result = aligner.align(empty_img, empty_lm, 96);
    std::cout << "  Empty image + align: "
              << (result.empty() ? "PASS (empty)" : "FAIL") << std::endl;

    FaceAlignerResult r2 =
        aligner.alignByRect(empty_img, empty_lm, 96, cv::Rect(), 0.15);
    std::cout << "  Empty image + alignByRect: "
              << (!r2.valid ? "PASS (invalid)" : "FAIL") << std::endl;

    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<cv::Point2f> wrong_lm(10,
                                       cv::Point2f(50.0f, 50.0f));
    result = aligner.align(img, wrong_lm, 96);
    std::cout << "  Wrong landmark count + align: "
              << (result.empty() ? "PASS (empty)" : "FAIL") << std::endl;

    FaceAlignerResult r3 = aligner.alignByRect(
        img, wrong_lm, 96, cv::Rect(10, 10, 50, 50), 0.15);
    std::cout << "  Wrong landmark count + alignByRect: "
              << (!r3.valid ? "PASS (invalid)" : "FAIL") << std::endl;

    cv::Rect zero_rect(10, 10, 0, 0);
    FaceAlignerResult r4 =
        aligner.alignByRect(img, empty_lm, 96, zero_rect, 0.15);
    std::cout << "  Zero-size rect + alignByRect: "
              << (!r4.valid ? "PASS (invalid)" : "FAIL") << std::endl;
}

void testAlignByRectVisual(const cv::Mat& image,
                           const std::vector<cv::Point>& landmarks_pts,
                           const std::vector<cv::Rect>& faces) {
    std::cout << "\n[Test 5] alignByRect visual verification..." << std::endl;

    if (landmarks_pts.size() != 68 || faces.empty()) {
        std::cout << "  SKIP" << std::endl;
        return;
    }

    std::vector<cv::Point2f> landmarks;
    for (const auto& p : landmarks_pts) {
        landmarks.push_back(
            cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
    }

    FaceAlignigner aligner;
    FaceAlignerResult result =
        aligner.alignByRect(image, landmarks, 256, faces[0], 0.15);

    if (!result.valid) {
        std::cout << "  FAIL: result invalid" << std::endl;
        return;
    }

    cv::Mat debug_img;
    cv::cvtColor(result.aligned_face, debug_img, cv::COLOR_BGR2RGB);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_RGB2BGR);

    drawLandmarks(debug_img, result.landmarks, GREEN);
    cv::imwrite("aligned_with_landmarks.jpg", debug_img);
    std::cout << "  Saved aligned face with landmarks: aligned_with_landmarks.jpg"
              << std::endl;

    cv::Mat original_viz = image.clone();
    cv::rectangle(original_viz, faces[0], BLUE, 2);
    cv::rectangle(original_viz, result.face_rect, RED, 2);
    cv::imwrite("original_with_rects.jpg", original_viz);
    std::cout << "  Saved original with face rects: original_with_rects.jpg"
              << std::endl;
    std::cout << "  Blue = original face rect, Red = expanded rect" << std::endl;

    float in_bounds = 0;
    for (const auto& lm : result.landmarks) {
        if (lm.x >= 0 && lm.x < 256 && lm.y >= 0 && lm.y < 256) {
            in_bounds++;
        }
    }
    float ratio = in_bounds / result.landmarks.size();
    std::cout << "  Landmarks inside aligned image: " << in_bounds << "/"
              << result.landmarks.size() << " (" << ratio * 100.0f << "%)"
              << std::endl;
    if (ratio > 0.8f) {
        std::cout << "  PASS: most landmarks within bounds" << std::endl;
    } else {
        std::cout << "  WARN: many landmarks outside aligned image" << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "===== Face Aligner Test =====" << std::endl;

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

    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Model: " << model_path << std::endl;

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

    DigitalHuman::core::FaceDetector detector;
    std::cout << "\n[0] Loading model and detecting face..." << std::endl;
    if (!detector.loadModel(model_path)) {
        std::cerr << "Model load failed" << std::endl;
        return 1;
    }

    std::vector<cv::Rect> faces = detector.detect(img);
    std::cout << "  Found " << faces.size() << " face(s)" << std::endl;

    if (faces.empty()) {
        std::cout << "No faces found. Try a clearer frontal face photo."
                  << std::endl;
        return 0;
    }

    std::vector<cv::Point> landmarks = detector.getLandmarks(img, faces[0]);
    std::cout << "  Landmarks extracted: " << landmarks.size() << std::endl;

    if (landmarks.size() != 68) {
        std::cerr << "Expected 68 landmarks, got " << landmarks.size()
                  << std::endl;
        return 1;
    }

    testEdgeCases();
    testTransformLandmarks(landmarks);
    testAlignByEyes(img, landmarks);
    testAlignByRect(img, landmarks, faces);
    testAlignByRectVisual(img, landmarks, faces);

    std::cout << "\n===== All Tests Done =====" << std::endl;
    return 0;
}
