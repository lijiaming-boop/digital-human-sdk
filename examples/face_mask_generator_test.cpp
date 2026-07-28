#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "core/face_mask_generator.h"

using namespace digital_human::core;

// 辅助函数：创建模拟的 68 点关键点（在指定图像尺寸内）
// 嘴部关键点索引 48~67 构成简单的椭圆形
std::vector<cv::Point> createSyntheticLandmarks(int img_w, int img_h) {
    std::vector<cv::Point> pts(68);

    float cx = img_w / 2.0f;
    float cy = img_h / 2.0f;

    // 下巴轮廓 (0-16)
    for (int i = 0; i <= 16; ++i) {
        float angle = -0.3f * 3.14159f + (i / 16.0f) * 0.6f * 3.14159f;
        pts[i] = cv::Point(static_cast<int>(cx + 0.3f * img_w * std::cos(angle)),
                           static_cast<int>(cy + 0.4f * img_h + 0.3f * img_h * std::sin(angle)));
    }

    // 眉毛 (17-26)
    for (int i = 0; i < 5; ++i) {
        pts[17 + i] = cv::Point(static_cast<int>(cx - 0.15f * img_w + i * 0.06f * img_w),
                                static_cast<int>(cy - 0.25f * img_h));
        pts[22 + i] = cv::Point(static_cast<int>(cx + 0.03f * img_w + i * 0.06f * img_w),
                                static_cast<int>(cy - 0.25f * img_h));
    }

    // 鼻子 (27-35)
    for (int i = 0; i < 9; ++i) {
        pts[27 + i] = cv::Point(static_cast<int>(cx - 0.04f * img_w + i * 0.01f * img_w),
                                static_cast<int>(cy - 0.05f * img_h + i * 0.04f * img_h));
    }

    // 眼睛 (36-47)
    for (int i = 0; i < 6; ++i) {
        float angle = i / 5.0f * 2.0f * 3.14159f;
        pts[36 + i] = cv::Point(static_cast<int>(cx - 0.12f * img_w + 0.06f * img_w * std::cos(angle)),
                                static_cast<int>(cy - 0.12f * img_h + 0.03f * img_h * std::sin(angle)));
        pts[42 + i] = cv::Point(static_cast<int>(cx + 0.06f * img_w + 0.06f * img_w * std::cos(angle)),
                                static_cast<int>(cy - 0.12f * img_h + 0.03f * img_h * std::sin(angle)));
    }

    // 嘴部外轮廓 (48-59): 椭圆形
    float mouth_cx = cx;
    float mouth_cy = cy + 0.28f * img_h;
    float mouth_rx = 0.12f * img_w;
    float mouth_ry = 0.06f * img_h;
    for (int i = 0; i < 12; ++i) {
        float angle = i / 12.0f * 2.0f * 3.14159f;
        pts[48 + i] = cv::Point(
            static_cast<int>(mouth_cx + mouth_rx * std::cos(angle)),
            static_cast<int>(mouth_cy + mouth_ry * std::sin(angle)));
    }

    // 嘴部内轮廓 (60-67): 较小的椭圆
    float inner_rx = 0.07f * img_w;
    float inner_ry = 0.03f * img_h;
    for (int i = 0; i < 8; ++i) {
        float angle = i / 8.0f * 2.0f * 3.14159f;
        pts[60 + i] = cv::Point(
            static_cast<int>(mouth_cx + inner_rx * std::cos(angle)),
            static_cast<int>(mouth_cy + inner_ry * std::sin(angle)));
    }

    return pts;
}

// 辅助函数：创建 96x96 空间内的模拟关键点
std::vector<cv::Point2f> createSyntheticLandmarks96() {
    std::vector<cv::Point2f> pts(68);

    // 简化的 96x96 空间分布
    for (int i = 0; i <= 16; ++i) {
        pts[i] = cv::Point2f(48.0f + 30.0f * std::cos(-0.3f * 3.14159f + i / 16.0f * 0.6f * 3.14159f),
                             60.0f + 35.0f * std::sin(-0.3f * 3.14159f + i / 16.0f * 0.6f * 3.14159f));
    }

    for (int i = 0; i < 5; ++i) {
        pts[17 + i] = cv::Point2f(30.0f + i * 5.0f, 25.0f);
        pts[22 + i] = cv::Point2f(50.0f + i * 5.0f, 25.0f);
    }

    for (int i = 0; i < 9; ++i) {
        pts[27 + i] = cv::Point2f(44.0f + i * 1.0f, 35.0f + i * 4.0f);
    }

    for (int i = 0; i < 6; ++i) {
        float a = i / 5.0f * 2.0f * 3.14159f;
        pts[36 + i] = cv::Point2f(38.0f + 6.0f * std::cos(a), 35.0f + 3.0f * std::sin(a));
        pts[42 + i] = cv::Point2f(58.0f + 6.0f * std::cos(a), 35.0f + 3.0f * std::sin(a));
    }

    // 嘴部 (48-67)
    for (int i = 0; i < 12; ++i) {
        float a = i / 12.0f * 2.0f * 3.14159f;
        pts[48 + i] = cv::Point2f(48.0f + 12.0f * std::cos(a), 68.0f + 6.0f * std::sin(a));
    }
    for (int i = 0; i < 8; ++i) {
        float a = i / 8.0f * 2.0f * 3.14159f;
        pts[60 + i] = cv::Point2f(48.0f + 7.0f * std::cos(a), 68.0f + 3.0f * std::sin(a));
    }

    return pts;
}

// 辅助：检查 mask 是否全为零
bool isAllZero(const cv::Mat& mask) {
    if (mask.empty()) return true;
    return cv::countNonZero(mask) == 0;
}

// 辅助：检查值是否在 [0, 1] 范围内
bool valuesInRange(const cv::Mat& mask, float lo, float hi) {
    if (mask.empty()) return true;
    double min_val, max_val;
    cv::minMaxLoc(mask, &min_val, &max_val);
    return min_val >= lo - 1e-5f && max_val <= hi + 1e-5f;
}

// ============================================================
// Test 1: generateMouthMask 正常情况
// ============================================================
void testGenerateMouthMaskNormal() {
    std::cout << "\n[Test 1] generateMouthMask normal cases..." << std::endl;

    FaceMaskGenerator gen;

    const int sizes[][2] = { {128, 128}, {256, 256}, {512, 512}, {640, 480} };

    for (const auto& sz : sizes) {
        cv::Size image_size(sz[0], sz[1]);
        auto landmarks = createSyntheticLandmarks(sz[0], sz[1]);

        // 1a. 默认参数 (dilate=5, blur=15)
        cv::Mat mask = gen.generateMouthMask(image_size, landmarks);
        bool dim_ok = (mask.cols == sz[0] && mask.rows == sz[1]);
        std::cout << "  size=" << sz[0] << "x" << sz[1]
                  << " dims_ok=" << (dim_ok ? "PASS" : "FAIL")
                  << " type=CV_32FC1? " << (mask.type() == CV_32FC1 ? "PASS" : "FAIL")
                  << " nonzero=" << (isAllZero(mask) ? "FAIL(all_zero)" : "PASS")
                  << " range_ok=" << (valuesInRange(mask, 0.0f, 1.0f) ? "PASS" : "FAIL")
                  << std::endl;

        // 1b. dilate_radius=0
        mask = gen.generateMouthMask(image_size, landmarks, 0, 15);
        std::cout << "    dilate=0: nonzero="
                  << (isAllZero(mask) ? "FAIL(all_zero)" : "PASS") << std::endl;

        // 1c. blur_sigma=0
        mask = gen.generateMouthMask(image_size, landmarks, 5, 0);
        std::cout << "    blur=0: nonzero="
                  << (isAllZero(mask) ? "FAIL(all_zero)" : "PASS") << std::endl;

        // 1d. 两者都为零
        mask = gen.generateMouthMask(image_size, landmarks, 0, 0);
        std::cout << "    dilate=0,blur=0: nonzero="
                  << (isAllZero(mask) ? "FAIL(all_zero)" : "PASS")
                  << " range_ok=" << (valuesInRange(mask, 0.0f, 1.0f) ? "PASS" : "FAIL")
                  << std::endl;

        // 1e. 大扩展半径
        mask = gen.generateMouthMask(image_size, landmarks, 20, 15);
        std::cout << "    dilate=20: nonzero="
                  << (isAllZero(mask) ? "FAIL(all_zero)" : "PASS") << std::endl;

        // 1f. 偶数 blur_sigma (会被自动纠正为奇数)
        mask = gen.generateMouthMask(image_size, landmarks, 5, 10);
        std::cout << "    blur=10(even): nonzero="
                  << (isAllZero(mask) ? "FAIL(all_zero)" : "PASS")
                  << " range_ok=" << (valuesInRange(mask, 0.0f, 1.0f) ? "PASS" : "FAIL")
                  << std::endl;
    }
}

// ============================================================
// Test 2: generateMouthMask 边界条件
// ============================================================
void testGenerateMouthMaskEdgeCases() {
    std::cout << "\n[Test 2] generateMouthMask edge cases..." << std::endl;

    FaceMaskGenerator gen;
    cv::Size size(256, 256);

    // 2a. 关键点不足 68 个
    std::vector<cv::Point> few_landmarks(50);
    cv::Mat mask = gen.generateMouthMask(size, few_landmarks);
    std::cout << "  <68 landmarks: "
              << (isAllZero(mask) ? "PASS (returns zero mask)" : "FAIL")
              << std::endl;

    // 2b. 空关键点
    std::vector<cv::Point> empty_landmarks;
    mask = gen.generateMouthMask(size, empty_landmarks);
    std::cout << "  empty landmarks: "
              << (isAllZero(mask) ? "PASS (returns zero mask)" : "FAIL")
              << std::endl;

    // 2c. 极小图像
    cv::Size tiny_size(16, 16);
    auto landmarks = createSyntheticLandmarks(16, 16);
    mask = gen.generateMouthMask(tiny_size, landmarks);
    std::cout << "  tiny image (16x16): dims=" << mask.cols << "x" << mask.rows
              << " " << ((mask.cols == 16 && mask.rows == 16) ? "PASS" : "FAIL")
              << " type=" << (mask.type() == CV_32FC1 ? "PASS" : "FAIL")
              << std::endl;

    // 2d. 非正方形图像
    cv::Size rect_size(640, 360);
    landmarks = createSyntheticLandmarks(640, 360);
    mask = gen.generateMouthMask(rect_size, landmarks);
    std::cout << "  rect image (640x360): dims=" << mask.cols << "x" << mask.rows
              << " " << ((mask.cols == 640 && mask.rows == 360) ? "PASS" : "FAIL")
              << " nonzero=" << (isAllZero(mask) ? "FAIL" : "PASS")
              << std::endl;
}

// ============================================================
// Test 3: generatePreciseMouthAlphaMask96 正常情况
// ============================================================
void testGeneratePreciseMouthMask96Normal() {
    std::cout << "\n[Test 3] generatePreciseMouthAlphaMask96 normal cases..."
              << std::endl;

    FaceMaskGenerator gen;
    auto landmarks = createSyntheticLandmarks96();

    cv::Mat mask = gen.generatePreciseMouthAlphaMask96(landmarks);

    std::cout << "  dims=96x96? "
              << ((mask.cols == 96 && mask.rows == 96) ? "PASS" : "FAIL")
              << " (actual: " << mask.cols << "x" << mask.rows << ")"
              << std::endl;

    std::cout << "  type=CV_32FC1? "
              << (mask.type() == CV_32FC1 ? "PASS" : "FAIL") << std::endl;

    std::cout << "  nonzero: " << (isAllZero(mask) ? "FAIL(all_zero)" : "PASS")
              << std::endl;

    std::cout << "  range [0,1]: "
              << (valuesInRange(mask, 0.0f, 1.0f) ? "PASS" : "FAIL") << std::endl;

    // 检查嘴部区域确实有非零值
    cv::Rect mouth_roi(24, 56, 48, 32);
    cv::Mat mouth_region = mask(mouth_roi);
    std::cout << "  mouth region nonzero: "
              << (cv::countNonZero(mouth_region) > 0 ? "PASS" : "FAIL")
              << std::endl;

    // 检查顶部区域（非嘴部）接近零
    cv::Mat top_region = mask(cv::Rect(0, 0, 96, 20));
    double top_max;
    cv::minMaxLoc(top_region, nullptr, &top_max);
    std::cout << "  top region max: " << top_max
              << " (should be near 0): "
              << (top_max < 0.1 ? "PASS" : "WARN") << std::endl;
}

// ============================================================
// Test 4: generatePreciseMouthAlphaMask96 边界条件
// ============================================================
void testGeneratePreciseMouthMask96EdgeCases() {
    std::cout << "\n[Test 4] generatePreciseMouthAlphaMask96 edge cases..."
              << std::endl;

    FaceMaskGenerator gen;

    // 4a. 关键点不足 68 个
    std::vector<cv::Point2f> few(50, cv::Point2f(48.0f, 48.0f));
    cv::Mat mask = gen.generatePreciseMouthAlphaMask96(few);
    std::cout << "  <68 landmarks: "
              << (isAllZero(mask) ? "PASS (returns zero mask)" : "FAIL")
              << " dims=" << mask.cols << "x" << mask.rows
              << std::endl;

    // 4b. 空关键点
    std::vector<cv::Point2f> empty;
    mask = gen.generatePreciseMouthAlphaMask96(empty);
    std::cout << "  empty landmarks: "
              << (isAllZero(mask) ? "PASS (returns zero mask)" : "FAIL")
              << std::endl;

    // 4c. 退化关键点（所有点在同一位置，凸包 < 3）
    std::vector<cv::Point2f> degenerate(68, cv::Point2f(48.0f, 48.0f));
    mask = gen.generatePreciseMouthAlphaMask96(degenerate);
    std::cout << "  degenerate (all same point): "
              << (isAllZero(mask) ? "PASS (returns zero mask)" : "FAIL")
              << std::endl;

    // 4d. 两个点构成的嘴部（凸包 < 3）
    std::vector<cv::Point2f> two_point_mouth(68);
    for (int i = 0; i < 68; ++i) {
        two_point_mouth[i] = (i >= 48 && i <= 57) ? cv::Point2f(40.0f, 68.0f)
                                                   : cv::Point2f(56.0f, 68.0f);
    }
    mask = gen.generatePreciseMouthAlphaMask96(two_point_mouth);
    std::cout << "  two-point mouth (hull<3): "
              << (isAllZero(mask) ? "PASS (returns zero mask)" : "FAIL")
              << std::endl;

    // 4e. 嘴部在图像边缘
    std::vector<cv::Point2f> edge_mouth(68);
    for (int i = 0; i < 68; ++i) {
        edge_mouth[i] = cv::Point2f(48.0f, 48.0f);  // 非嘴部占位
    }
    // 嘴部靠近顶部边缘
    for (int i = 48; i <= 67; ++i) {
        edge_mouth[i] = cv::Point2f(48.0f + (i - 48) * 1.0f, 3.0f + (i % 3));
    }
    mask = gen.generatePreciseMouthAlphaMask96(edge_mouth);
    std::cout << "  mouth near top edge: dims=" << mask.cols << "x" << mask.rows
              << " " << ((mask.cols == 96 && mask.rows == 96) ? "PASS" : "FAIL")
              << " type=" << (mask.type() == CV_32FC1 ? "PASS" : "FAIL")
              << " range_ok=" << (valuesInRange(mask, 0.0f, 1.0f) ? "PASS" : "FAIL")
              << std::endl;
}

// ============================================================
// Test 5: to3ChannelMask
// ============================================================
void testTo3ChannelMask() {
    std::cout << "\n[Test 5] to3ChannelMask..." << std::endl;

    FaceMaskGenerator gen;

    // 5a. 空输入
    cv::Mat empty;
    cv::Mat result = gen.to3ChannelMask(empty);
    std::cout << "  empty input: "
              << (result.empty() ? "PASS" : "FAIL") << std::endl;

    // 5b. CV_8UC1 输入
    cv::Mat u8_mask(100, 100, CV_8UC1, cv::Scalar(128));
    result = gen.to3ChannelMask(u8_mask);
    std::cout << "  CV_8UC1 input: "
              << "channels=" << result.channels()
              << " " << (result.channels() == 3 ? "PASS" : "FAIL")
              << " type=" << (result.type() == CV_32FC3 ? "PASS" : "FAIL");
    // 检查归一化值 (128/255 ≈ 0.502)
    double mid_val = result.at<cv::Vec3f>(50, 50)[0];
    std::cout << " value@center=" << mid_val
              << " (expect ~0.502): "
              << (std::abs(mid_val - 128.0 / 255.0) < 0.01 ? "PASS" : "FAIL")
              << std::endl;

    // 5c. CV_32FC1 输入 (不需要归一化)
    cv::Mat f32_mask(100, 100, CV_32FC1, cv::Scalar(0.75f));
    result = gen.to3ChannelMask(f32_mask);
    std::cout << "  CV_32FC1 input: "
              << "channels=" << result.channels()
              << " " << (result.channels() == 3 ? "PASS" : "FAIL")
              << " type=" << (result.type() == CV_32FC3 ? "PASS" : "FAIL");
    double val = result.at<cv::Vec3f>(50, 50)[0];
    std::cout << " value@center=" << val
              << " (expect 0.75): "
              << (std::abs(val - 0.75f) < 0.01 ? "PASS" : "FAIL")
              << std::endl;

    // 5d. 验证三个通道值完全一致
    cv::Mat f32_single(50, 50, CV_32FC1, cv::Scalar(0.3f));
    result = gen.to3ChannelMask(f32_single);
    cv::Vec3f px = result.at<cv::Vec3f>(25, 25);
    bool all_same = (std::abs(px[0] - px[1]) < 1e-6f) &&
                    (std::abs(px[1] - px[2]) < 1e-6f);
    std::cout << "  all 3 channels equal: " << (all_same ? "PASS" : "FAIL")
              << " (" << px[0] << ", " << px[1] << ", " << px[2] << ")"
              << std::endl;
}

// ============================================================
// Test 6: 与 generateMouthMask 结合使用
// ============================================================
void testIntegration() {
    std::cout << "\n[Test 6] Integration: generateMouthMask + to3ChannelMask..."
              << std::endl;

    FaceMaskGenerator gen;
    auto landmarks = createSyntheticLandmarks(256, 256);

    cv::Mat alpha = gen.generateMouthMask(cv::Size(256, 256), landmarks, 5, 15);
    cv::Mat alpha_3c = gen.to3ChannelMask(alpha);

    std::cout << "  alpha dims=" << alpha.cols << "x" << alpha.rows
              << " channels=" << alpha.channels()
              << std::endl;
    std::cout << "  alpha_3c dims=" << alpha_3c.cols << "x" << alpha_3c.rows
              << " channels=" << alpha_3c.channels()
              << std::endl;

    // 验证 alpha 和 alpha_3c 每个通道的对应值一致
    cv::Mat alpha_f;
    alpha.convertTo(alpha_f, CV_32FC1, 1.0);  // 复制
    cv::Vec3f px_3c = alpha_3c.at<cv::Vec3f>(128, 128);
    float px_1c = alpha_f.at<float>(128, 128);
    bool consistent = (std::abs(px_3c[0] - px_1c) < 1e-6f);
    std::cout << "  value consistency: " << (consistent ? "PASS" : "FAIL")
              << std::endl;

    // 测试将 alpha 应用到实际图像
    cv::Mat fake_img(256, 256, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::Mat img_float;
    fake_img.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

    cv::Mat blended;
    cv::multiply(img_float, alpha_3c, blended);

    double min_val, max_val;
    cv::minMaxLoc(blended, &min_val, &max_val);
    std::cout << "  blended range: [" << min_val << ", " << max_val
              << "] " << (min_val >= 0.0 && max_val <= 1.0 ? "PASS" : "FAIL")
              << std::endl;

    // 保存一张可视化结果（调试用）
    cv::Mat vis_mask;
    alpha.convertTo(vis_mask, CV_8UC1, 255.0);
    cv::imwrite("face_mask_debug.png", vis_mask);
    std::cout << "  Saved visualization: face_mask_debug.png" << std::endl;
}

int main() {
    std::cout << "===== FaceMaskGenerator Test =====" << std::endl;

    testGenerateMouthMaskNormal();
    testGenerateMouthMaskEdgeCases();
    testGeneratePreciseMouthMask96Normal();
    testGeneratePreciseMouthMask96EdgeCases();
    testTo3ChannelMask();
    testIntegration();

    std::cout << "\n===== All Tests Done =====" << std::endl;
    return 0;
}
