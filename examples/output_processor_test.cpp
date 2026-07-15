/**
 * @file output_processor_test.cpp
 * @brief OutputProcessor 模块验收测试
 *
 * 覆盖范围：
 * - ncnn::Mat → cv::Mat 转换（格式、通道、像素值）
 * - 人脸区域提取（裁剪、缩放）
 * - 逆变换（M_inv 映射回原始坐标）
 * - 人脸融合（alpha blending）
 * - 后处理（锐化、色彩融合）
 * - 全流程管道
 * - 空输入/边界处理
 * - Move 语义
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>

#include "model/output_processor.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <ncnn/mat.h>

using namespace digital_human::model;

// ============================================================================
// 辅助函数
// ============================================================================

static int gPassed = 0;
static int gFailed = 0;

#define TEST_NAME(name) \
    std::cout << "\n====== " << name << " ======" << std::endl;

#define TEST_CHECK(cond, desc)                          \
    do {                                                \
        if (cond) {                                     \
            std::cout << "  [PASS] " << desc << std::endl; \
            gPassed++;                                   \
        } else {                                         \
            std::cout << "  [FAIL] " << desc << std::endl; \
            gFailed++;                                   \
        }                                                \
    } while (0)

/// @brief 生成模拟模型输出的 ncnn::Mat (RGB float, [0,1])
static ncnn::Mat createModelOutput(int w = 448, int h = 96, bool extreme = false) {
    ncnn::Mat mat(w, h, 3);
    for (int c = 0; c < 3; ++c) {
        float* ch = mat.channel(c);
        for (int i = 0; i < w * h; ++i) {
            if (extreme) {
                // 极端值：0 或 1
                ch[i] = (i % 2 == 0) ? 0.0f : 1.0f;
            } else {
                ch[i] = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
    }
    return mat;
}

/// @brief 生成模拟人脸图像 (BGR uint8)
static cv::Mat createFaceImage(int w = 192, int h = 192, uchar val = 128) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(val, val, val));
    // 绘制一些渐变模式以便视觉验证
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(val + x * 50 / w),
                static_cast<uchar>(val + y * 50 / h),
                static_cast<uchar>(val)
            );
        }
    }
    return img;
}

/// @brief 生成模拟人脸遮罩 (CV_32FC1, [0,1])
static cv::Mat createFaceMask(int w = 192, int h = 192, float fill = 0.5f) {
    cv::Mat mask(h, w, CV_32FC1, cv::Scalar(fill));
    return mask;
}

/// @brief 生成模拟逆仿射变换矩阵 M_inv (2×3)
static cv::Mat createMInv() {
    // 单位仿射变换（不做变换）
    cv::Mat M_inv = (cv::Mat_<double>(2, 3) <<
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0);
    return M_inv;
}

/// @brief 生成带旋转缩放的 M_inv
static cv::Mat createMInvNonTrivial() {
    // 缩放 0.8 + 旋转 5 度
    double angle = 5.0 * CV_PI / 180.0;
    double s = 0.8;
    cv::Mat M = (cv::Mat_<double>(2, 3) <<
        s * cos(angle), -s * sin(angle), 10.0,
        s * sin(angle),  s * cos(angle), 5.0);
    return M;
}

// ============================================================================
// 测试用例
// ============================================================================

// ---- Test 1: ncnn → cv::Mat 转换 ----
static void testNcnnToCvMat() {
    TEST_NAME("Test 1: ncnn::Mat → cv::Mat 转换");

    OutputProcessor proc;

    // 1.1 正常转换
    ncnn::Mat ncnn_out = createModelOutput(96, 96);
    cv::Mat cv_out = proc.OutputToMat(ncnn_out, 96, 96);
    TEST_CHECK(!cv_out.empty(), "1.1 转换结果不为空");
    TEST_CHECK(cv_out.rows == 96, "1.1 高度 == 96");
    TEST_CHECK(cv_out.cols == 96, "1.1 宽度 == 96");
    TEST_CHECK(cv_out.channels() == 3, "1.1 通道数 == 3");
    TEST_CHECK(cv_out.type() == CV_8UC3, "1.1 类型为 CV_8UC3");

    // 1.2 像素值验证：值域 [0, 255]
    bool inRange = true;
    for (int y = 0; y < cv_out.rows && inRange; ++y) {
        for (int x = 0; x < cv_out.cols && inRange; ++x) {
            auto px = cv_out.at<cv::Vec3b>(y, x);
            if (px[0] > 255 || px[1] > 255 || px[2] > 255) {
                inRange = false;
            }
        }
    }
    TEST_CHECK(inRange, "1.2 像素值在 [0, 255] 范围内");

    // 1.3 空输入
    ncnn::Mat empty_ncnn;
    cv::Mat empty_out = proc.OutputToMat(empty_ncnn, 96, 96);
    TEST_CHECK(empty_out.empty(), "1.3 空 ncnn::Mat → 空 cv::Mat");

    // 1.4 通道数不匹配
    ncnn::Mat wrong_ch(10, 10, 1);  // 1 通道
    cv::Mat wrong_out = proc.OutputToMat(wrong_ch, 10, 10);
    TEST_CHECK(wrong_out.empty(), "1.4 通道数不为 3 → 空 cv::Mat");
}

// ---- Test 2: 人脸区域提取 ----
static void testFaceRegionExtract() {
    TEST_NAME("Test 2: 人脸区域提取");

    OutputProcessor proc;

    // 2.1 从 448×96 输出中提取 96×96 人脸
    ncnn::Mat wide_out = createModelOutput(448, 96);
    cv::Mat face = proc.OutputToMat(wide_out, 96, 96);
    TEST_CHECK(!face.empty(), "2.1 提取结果不为空");
    TEST_CHECK(face.cols == 96, "2.1 宽度 == 96");
    TEST_CHECK(face.rows == 96, "2.1 高度 == 96");

    // 2.2 从 96×96 输出直接提取
    ncnn::Mat exact_out = createModelOutput(96, 96);
    cv::Mat exact_face = proc.OutputToMat(exact_out, 96, 96);
    TEST_CHECK(!exact_face.empty(), "2.2 精确尺寸提取不为空");

    // 2.3 缩放情况（输出小于目标）
    ncnn::Mat small_out = createModelOutput(48, 48);
    cv::Mat scaled_face = proc.OutputToMat(small_out, 96, 96);
    TEST_CHECK(!scaled_face.empty(), "2.3 缩放到 96×96");
    TEST_CHECK(scaled_face.cols == 96, "2.3 缩放后宽度 == 96");
    TEST_CHECK(scaled_face.rows == 96, "2.3 缩放后高度 == 96");

    // 2.4 无效尺寸参数
    ncnn::Mat valid_out = createModelOutput(96, 96);
    cv::Mat bad_size = proc.OutputToMat(valid_out, 0, 0);
    TEST_CHECK(bad_size.empty(), "2.4 face_w=0 → 空 Mat");
}

// ---- Test 3: 逆变换 ----
static void testInverseTransform() {
    TEST_NAME("Test 3: 逆变换");

    OutputProcessor proc;

    cv::Mat face = createFaceImage(96, 96);
    cv::Mat M_inv = createMInv();
    cv::Size orig_size(192, 192);

    // 3.1 单位变换（恒等映射）
    cv::Mat result = proc.InverseTransform(face, M_inv, orig_size);
    TEST_CHECK(!result.empty(), "3.1 逆变换结果不为空");
    TEST_CHECK(result.cols == orig_size.width, "3.1 宽度匹配原始尺寸");
    TEST_CHECK(result.rows == orig_size.height, "3.1 高度匹配原始尺寸");

    // 3.2 非平凡变换（旋转+缩放）
    cv::Mat M_inv2 = createMInvNonTrivial();
    cv::Mat result2 = proc.InverseTransform(face, M_inv2, orig_size);
    TEST_CHECK(!result2.empty(), "3.2 非平凡变换结果不为空");
    TEST_CHECK(result2.cols == orig_size.width, "3.2 宽度匹配");
    TEST_CHECK(result2.rows == orig_size.height, "3.2 高度匹配");

    // 3.3 空输入
    cv::Mat empty_face;
    cv::Mat empty_result = proc.InverseTransform(empty_face, M_inv, orig_size);
    TEST_CHECK(empty_result.empty(), "3.3 空人脸 → 空 Mat");

    // 3.4 空 M_inv
    cv::Mat empty_M;
    cv::Mat empty_M_result = proc.InverseTransform(face, empty_M, orig_size);
    TEST_CHECK(empty_M_result.empty(), "3.4 空 M_inv → 空 Mat");

    // 3.5 无效尺寸
    cv::Mat bad_size_result = proc.InverseTransform(face, M_inv, cv::Size(0, 0));
    TEST_CHECK(bad_size_result.empty(), "3.5 无效尺寸 → 空 Mat");
}

// ---- Test 4: 人脸融合 ----
static void testFaceFusion() {
    TEST_NAME("Test 4: 人脸融合");

    OutputProcessor proc;

    cv::Mat original = createFaceImage(192, 192, 200);   // 亮色背景
    cv::Mat generated = createFaceImage(192, 192, 50);   // 暗色前景
    cv::Mat mask = createFaceMask(192, 192, 0.5f);       // 半透明遮罩

    // 4.1 基础融合
    cv::Mat fused = proc.FaceFusion(original, generated, mask);
    TEST_CHECK(!fused.empty(), "4.1 融合结果不为空");
    TEST_CHECK(fused.cols == 192, "4.1 宽度 == 192");
    TEST_CHECK(fused.rows == 192, "4.1 高度 == 192");
    TEST_CHECK(fused.channels() == 3, "4.1 通道数 == 3");

    // 4.2 验证融合逻辑：mask=0.5, orig=200, gen=50
    // 理论值: 0.5*50 + 0.5*200 = 125
    cv::Vec3b pixel = fused.at<cv::Vec3b>(0, 0);
    bool blendCorrect = true;
    for (int c = 0; c < 3; ++c) {
        int diff = static_cast<int>(pixel[c]) - 125;
        if (std::abs(diff) > 3) {  // 允许 3 的误差
            blendCorrect = false;
            break;
        }
    }
    TEST_CHECK(blendCorrect, "4.2 融合逻辑正确 (0.5*50 + 0.5*200 ≈ 125)");

    // 4.3 全一遮罩（完全显示生成图像）
    cv::Mat mask_ones = createFaceMask(192, 192, 1.0f);
    cv::Mat fused_ones = proc.FaceFusion(original, generated, mask_ones);
    cv::Vec3b px_one = fused_ones.at<cv::Vec3b>(0, 0);
    bool fullGen = true;
    cv::Vec3b gen_px2 = generated.at<cv::Vec3b>(0, 0);
    for (int c = 0; c < 3; ++c) {
        int diff = static_cast<int>(px_one[c]) - static_cast<int>(gen_px2[c]);
        if (std::abs(diff) > 3) fullGen = false;
    }
    TEST_CHECK(fullGen, "4.3 全一遮罩 → 完全显示生成图像");

    // 4.4 全零遮罩（完全显示原始图像）
    cv::Mat mask_zeros = createFaceMask(192, 192, 0.0f);
    cv::Mat fused_zeros = proc.FaceFusion(original, generated, mask_zeros);
    cv::Vec3b px_zero = fused_zeros.at<cv::Vec3b>(0, 0);
    bool fullOrig = true;
    cv::Vec3b orig_px2 = original.at<cv::Vec3b>(0, 0);
    for (int c = 0; c < 3; ++c) {
        int diff = static_cast<int>(px_zero[c]) - static_cast<int>(orig_px2[c]);
        if (std::abs(diff) > 3) fullOrig = false;
    }
    TEST_CHECK(fullOrig, "4.4 全零遮罩 → 完全显示原始图像");
}

// ---- Test 5: 锐化 ----
static void testSharpen() {
    TEST_NAME("Test 5: 锐化");

    OutputProcessor proc;

    // 使用有边缘的图像（锐化对边缘效果更明显）
    cv::Mat image(96, 96, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::rectangle(image, cv::Rect(20, 20, 56, 56), cv::Scalar(200, 200, 200), -1);
    cv::Mat sharpened = proc.Sharpen(image, 2.0f);

    // 5.1 锐化结果不为空
    TEST_CHECK(!sharpened.empty(), "5.1 锐化结果不为空");
    TEST_CHECK(sharpened.size() == image.size(), "5.1 锐化后尺寸不变");

    // 5.2 锐化后图像与原始不同
    bool different = false;
    for (int y = 0; y < image.rows && !different; ++y) {
        for (int x = 0; x < image.cols && !different; ++x) {
            auto p1 = image.at<cv::Vec3b>(y, x);
            auto p2 = sharpened.at<cv::Vec3b>(y, x);
            int sum_diff = std::abs(static_cast<int>(p1[0]) - static_cast<int>(p2[0]))
                         + std::abs(static_cast<int>(p1[1]) - static_cast<int>(p2[1]))
                         + std::abs(static_cast<int>(p1[2]) - static_cast<int>(p2[2]));
            if (sum_diff > 5) {
                different = true;
            }
        }
    }
    TEST_CHECK(different, "5.2 锐化后图像与原始不同");

    // 5.3 强度为 0 时等于原图
    cv::Mat no_sharpen = proc.Sharpen(image, 0.0f);
    bool same = true;
    for (int y = 0; y < image.rows && same; ++y) {
        for (int x = 0; x < image.cols && same; ++x) {
            auto p1 = image.at<cv::Vec3b>(y, x);
            auto p2 = no_sharpen.at<cv::Vec3b>(y, x);
            int sum_diff = std::abs(static_cast<int>(p1[0]) - static_cast<int>(p2[0]))
                         + std::abs(static_cast<int>(p1[1]) - static_cast<int>(p2[1]))
                         + std::abs(static_cast<int>(p1[2]) - static_cast<int>(p2[2]));
            if (sum_diff > 3) same = false;
        }
    }
    TEST_CHECK(same, "5.3 strength=0 时锐化结果 ≈ 原图");

    // 5.4 空输入
    cv::Mat empty_img;
    cv::Mat empty_result = proc.Sharpen(empty_img, 1.0f);
    TEST_CHECK(empty_result.empty(), "5.4 空输入 → 空 Mat");
}

// ---- Test 6: 色彩融合 ----
static void testColorBlend() {
    TEST_NAME("Test 6: 色彩融合");

    OutputProcessor proc;

    cv::Mat generated = createFaceImage(96, 96, 100);  // 暗色
    cv::Mat original  = createFaceImage(96, 96, 200);  // 亮色

    // 6.1 alpha=0.7 融合
    cv::Mat blended = proc.ColorBlend(generated, original, 0.7f);
    TEST_CHECK(!blended.empty(), "6.1 色彩融合结果不为空");
    TEST_CHECK(blended.size() == original.size(), "6.1 融合后尺寸不变");

    // 6.2 alpha=1.0 完全显示生成图像
    cv::Mat blend_1 = proc.ColorBlend(generated, original, 1.0f);
    cv::Vec3b px1 = blend_1.at<cv::Vec3b>(0, 0);
    cv::Vec3b px_gen = generated.at<cv::Vec3b>(0, 0);
    bool fullGen = true;
    for (int c = 0; c < 3; ++c) {
        int diff = static_cast<int>(px1[c]) - static_cast<int>(px_gen[c]);
        if (std::abs(diff) > 2) fullGen = false;
    }
    TEST_CHECK(fullGen, "6.2 alpha=1.0 → 完全显示生成图像");

    // 6.3 alpha=0.0 完全显示原始图像
    cv::Mat blend_0 = proc.ColorBlend(generated, original, 0.0f);
    cv::Vec3b px0 = blend_0.at<cv::Vec3b>(0, 0);
    cv::Vec3b px_orig = original.at<cv::Vec3b>(0, 0);
    bool fullOrig = true;
    for (int c = 0; c < 3; ++c) {
        int diff = static_cast<int>(px0[c]) - static_cast<int>(px_orig[c]);
        if (std::abs(diff) > 2) fullOrig = false;
    }
    TEST_CHECK(fullOrig, "6.3 alpha=0.0 → 完全显示原始图像");
}

// ---- Test 7: 全流程管道 ----
static void testFullPipeline() {
    TEST_NAME("Test 7: 全流程管道");

    OutputProcessor proc;

    ncnn::Mat model_out = createModelOutput(96, 96);
    cv::Mat original_face = createFaceImage(192, 192);
    cv::Mat face_mask = createFaceMask(192, 192, 0.5f);
    cv::Mat M_inv = createMInv();

    // 7.1 完整管道
    cv::Mat result = proc.Process(model_out, original_face, face_mask, M_inv);
    TEST_CHECK(!result.empty(), "7.1 全流程处理结果不为空");
    TEST_CHECK(result.cols == 192, "7.1 输出宽度 == 192");
    TEST_CHECK(result.rows == 192, "7.1 输出高度 == 192");
    TEST_CHECK(result.channels() == 3, "7.1 输出通道数 == 3");
}

// ---- Test 8: PostProcess ----
static void testPostProcess() {
    TEST_NAME("Test 8: 后处理管道");

    OutputProcessor proc;

    cv::Mat fused = createFaceImage(96, 96, 128);
    cv::Mat original = createFaceImage(96, 96, 200);

    // 8.1 完整后处理（锐化 + 色彩融合）
    cv::Mat result = proc.PostProcess(fused, original, true, true);
    TEST_CHECK(!result.empty(), "8.1 完整后处理结果不为空");
    TEST_CHECK(result.size() == fused.size(), "8.1 后处理尺寸不变");

    // 8.2 仅锐化
    cv::Mat only_sharpen = proc.PostProcess(fused, cv::Mat(), true, false);
    TEST_CHECK(!only_sharpen.empty(), "8.2 仅锐化结果不为空");

    // 8.3 仅色彩融合
    cv::Mat only_blend = proc.PostProcess(fused, original, false, true);
    TEST_CHECK(!only_blend.empty(), "8.3 仅色彩融合结果不为空");

    // 8.4 无后处理
    cv::Mat none = proc.PostProcess(fused, cv::Mat(), false, false);
    TEST_CHECK(!none.empty(), "8.4 无后处理结果不为空");

    // 8.5 空输入
    cv::Mat empty_result = proc.PostProcess(cv::Mat(), cv::Mat(), true, true);
    TEST_CHECK(empty_result.empty(), "8.5 空输入 → 空 Mat");
}

// ---- Test 9: 遮罩边界情况 ----
static void testMaskEdgeCases() {
    TEST_NAME("Test 9: 遮罩边界情况");

    OutputProcessor proc;

    cv::Mat original = createFaceImage(96, 96, 200);
    cv::Mat generated = createFaceImage(96, 96, 50);

    // 9.1 尺寸不匹配的遮罩（自动缩放）
    cv::Mat mask_small(48, 48, CV_32FC1, cv::Scalar(0.5f));
    cv::Mat fused = proc.FaceFusion(original, generated, mask_small);
    TEST_CHECK(!fused.empty(), "9.1 遮罩尺寸不匹配 → 自动缩放");

    // 9.2 遮罩为 CV_8UC1
    cv::Mat mask_8uc1(96, 96, CV_8UC1, cv::Scalar(128));
    cv::Mat fused_8u = proc.FaceFusion(original, generated, mask_8uc1);
    TEST_CHECK(!fused_8u.empty(), "9.2 CV_8UC1 遮罩 → 正常融合");

    // 9.3 全零遮罩
    cv::Mat mask_zero(96, 96, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat fused_zero = proc.FaceFusion(original, generated, mask_zero);
    // 应该等于原始图像
    bool eqOrig = true;
    for (int y = 0; y < 96 && eqOrig; ++y) {
        for (int x = 0; x < 96 && eqOrig; ++x) {
            auto p1 = fused_zero.at<cv::Vec3b>(y, x);
            auto p2 = original.at<cv::Vec3b>(y, x);
            int sum_diff = std::abs(static_cast<int>(p1[0]) - static_cast<int>(p2[0]))
                         + std::abs(static_cast<int>(p1[1]) - static_cast<int>(p2[1]))
                         + std::abs(static_cast<int>(p1[2]) - static_cast<int>(p2[2]));
            if (sum_diff > 5) {
                eqOrig = false;
            }
        }
    }
    TEST_CHECK(eqOrig, "9.3 全零遮罩 → 完全显示原始图像");
}

// ---- Test 10: Move 语义 ----
static void testMoveSemantics() {
    TEST_NAME("Test 10: Move 语义");

    OutputProcessor proc1;
    ncnn::Mat model_out = createModelOutput(96, 96);
    cv::Mat original = createFaceImage(192, 192);
    cv::Mat mask = createFaceMask(192, 192, 0.5f);
    cv::Mat M_inv = createMInv();

    // 先验证 proc1 正常工作
    cv::Mat result1 = proc1.OutputToMat(model_out, 96, 96);
    TEST_CHECK(!result1.empty(), "10.1 原始对象工作正常");

    // Move 构造
    OutputProcessor proc2(std::move(proc1));
    cv::Mat result2 = proc2.OutputToMat(model_out, 96, 96);
    TEST_CHECK(!result2.empty(), "10.2 Move 构造后新对象工作正常");

    // Move 赋值
    OutputProcessor proc3;
    proc3 = std::move(proc2);
    cv::Mat result3 = proc3.OutputToMat(model_out, 96, 96);
    TEST_CHECK(!result3.empty(), "10.3 Move 赋值后新对象工作正常");

    cv::Mat result4 = proc3.InverseTransform(result3, M_inv, original.size());
    TEST_CHECK(!result4.empty(), "10.4 Move 后逆变换正常");

    cv::Mat result5 = proc3.FaceFusion(original, result4, mask);
    TEST_CHECK(!result5.empty(), "10.5 Move 后融合正常");
}

// ---- Test 11: 极端像素值 ----
static void testExtremeValues() {
    TEST_NAME("Test 11: 极端像素值");

    OutputProcessor proc;

    // 11.1 极端值 [0,1] 交替
    ncnn::Mat extreme = createModelOutput(96, 96, true);
    cv::Mat cv_out = proc.OutputToMat(extreme, 96, 96);
    TEST_CHECK(!cv_out.empty(), "11.1 极端值转换成功");

    // 验证像素值要么 0 要么 255
    bool valid = true;
    for (int y = 0; y < cv_out.rows && valid; ++y) {
        for (int x = 0; x < cv_out.cols && valid; ++x) {
            auto px = cv_out.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                if (px[c] != 0 && px[c] != 255) {
                    valid = false;
                    break;
                }
            }
        }
    }
    TEST_CHECK(valid, "11.1 交替 0/1 → 交替 0/255");
}

// ---- Test 12: 不同尺寸输入 ----
static void testNonDefaultSizes() {
    TEST_NAME("Test 12: 不同尺寸输入");

    OutputProcessor proc;

    // 12.1 非标准输出尺寸
    ncnn::Mat custom_out = createModelOutput(200, 150);
    cv::Mat custom_face = proc.OutputToMat(custom_out, 80, 80);
    TEST_CHECK(!custom_face.empty(), "12.1 200×150 输出 → 80×80 人脸");
    TEST_CHECK(custom_face.cols == 80, "12.1 宽度 == 80");
    TEST_CHECK(custom_face.rows == 80, "12.1 高度 == 80");

    // 12.2 大尺寸融合
    cv::Mat large_orig = createFaceImage(400, 300);
    cv::Mat large_gen = createFaceImage(400, 300);
    cv::Mat large_mask = createFaceMask(400, 300, 0.3f);
    cv::Mat large_fused = proc.FaceFusion(large_orig, large_gen, large_mask);
    TEST_CHECK(!large_fused.empty(), "12.2 大尺寸 (400×300) 融合成功");
    TEST_CHECK(large_fused.cols == 400, "12.2 融合宽度 == 400");
    TEST_CHECK(large_fused.rows == 300, "12.2 融合高度 == 300");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  OutputProcessor 模块验收测试"                  << std::endl;
    std::cout << "==============================================" << std::endl;

    testNcnnToCvMat();
    testFaceRegionExtract();
    testInverseTransform();
    testFaceFusion();
    testSharpen();
    testColorBlend();
    testFullPipeline();
    testPostProcess();
    testMaskEdgeCases();
    testMoveSemantics();
    testExtremeValues();
    testNonDefaultSizes();

    // ==========================================
    // 汇总
    // ==========================================
    std::cout << "\n==============================================" << std::endl;
    std::cout << "  测试汇总" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  通过: " << gPassed << std::endl;
    std::cout << "  失败: " << gFailed << std::endl;
    std::cout << "  总计: " << (gPassed + gFailed) << std::endl;
    std::cout << "==============================================" << std::endl;

    return (gFailed == 0) ? 0 : 1;
}
