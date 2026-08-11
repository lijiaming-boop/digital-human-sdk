/**
 * @file verify_postprocess_fix.cpp
 * @brief 验证 ProcessROI 修复效果 — 量化 ROI 内 mask 外像素是否严格等于原图
 *
 * 不依赖 dlib/ncnn 模型,使用合成数据直接驱动后处理流水线:
 *   1. 构造 1920×1384 原图 (模拟 face.jpg 尺寸)
 *   2. 构造合成 model_output (颜色明显不同, 模拟 Wav2Lip 输出)
 *   3. 构造已知 M_inv (平移+缩放) 和 mask (矩形区域)
 *   4. 调用 ProcessROI
 *   5. 量化:
 *      - ROI 内 mask 外像素 vs 原图 (应严格相等, 修复后不变量)
 *      - ROI 外像素 vs 原图 (应严格相等)
 *      - mask 内像素 vs 原图 (应有差异, 证明拟合生效)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <mat.h>

#include "model/output_processor.h"

using namespace digital_human;

int main() {
    std::cout << "==============================================\n";
    std::cout << "  ProcessROI 修复效果验证 (合成数据)\n";
    std::cout << "==============================================\n";

    // ---- 1. 原图: 1920×1384, 用渐变填充 (便于检测任何像素改动) ----
    const int W = 1920, H = 1384;
    cv::Mat original(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // BGR 渐变, 每个像素唯一可识别
            original.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((x * 13 + y * 7) % 256),
                static_cast<uchar>((x * 5 + y * 11) % 256),
                static_cast<uchar>((x * 17 + y * 3) % 256));
        }
    }

    // ---- 2. 合成 model_output: 96×96×3, 纯红 (与原图任意像素都不同) ----
    // 模型输出布局: w=96, h=96, c=3, RGB float [0,1]
    const int FW = 96, FH = 96;
    ncnn::Mat model_out(FW, FH, 3);
    for (int y = 0; y < FH; ++y) {
        for (int x = 0; x < FW; ++x) {
            // RGB = (1, 0, 0) 纯红, 转 BGR uint8 = (0, 0, 255)
            model_out.channel(0).row(y)[x] = 1.0f;  // R
            model_out.channel(1).row(y)[x] = 0.0f;  // G
            model_out.channel(2).row(y)[x] = 0.0f;  // B
        }
    }

    // ---- 3. 构造 M_inv: 把 96×96 对齐脸映射到原图中央 ----
    // 正向 M: 原图中央 (cx, cy) 缩放到 96×96
    // 逆 M (M_inv): 96×96 → 原图中央, 缩放因子 s
    const double cx = 800.0, cy = 600.0;  // 原图中人脸中心
    const double s = 6.0;                 // 96*6 = 576px 人脸宽度
    // M_inv: [s 0 cx-s*48; 0 s cy-s*48]
    cv::Mat M_inv = (cv::Mat_<float>(2, 3) <<
        s, 0, cx - s * FW / 2.0,
        0, s, cy - s * FH / 2.0);

    // ---- 4. face_rect (提供 ROI 参考) ----
    cv::Rect face_rect(
        static_cast<int>(cx - s * FW / 2),
        static_cast<int>(cy - s * FH / 2),
        static_cast<int>(s * FW),
        static_cast<int>(s * FH));

    // ---- 5. mask: 在 96×96 对齐空间构造一个矩形 mouth mask ----
    // 嘴唇位于 96×96 的中下部, 模拟 generatePreciseMouthAlphaMask96 的输出
    cv::Mat mask_96 = cv::Mat::zeros(FH, FW, CV_32FC1);
    // 中下部矩形 + 高斯模糊 (模拟真实 mask 的渐变)
    cv::Rect mouth_rect(30, 60, 36, 20);
    cv::rectangle(mask_96, mouth_rect, cv::Scalar(1.0f), cv::FILLED);
    cv::GaussianBlur(mask_96, mask_96, cv::Size(13, 13), 0);

    // ---- 6. 调用 ProcessROI ----
    model::OutputProcessor op;
    cv::Mat result = op.ProcessROI(model_out, original, mask_96, M_inv, face_rect, 0.25f);

    if (result.empty()) {
        std::cerr << "FAIL: ProcessROI 返回空\n";
        return 1;
    }

    std::cout << "[OK] ProcessROI 完成, 输出尺寸: "
              << result.cols << "x" << result.rows << "\n";

    // ---- 7. 量化分析 ----
    // 7.1 计算 diff
    cv::Mat diff;
    cv::absdiff(result, original, diff);
    diff.convertTo(diff, CV_32F);
    cv::Mat gray_diff;
    cv::cvtColor(diff, gray_diff, cv::COLOR_BGR2GRAY);

    // 7.2 把 mask 投影到原图坐标 (与 ProcessROI 内部一致)
    cv::Mat mask_full;
    cv::warpAffine(mask_96, mask_full, M_inv, original.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0f));

    // 7.3 ROI 边界 (用 ProcessROI 的逻辑重新算)
    cv::Mat M64;
    M_inv.convertTo(M64, CV_64F);
    auto project = [&](double x, double y) {
        return cv::Point2d(
            M64.at<double>(0, 0) * x + M64.at<double>(0, 1) * y + M64.at<double>(0, 2),
            M64.at<double>(1, 0) * x + M64.at<double>(1, 1) * y + M64.at<double>(1, 2));
    };
    std::vector<cv::Point2d> corners = {
        project(0, 0), project(FW, 0), project(FW, FH), project(0, FH)
    };
    double x0 = corners[0].x, y0 = corners[0].y, x1 = corners[0].x, y1 = corners[0].y;
    for (const auto& p : corners) {
        x0 = std::min(x0, p.x); y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
    }
    double margin = 0.25;
    int mx = static_cast<int>((x1 - x0) * margin);
    int my = static_cast<int>((y1 - y0) * margin);
    cv::Rect roi(static_cast<int>(std::floor(x0)) - mx,
                 static_cast<int>(std::floor(y0)) - my,
                 static_cast<int>(std::ceil(x1 - x0)) + 2 * mx,
                 static_cast<int>(std::ceil(y1 - y0)) + 2 * my);
    roi |= face_rect;
    roi &= cv::Rect(0, 0, W, H);

    std::cout << "\n--- ROI 信息 ---\n";
    std::cout << "ROI: (" << roi.x << "," << roi.y << ") "
              << roi.width << "x" << roi.height << "\n";
    std::cout << "ROI 面积: " << roi.area() << " / 全图: " << W * H
              << " (" << std::fixed << std::setprecision(1)
              << 100.0 * roi.area() / (W * H) << "%)\n";

    // 7.4 分类统计
    long total = 0, in_roi = 0, out_roi = 0;
    long in_mask = 0, out_mask_in_roi = 0;
    double max_diff_out_roi = 0, max_diff_out_mask_in_roi = 0;
    double sum_diff_out_roi = 0, sum_diff_out_mask_in_roi = 0, sum_diff_in_mask = 0;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float d = gray_diff.at<float>(y, x);
            bool in_roi_flag = roi.contains(cv::Point(x, y));
            float m = mask_full.at<float>(y, x);
            total++;
            if (in_roi_flag) {
                in_roi++;
                if (m > 0.01f) {
                    in_mask++;
                    sum_diff_in_mask += d;
                } else {
                    out_mask_in_roi++;
                    sum_diff_out_mask_in_roi += d;
                    if (d > max_diff_out_mask_in_roi) max_diff_out_mask_in_roi = d;
                }
            } else {
                out_roi++;
                sum_diff_out_roi += d;
                if (d > max_diff_out_roi) max_diff_out_roi = d;
            }
        }
    }

    std::cout << "\n--- 修复效果量化 ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[A] ROI 外像素 (应严格 == 原图):\n";
    std::cout << "    像素数: " << out_roi << "\n";
    std::cout << "    平均 diff: " << sum_diff_out_roi / out_roi << "\n";
    std::cout << "    最大 diff: " << max_diff_out_roi << "\n";
    std::cout << (max_diff_out_roi < 0.5 ? "    [PASS] ROI 外无污染\n"
                                          : "    [FAIL] ROI 外有污染\n");

    std::cout << "[B] ROI 内 mask 外像素 (修复后应 == 原图):\n";
    std::cout << "    像素数: " << out_mask_in_roi << "\n";
    std::cout << "    平均 diff: " << sum_diff_out_mask_in_roi / out_mask_in_roi << "\n";
    std::cout << "    最大 diff: " << max_diff_out_mask_in_roi << "\n";
    std::cout << (max_diff_out_mask_in_roi < 0.5 ? "    [PASS] ROI 内 mask 外无污染 (贴回方块消除)\n"
                                                  : "    [FAIL] ROI 内 mask 外有污染 (贴回方块仍存在)\n");

    std::cout << "[C] mask 内像素 (应有差异, 证明拟合生效):\n";
    std::cout << "    像素数: " << in_mask << "\n";
    std::cout << "    平均 diff: " << sum_diff_in_mask / in_mask << "\n";
    std::cout << (sum_diff_in_mask / in_mask > 5.0 ? "    [PASS] mask 内有显著变化 (拟合生效)\n"
                                                    : "    [WARN] mask 内变化较小\n");

    // 7.5 保存可视化
    std::string out_dir = std::string(ASSETS_DIR) + "/verify";
    std::filesystem::create_directories(out_dir);
    cv::imwrite(out_dir + "/original.png", original);
    cv::imwrite(out_dir + "/result_fixed.png", result);

    // mask 可视化 (放大到原图)
    cv::Mat mask_vis;
    mask_full.convertTo(mask_vis, CV_8UC1, 255.0);
    cv::applyColorMap(mask_vis, mask_vis, cv::COLORMAP_JET);
    cv::imwrite(out_dir + "/mask_full.png", mask_vis);

    // diff 可视化
    cv::Mat diff_vis;
    gray_diff.convertTo(diff_vis, CV_8UC1);
    cv::applyColorMap(diff_vis, diff_vis, cv::COLORMAP_HOT);
    cv::imwrite(out_dir + "/diff.png", diff_vis);

    // ROI 边界叠加
    cv::Mat roi_vis = result.clone();
    cv::rectangle(roi_vis, roi, cv::Scalar(0, 255, 0), 3);
    cv::imwrite(out_dir + "/result_with_roi.png", roi_vis);

    std::cout << "\n--- 可视化已保存 ---\n";
    std::cout << "  " << out_dir << "/original.png\n";
    std::cout << "  " << out_dir << "/result_fixed.png\n";
    std::cout << "  " << out_dir << "/mask_full.png\n";
    std::cout << "  " << out_dir << "/diff.png\n";
    std::cout << "  " << out_dir << "/result_with_roi.png\n";

    return 0;
}
