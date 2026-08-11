/**
 * @file dump_fusion_mask.cpp
 * @brief 导出 prepareFusionMask 的实际输出,验证 mask 渐变带是否加宽
 */
#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"

using namespace digital_human;

int main() {
    std::string assets = ASSETS_DIR;
    std::string model_dir = std::string(PROJECT_SOURCE_DIR) + "/models";
    cv::Mat img = cv::imread(assets + "/face.jpg");
    if (img.empty()) { std::cerr << "face.jpg load fail\n"; return 1; }

    core::FaceDetector detector;
    detector.loadModel(model_dir + "/face");
    auto faces = detector.detect(img);
    if (faces.empty()) { std::cerr << "no face\n"; return 1; }
    auto pts = detector.getLandmarks(img, faces[0]);
    std::vector<cv::Point2f> lms;
    for (auto& p : pts) lms.emplace_back((float)p.x, (float)p.y);

    core::FaceAligner aligner;
    auto result = aligner.alignByRect(img, lms, 96, faces[0], 0.15);
    if (!result.valid) { std::cerr << "align fail\n"; return 1; }

    cv::Mat M = result.M;
    cv::Mat M_inv = result.M_inv;
    auto lms96 = aligner.transform_landmarks(lms, M);

    core::FaceMaskGenerator mask_gen;
    auto mask96 = mask_gen.generatePreciseMouthAlphaMask96(lms96);
    std::cout << "mask96 size: " << mask96.cols << "x" << mask96.rows << "\n";

    // ===== 路径 A: generateMouthMask (video_output_test 实际使用的路径) =====
    auto mask_a = mask_gen.generateMouthMask(img.size(), pts);

    // ===== 路径 B: 96 空间 mask 经 doInverseTransformMask =====
    cv::Mat mask_b(img.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::warpAffine(mask96, mask_b, M_inv, img.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0f));
    cv::threshold(mask_b, mask_b, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(mask_b, mask_b, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::GaussianBlur(mask_b, mask_b, cv::Size(21, 21), 0);
    cv::threshold(mask_b, mask_b, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(mask_b, mask_b, 0.0, 0.0, cv::THRESH_TOZERO);

    std::string out_dir = assets + "/mask_debug";
    std::filesystem::create_directories(out_dir);

    auto save_vis = [&](const cv::Mat& m, const std::string& name) {
        cv::Mat vis;
        m.convertTo(vis, CV_8UC1, 255.0);
        cv::applyColorMap(vis, vis, cv::COLORMAP_JET);
        cv::imwrite(out_dir + "/" + name, vis);
    };
    save_vis(mask_a, "mask_generateMouthMask.png");
    save_vis(mask_b, "mask_96path.png");

    auto report = [&](const cv::Mat& m, const std::string& label) {
        double minv, maxv;
        cv::Point minp, maxp;
        cv::minMaxLoc(m, &minv, &maxv, &minp, &maxp);
        std::cout << "\n[" << label << "] center=(" << maxp.x << "," << maxp.y << ")\n";
        int dirs[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,-1},{1,-1},{-1,1}};
        std::cout << "渐变带宽度 (mask 从 >0.9 降到 <0.05):\n";
        for (auto& d : dirs) {
            int x = maxp.x, y = maxp.y;
            int core_end = -1, fade_end = -1;
            for (int step = 0; step < 300 && x >= 0 && x < img.cols && y >= 0 && y < img.rows; ++step) {
                float v = m.at<float>(y, x);
                if (core_end < 0 && v < 0.9f) core_end = step;
                if (core_end >= 0 && v < 0.05f) { fade_end = step; break; }
                x += d[0]; y += d[1];
            }
            int width = (core_end >= 0 && fade_end >= 0) ? fade_end - core_end : -1;
            std::cout << "  (" << d[0] << "," << d[1] << "): core=" << core_end
                      << " fade=" << fade_end << " 宽度=" << width << "px\n";
        }
    };
    report(mask_a, "generateMouthMask (视频实际路径, blur_sigma=35)");
    report(mask_b, "96空间路径 (含本次修复 blur 21)");

    std::cout << "\n保存: " << out_dir << "/mask_generateMouthMask.png\n";
    std::cout << "保存: " << out_dir << "/mask_96path.png\n";
    return 0;
}
