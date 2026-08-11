/**
 * @file visual_output_test.cpp
 * @brief 可视化输出测试 — 将推理结果保存为图片
 *
 * 流程:
 *   1. 加载 face.jpg → 人脸检测 → 对齐 → 遮罩
 *   2. 加载 zw.mp3 → Mel 特征提取
 *   3. 模型推理 → 保存输出图片到 assets/output/
 *   4. 将生成的人脸融合回原图 → 保存融合效果
 *
 * 用法: ./bin/visual_output_test
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <mat.h>

#include "audio/audio_loader.h"
#include "audio/audio_noise_reduction.h"
#include "audio/audio_framer.h"
#include "audio/audio_vad.h"
#include "audio/audio_preemphasis.h"
#include "audio/audio_rms_normalize.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_cmvn.h"

#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"
#include "core/image_loader.h"
#include "core/packet.h"

#include "model/model_inferencer.h"
#include "model/output_processor.h"

using namespace digital_human;

// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "  可视化输出测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    std::string assets_dir = ASSETS_DIR;
    if (argc > 1) assets_dir = argv[1];

    std::string mp3_path   = assets_dir + "/zw.mp3";
    std::string face_path  = assets_dir + "/face.jpg";
    std::string model_dir  = std::string(PROJECT_SOURCE_DIR) + "/models";
    std::string out_dir    = assets_dir + "/output";

    // 创建输出目录
    std::filesystem::create_directories(out_dir);
    std::cout << "[INFO] 输出目录: " << out_dir << std::endl;

    // ====================================================================
    // 1. 加载人脸图像 → 检测 → 对齐 → 遮罩
    // ====================================================================
    std::cout << "\n========== [1] 人脸处理 ==========" << std::endl;
    cv::Mat original_img = cv::imread(face_path);
    if (original_img.empty()) {
        std::cerr << "[FAIL] 无法加载人脸图像: " << face_path << std::endl;
        return 1;
    }
    std::cout << "[OK] 人脸加载: " << original_img.cols << "x" << original_img.rows << std::endl;

    // 人脸检测
    digital_human::core::FaceDetector detector;
    std::string detector_model = model_dir + "/face";
    if (std::filesystem::exists(detector_model)) {
        detector.loadModel(detector_model);
    }
    auto faces = detector.detect(original_img);
    if (faces.empty()) {
        std::cerr << "[FAIL] 未检测到人脸" << std::endl;
        return 1;
    }
    auto face_rect = faces[0];
    std::cout << "[OK] 人脸检测: " << face_rect.x << "," << face_rect.y
              << " " << face_rect.width << "x" << face_rect.height << std::endl;

    // 关键点 + 对齐
    auto landmarks = detector.getLandmarks(original_img, face_rect);
    digital_human::core::FaceAligner aligner;
    std::vector<cv::Point2f> landmarks_f;
    for (const auto& pt : landmarks) {
        landmarks_f.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }
    auto align_result = aligner.alignByRect(original_img, landmarks_f, 96, face_rect);
    if (!align_result.valid) {
        std::cerr << "[FAIL] 人脸对齐失败" << std::endl;
        return 1;
    }
    cv::imwrite(out_dir + "/01_aligned_face.jpg", align_result.aligned_face);
    std::cout << "[OK] 对齐人脸已保存: 01_aligned_face.jpg" << std::endl;

    // 口唇遮罩
    digital_human::core::FaceMaskGenerator mask_gen;
    cv::Mat mouth_mask = mask_gen.generateMouthMask(original_img.size(), landmarks);
    cv::Mat mouth_mask_3ch;
    cv::cvtColor(mouth_mask * 255, mouth_mask_3ch, cv::COLOR_GRAY2BGR);
    cv::imwrite(out_dir + "/02_mouth_mask.jpg", mouth_mask_3ch);
    std::cout << "[OK] 口唇遮罩已保存: 02_mouth_mask.jpg" << std::endl;

    // 96x96 精细遮罩
    auto precise_mask = mask_gen.generatePreciseMouthAlphaMask96(align_result.landmarks);
    cv::imwrite(out_dir + "/03_precise_mask_96.jpg", precise_mask * 255);
    std::cout << "[OK] 精细遮罩已保存: 03_precise_mask_96.jpg (96x96)" << std::endl;

    // 保存原图标注
    cv::Mat annotated = original_img.clone();
    cv::rectangle(annotated, face_rect, cv::Scalar(0, 255, 0), 3);
    cv::putText(annotated, "Face", cv::Point(face_rect.x, face_rect.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::imwrite(out_dir + "/00_original_with_face.jpg", annotated);
    std::cout << "[OK] 标注图像已保存: 00_original_with_face.jpg" << std::endl;

    // ====================================================================
    // 2. 加载音频 → Mel 特征
    // ====================================================================
    std::cout << "\n========== [2] 音频特征提取 ==========" << std::endl;
    audio::AudioLoader loader;
    audio::AudioData audio;
    try {
        audio = loader.load(mp3_path);
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] 音频加载失败: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "[OK] 音频加载: " << audio.duration << "s, "
              << audio.sampleRate << "Hz, " << audio.channels << "ch" << std::endl;

    // 提取第一段 Mel 特征（取前 1.6s = 80 帧，匹配模型输入）
    const int sr = audio.sampleRate;
    const int target_mel_frames = 80;  // 模型期望 80 帧

    audio::NoiseReduction nr(10, 0.02f);
    auto denoised = nr.process(audio.samples, sr);

    audio::RMSNormalize rms_norm(0.056f);
    auto normalized = rms_norm.process(denoised);

    audio::PreEmphasis pe(0.97f);
    auto emphasized = pe.process(normalized);

    audio::AudioFramer framer;
    audio::FrameConfig fcfg{400, 160};
    auto frames = framer.frame(emphasized, fcfg);

    audio::MelFeatureExtract mel_ext;
    audio::MelConfig mcfg{512, 80, sr, 0.0f, 8000.0f};
    auto mel_full = mel_ext.extract(frames, mcfg);

    audio::CMVN cmvn;
    auto feat_full = cmvn.process(mel_full);

    // 取前 target_mel_frames 帧
    cv::Mat feat_input;
    if (feat_full.rows >= target_mel_frames) {
        feat_input = feat_full(cv::Rect(0, 0, feat_full.cols, target_mel_frames)).clone();
    } else {
        // 不足则补零
        feat_input = cv::Mat::zeros(target_mel_frames, feat_full.cols, CV_32F);
        feat_full.copyTo(feat_input(cv::Rect(0, 0, feat_full.cols, feat_full.rows)));
    }
    std::cout << "[OK] Mel 特征: " << feat_input.rows << "x" << feat_input.cols
              << " (目标 " << target_mel_frames << " 帧)" << std::endl;

    // 保存 Mel 特征可视化
    cv::Mat mel_viz;
    cv::normalize(feat_input, mel_viz, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(mel_viz, mel_viz, cv::COLORMAP_JET);
    cv::imwrite(out_dir + "/04_mel_spectrogram.jpg", mel_viz);
    std::cout << "[OK] Mel 频谱已保存: 04_mel_spectrogram.jpg" << std::endl;

    // ====================================================================
    // 3. 模型推理
    // ====================================================================
    std::cout << "\n========== [3] 模型推理 ==========" << std::endl;
    model::ModelInferencer inferencer;
    if (!inferencer.Init(model_dir)) {
        std::cerr << "[FAIL] 模型初始化失败" << std::endl;
        return 1;
    }
    std::cout << "[OK] 模型初始化: 线程=" << inferencer.GetThreadCount()
              << " GPU=" << (inferencer.IsGPUEnabled() ? "yes" : "no")
              << " 目标延迟=" << inferencer.GetTargetLatencyMs() << "ms" << std::endl;

    // 准备人脸输入 (6 通道)
    cv::Mat aligned_face = align_result.aligned_face;
    int w = aligned_face.cols, h = aligned_face.rows;

    // cv::Mat(BGR) → ncnn::Mat(96x96x6, RGB float)
    ncnn::Mat face_ncnn(w, h, 6);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3b bgr = aligned_face.at<cv::Vec3b>(y, x);
            face_ncnn.channel(0).row(y)[x] = bgr[2] / 255.0f;  // R
            face_ncnn.channel(1).row(y)[x] = bgr[1] / 255.0f;  // G
            face_ncnn.channel(2).row(y)[x] = bgr[0] / 255.0f;  // B
            face_ncnn.channel(3).row(y)[x] = bgr[2] / 255.0f;  // R (copy)
            face_ncnn.channel(4).row(y)[x] = bgr[1] / 255.0f;  // G (copy)
            face_ncnn.channel(5).row(y)[x] = bgr[0] / 255.0f;  // B (copy)
        }
    }

    // 准备 Mel 输入 (80x80x1)
    ncnn::Mat audio_ncnn(feat_input.cols, feat_input.rows, 1);
    for (int y = 0; y < feat_input.rows; ++y) {
        const float* row = feat_input.ptr<float>(y);
        float* dst = audio_ncnn.channel(0).row(y);
        std::memcpy(dst, row, feat_input.cols * sizeof(float));
    }

    // 推理
    auto t0 = std::chrono::steady_clock::now();
    ncnn::Mat output = inferencer.Infer(audio_ncnn, face_ncnn);
    double infer_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    if (output.empty()) {
        std::cerr << "[FAIL] 推理失败" << std::endl;
        return 1;
    }
    std::cout << "[OK] 推理成功: " << output.w << "x" << output.h << " c=" << output.c
              << " 耗时=" << infer_ms << "ms" << std::endl;

    // ====================================================================
    // 4. 输出处理器 → 融合到原图
    // ====================================================================
    std::cout << "\n========== [4] 输出后处理 ==========" << std::endl;

    model::OutputProcessor output_proc;

    // 4a. ncnn::Mat → cv::Mat (BGR)
    cv::Mat face_bgr = output_proc.OutputToMat(output, 96, 96);
    cv::imwrite(out_dir + "/05_generated_face.jpg", face_bgr);
    std::cout << "[OK] 生成人脸已保存: 05_generated_face.jpg" << std::endl;

    // 4b. 逆变换回原图坐标空间
    cv::Mat warped = output_proc.InverseTransform(face_bgr, align_result.M_inv,
                                                    original_img.size());
    cv::imwrite(out_dir + "/06_warped_face.jpg", warped);
    std::cout << "[OK] 逆变换人脸已保存: 06_warped_face.jpg" << std::endl;

    // 4c. 口唇融合
    cv::Mat fused = output_proc.FaceFusion(original_img, warped, mouth_mask);
    cv::imwrite(out_dir + "/07_fused_result.jpg", fused);
    std::cout << "[OK] 融合结果已保存: 07_fused_result.jpg" << std::endl;

    // 4d. 后处理（锐化 + 色彩融合）
    cv::Mat result = output_proc.PostProcess(fused, original_img);
    cv::imwrite(out_dir + "/08_final_output.jpg", result);
    std::cout << "[OK] 最终输出已保存: 08_final_output.jpg" << std::endl;

    // 4e. 全流程一键处理
    auto t1 = std::chrono::steady_clock::now();
    cv::Mat full = output_proc.Process(output, original_img, mouth_mask, align_result.M_inv);
    double process_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count();
    cv::imwrite(out_dir + "/09_full_pipeline.jpg", full);
    std::cout << "[OK] 全流程输出已保存: 09_full_pipeline.jpg"
              << " (后处理耗时=" << process_ms << "ms)" << std::endl;

    // ====================================================================
    // 汇总
    // ====================================================================
    std::cout << "\n==============================================" << std::endl;
    std::cout << "  输出文件清单" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  00_original_with_face.jpg  — 原图+人脸框" << std::endl;
    std::cout << "  01_aligned_face.jpg       — 对齐人脸 (96x96)" << std::endl;
    std::cout << "  02_mouth_mask.jpg          — 口唇遮罩" << std::endl;
    std::cout << "  03_precise_mask_96.jpg     — 精细遮罩 (96x96)" << std::endl;
    std::cout << "  04_mel_spectrogram.jpg     — Mel 频谱可视化" << std::endl;
    std::cout << "  05_generated_face.jpg      — 模型生成的唇形人脸" << std::endl;
    std::cout << "  06_warped_face.jpg          — 逆变换回原空间" << std::endl;
    std::cout << "  07_fused_result.jpg         — 口唇融合效果" << std::endl;
    std::cout << "  08_final_output.jpg         — 后处理效果" << std::endl;
    std::cout << "  09_full_pipeline.jpg        — 全流程一键输出" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  推理延迟: " << infer_ms << " ms" << std::endl;
    std::cout << "  后处理:  " << process_ms << " ms" << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
