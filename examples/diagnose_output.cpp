/**
 * @file diagnose_output.cpp
 * @brief 诊断输出 — 检查唇形是否有变化
 *
 * 生成 N 帧并逐帧对比像素差异，输出帧间变化热力图。
 * 如果模型正常，连续帧之间口唇区域应有明显像素变化。
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

#include "model/model_inferencer.h"
#include "model/output_processor.h"

using namespace digital_human;

static ncnn::Mat faceToNCNN(const cv::Mat& bgr, int w, int h) {
    ncnn::Mat out(w, h, 6);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            auto px = bgr.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                out.channel(c).row(y)[x] = px[2-c] / 255.0f;
                out.channel(c+3).row(y)[x] = px[2-c] / 255.0f;
            }
        }
    return out;
}

static ncnn::Mat melToNCNN(const cv::Mat& feat, int start, int n) {
    ncnn::Mat out(feat.cols, n, 1);
    int last = feat.rows - 1;
    for (int y = 0; y < n; ++y)
        std::memcpy(out.channel(0).row(y), feat.ptr<float>(std::min(start+y, last)),
                    feat.cols * sizeof(float));
    return out;
}

int main() {
    std::string assets_dir = ASSETS_DIR;
    std::string model_dir  = std::string(PROJECT_SOURCE_DIR) + "/models";
    std::string out_dir    = assets_dir + "/diagnose";
    std::filesystem::create_directories(out_dir);

    // 1. 人脸
    cv::Mat orig = cv::imread(assets_dir + "/face.jpg");
    digital_human::core::FaceDetector det;
    det.loadModel(model_dir + "/face");
    auto faces = det.detect(orig);
    auto pts = det.getLandmarks(orig, faces[0]);
    digital_human::core::FaceAligner aligner;
    std::vector<cv::Point2f> lmf;
    for (auto& p : pts) lmf.emplace_back((float)p.x, (float)p.y);
    auto ar = aligner.alignByRect(orig, lmf, 96, faces[0]);
    ncnn::Mat face_ncnn = faceToNCNN(ar.aligned_face, 96, 96);

    digital_human::core::FaceMaskGenerator mask_gen;
    cv::Mat mouth_mask = mask_gen.generateMouthMask(orig.size(), pts);

    // 2. 音频 Mel
    audio::AudioLoader loader;
    auto audio = loader.load(assets_dir + "/zw.mp3");
    audio::NoiseReduction nr(10, 0.02f);
    audio::RMSNormalize rn(0.056f);
    audio::PreEmphasis pe(0.97f);
    audio::AudioFramer fr;
    auto frames = fr.frame(pe.process(rn.process(nr.process(audio.samples, audio.sampleRate))), {400, 160});
    audio::MelFeatureExtract me;
    audio::CMVN cmvn;
    auto feat = cmvn.process(me.extract(frames, {512, 80, audio.sampleRate, 0, 8000}));

    // 3. 模型
    model::ModelInferencer inf;
    inf.Init(model_dir);
    model::OutputProcessor op;

    // 4. 生成 10 帧（不同音频段）
    std::cout << "生成 10 帧进行对比..." << std::endl;
    int ctx = 80, half = ctx / 2;
    std::vector<cv::Mat> frames_out(10);

    for (int i = 0; i < 10; ++i) {
        int center = i * 30 + half;  // 每帧间隔 30 Mel 帧
        int start = std::max(0, std::min(center - half, feat.rows - ctx));
        auto output = inf.Infer(melToNCNN(feat, start, ctx), face_ncnn);

        // 保存模型原始输出 (448x96)
        cv::Mat raw_face = op.OutputToMat(output, 96, 96);
        cv::imwrite(out_dir + "/raw_" + std::to_string(i) + ".jpg", raw_face);

        // 完整融合
        frames_out[i] = op.Process(output, orig, mouth_mask, ar.M_inv);
        cv::imwrite(out_dir + "/frame_" + std::to_string(i) + ".jpg", frames_out[i]);

        std::cout << "  帧 " << i << ": center=" << center << " mel_frame" << std::endl;
    }

    // 5. 逐帧对比（检测口唇区域变化）
    std::cout << "\n帧间差异分析 (口唇区域):" << std::endl;
    auto lip_roi = faces[0];  // 人脸区域即口唇区域

    for (int i = 1; i < 10; ++i) {
        cv::Mat diff;
        cv::absdiff(frames_out[i](lip_roi), frames_out[i-1](lip_roi), diff);
        cv::Mat gray;
        cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
        double mean_diff = cv::mean(gray)[0];

        std::cout << "  帧" << (i-1) << "→帧" << i
                  << ": 口唇区域平均像素变化 = " << std::fixed << std::setprecision(2) << mean_diff
                  << (mean_diff > 1.0 ? " [有变化]" : " [几乎无变化]")
                  << std::endl;
    }

    // 6. 与原图对比
    std::cout << "\n与原图对比 (口唇区域):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Mat diff;
        cv::absdiff(frames_out[i](lip_roi), orig(lip_roi), diff);
        cv::Mat gray;
        cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
        double mean_diff = cv::mean(gray)[0];
        std::cout << "  原图→帧" << i << ": 口唇区域平均像素变化 = " << mean_diff
                  << (mean_diff > 2.0 ? " [明显变化]" : " [几乎无变化]")
                  << std::endl;
    }

    // 7. 保存对比图
    cv::Mat grid(96 * 2, 96 * 5, CV_8UC3);
    for (int i = 0; i < 10; ++i) {
        cv::Mat raw = cv::imread(out_dir + "/raw_" + std::to_string(i) + ".jpg");
        if (!raw.empty()) {
            int x = (i % 5) * 96;
            int y = (i / 5) * 96;
            raw.copyTo(grid(cv::Rect(x, y, 96, 96)));
        }
    }
    cv::imwrite(out_dir + "/raw_grid.jpg", grid);
    std::cout << "\n对比图已保存: " << out_dir << "/raw_grid.jpg" << std::endl;
    std::cout << "逐帧图已保存: " << out_dir << "/frame_0.jpg ~ frame_9.jpg" << std::endl;

    return 0;
}
