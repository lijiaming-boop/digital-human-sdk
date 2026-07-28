/**
 * @file lip_sync_diagnose_test.cpp
 * @brief 诊断 Wav2Lip 推理输出是否随音频输入变化
 *
 * 用 3 组明显不同的音频 Mel 输入 + 相同人脸输入推理，
 * 比较原始 ncnn 输出的差异。若差异≈0 则模型/权重有问题。
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ncnn/mat.h>

#include "audio/audio_loader.h"
#include "audio/audio_noise_reduction.h"
#include "audio/audio_framer.h"
#include "audio/audio_preemphasis.h"
#include "audio/audio_rms_normalize.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_cmvn.h"
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "model/model_inferencer.h"

using namespace digital_human;

static ncnn::Mat faceToNCNN(const cv::Mat& bgr, int w, int h) {
    ncnn::Mat out(w, h, 6);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            auto px = bgr.at<cv::Vec3b>(y, x);
            out.channel(0).row(y)[x] = px[2] / 255.0f;
            out.channel(1).row(y)[x] = px[1] / 255.0f;
            out.channel(2).row(y)[x] = px[0] / 255.0f;
            out.channel(3).row(y)[x] = px[2] / 255.0f;
            out.channel(4).row(y)[x] = px[1] / 255.0f;
            out.channel(5).row(y)[x] = px[0] / 255.0f;
        }
    return out;
}

static ncnn::Mat melToNCNN(const cv::Mat& feat, int start, int n) {
    // 正确布局: ncnn::Mat(w=时间帧数, h=mel bins=80, c=1)
    // row(bin)[frame] = feat(frame, bin)
    ncnn::Mat out(n, feat.cols, 1);
    int last = feat.rows - 1;
    for (int bin = 0; bin < feat.cols; ++bin) {
        float* row = out.channel(0).row(bin);
        for (int f = 0; f < n; ++f) {
            int sf = std::min(start + f, last);
            row[f] = feat.at<float>(sf, bin);
        }
    }
    return out;
}

static double ncnnDiff(const ncnn::Mat& a, const ncnn::Mat& b) {
    if (a.empty() || b.empty()) return -1;
    if (a.w != b.w || a.h != b.h || a.c != b.c) return -2;
    double sum = 0;
    int n = 0;
    for (int c = 0; c < a.c; ++c) {
        const float* pa = a.channel(c);
        const float* pb = b.channel(c);
        for (int i = 0; i < a.w * a.h; ++i) {
            sum += std::fabs(pa[i] - pb[i]);
            ++n;
        }
    }
    return n ? sum / n : -3;
}

int main(int argc, char** argv) {
    std::string assets_dir = ASSETS_DIR;
    std::string model_dir  = std::string(PROJECT_SOURCE_DIR) + "/models";

    // ---- 人脸 ----
    cv::Mat original = cv::imread(assets_dir + "/face.jpg");
    if (original.empty()) { std::cerr << "no face.jpg\n"; return 1; }
    core::FaceDetector detector;
    detector.loadModel(model_dir + "/shape_predictor_68_face_landmarks.dat");
    auto faces = detector.detect(original);
    if (faces.empty()) { std::cerr << "no face detected\n"; return 1; }
    auto pts = detector.getLandmarks(original, faces[0]);
    core::FaceAligner aligner;
    std::vector<cv::Point2f> lmf;
    for (auto& p : pts) lmf.emplace_back((float)p.x, (float)p.y);
    auto ar = aligner.alignByRect(original, lmf, 96, faces[0]);
    ncnn::Mat face_ncnn = faceToNCNN(ar.aligned_face, 96, 96);

    // ---- 音频 Mel ----
    audio::AudioLoader loader;
    auto audio = loader.load(assets_dir + "/zw.mp3");
    std::cout << "audio: " << audio.duration << "s @ " << audio.sampleRate
              << "Hz, samples=" << audio.samples.size() << std::endl;

    audio::NoiseReduction nr(10, 0.02f);
    audio::RMSNormalize rn(0.056f);
    audio::PreEmphasis pe(0.97f);
    audio::AudioFramer fr;
    auto frames = fr.frame(pe.process(rn.process(nr.process(audio.samples, audio.sampleRate))),
                           {400, 160});
    audio::MelFeatureExtract me;
    audio::CMVN cmvn;
    auto feat = cmvn.process(me.extract(frames, {512, 80, audio.sampleRate, 0, 8000}));
    std::cout << "mel feat: " << feat.rows << "x" << feat.cols << std::endl;

    // 打印 Mel 特征自身变化量（确认音频输入确实在变化）
    {
        cv::Mat d;
        cv::absdiff(feat.row(100), feat.row(500), d);
        std::cout << "mel row100 vs row500: mean abs diff = "
                  << cv::mean(d)[0] << std::endl;
    }

    // ---- 推理 ----
    model::ModelInferencer inf;
    if (!inf.Init(model_dir)) { std::cerr << "model init fail\n"; return 1; }

    // 取 3 个相距很远的窗口：开头 / 中间 / 靠后
    // 正确的模型输入: 80 mel bins × 16 时间帧 (ncnn: w=16, h=80, c=1)
    int ctx = 16;
    int starts[3] = {0, feat.rows / 2 - ctx / 2, feat.rows - ctx - 1};
    std::vector<ncnn::Mat> outs;
    for (int i = 0; i < 3; ++i) {
        if (starts[i] < 0) starts[i] = 0;
        ncnn::Mat o = inf.Infer(melToNCNN(feat, starts[i], ctx), face_ncnn);
        if (o.empty()) { std::cerr << "infer " << i << " fail\n"; return 1; }
        std::cout << "infer[" << i << "] start=" << starts[i]
                  << " shape=" << o.w << "x" << o.h << "x" << o.c << std::endl;

        // ---- 打印原始 float 值域，判断是否 tanh [-1,1] 输出 ----
        {
            float mn = 1e9f, mx = -1e9f; double sum = 0; long n = 0;
            for (int c = 0; c < o.c; ++c) {
                const float* p = o.channel(c);
                for (int k = 0; k < o.w * o.h; ++k) {
                    mn = std::min(mn, p[k]); mx = std::max(mx, p[k]);
                    sum += p[k]; ++n;
                }
            }
            std::cout << "  raw float range: min=" << mn << " max=" << mx
                      << " mean=" << (sum / n) << std::endl;
        }
        outs.push_back(o);
    }

    double d01 = ncnnDiff(outs[0], outs[1]);
    double d02 = ncnnDiff(outs[0], outs[2]);
    double d12 = ncnnDiff(outs[1], outs[2]);
    std::cout << "\n===== 原始模型输出差异（[0,1] 域） =====\n";
    std::cout << "out0 vs out1: " << d01 << std::endl;
    std::cout << "out0 vs out2: " << d02 << std::endl;
    std::cout << "out1 vs out2: " << d12 << std::endl;

    // 保存中间帧人脸区域图片（center 96x96 crop），人眼可直接对比
    for (int i = 0; i < 3; ++i) {
        ncnn::Mat& o = outs[i];
        cv::Mat img(o.h, o.w, CV_8UC3);
        for (int y = 0; y < o.h; ++y) {
            auto* row = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < o.w; ++x) {
                int idx = y * o.w + x;
                row[x] = cv::Vec3b(
                    (uchar)std::clamp(o.channel(2)[idx] * 255.f, 0.f, 255.f),
                    (uchar)std::clamp(o.channel(1)[idx] * 255.f, 0.f, 255.f),
                    (uchar)std::clamp(o.channel(0)[idx] * 255.f, 0.f, 255.f));
            }
        }
        std::string p = assets_dir + "/output/diag_raw_out_" + std::to_string(i) + ".png";
        cv::imwrite(p, img);
        std::cout << "saved " << p << std::endl;
    }

    // 判定
    if (d01 > 0.001 || d02 > 0.001) {
        std::cout << "\n[结论] 模型输出随音频变化 — 推理有效，问题在后处理/融合环节\n";
        return 0;
    }
    std::cout << "\n[结论] 模型输出几乎恒定 — 问题在模型加载/权重/输入格式\n";
    return 2;
}
