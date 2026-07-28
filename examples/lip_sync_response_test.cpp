/**
 * @file lip_sync_response_test.cpp
 * @brief 严格诊断 Wav2Lip 模型是否对音频输入有响应
 *
 * 与 lip_sync_diagnose_test.cpp 的区别：
 *   旧版用同一段音频的不同时间窗口对比，但语音 Mel 在不同窗口本就相似，
 *   判定不严谨。本测试用【静音 vs 语音】两组截然不同的音频对照：
 *     A. assets/silence_30s_16k_mono.wav  (静音)
 *     B. assets/voice_30s_16k_mono.wav    (语音)
 *   同一张脸 + 两组音频 → 各取 N 个窗口推理 → 量化对比。
 *
 * 判定逻辑：
 *   - Mel 输入差异大 (基线) + 模型输出差异 < 0.002  → 模型无响应 (权重/输入格式错)
 *   - 模型输出差异 > 0.01                            → 模型有响应 (问题在融合/指标)
 *   - 中间值                                          → 弱响应
 */

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <system_error>

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
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "model/model_inferencer.h"

using namespace digital_human;

// ============================================================================
// 辅助函数 (复用自 lip_sync_diagnose_test.cpp)
// ============================================================================

/// @brief BGR cv::Mat → ncnn::Mat (w=96, h=96, c=6)，Wav2Lip 标准人脸输入
static ncnn::Mat faceToNCNN(const cv::Mat& bgr, int w, int h) {
    ncnn::Mat out(w, h, 6);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            auto px = bgr.at<cv::Vec3b>(y, x);
            out.channel(0).row(y)[x] = px[2] / 255.0f;  // R
            out.channel(1).row(y)[x] = px[1] / 255.0f;  // G
            out.channel(2).row(y)[x] = px[0] / 255.0f;  // B
            out.channel(3).row(y)[x] = px[2] / 255.0f;  // R (第二份参考帧)
            out.channel(4).row(y)[x] = px[1] / 255.0f;
            out.channel(5).row(y)[x] = px[0] / 255.0f;
        }
    return out;
}

/// @brief Mel 特征 (T×80) → ncnn::Mat (w=帧数, h=80, c=1)
static ncnn::Mat melToNCNN(const cv::Mat& feat, int start, int n) {
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

/// @brief 两个 ncnn::Mat 的平均绝对差
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

/// @brief 两个 ncnn::Mat 在指定 ROI 区域的平均绝对差
static double ncnnDiffROI(const ncnn::Mat& a, const ncnn::Mat& b,
                          int x0, int y0, int x1, int y1) {
    if (a.empty() || b.empty()) return -1;
    if (a.w != b.w || a.h != b.h || a.c != b.c) return -2;
    double sum = 0;
    int n = 0;
    for (int c = 0; c < a.c; ++c) {
        const float* pa = a.channel(c);
        const float* pb = b.channel(c);
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                int idx = y * a.w + x;
                sum += std::fabs(pa[idx] - pb[idx]);
                ++n;
            }
        }
    }
    return n ? sum / n : -3;
}

/// @brief 打印 ncnn::Mat 的值域统计
static void printStats(const ncnn::Mat& m, const std::string& tag) {
    float mn = 1e9f, mx = -1e9f;
    double sum = 0;
    long n = 0;
    for (int c = 0; c < m.c; ++c) {
        const float* p = m.channel(c);
        for (int k = 0; k < m.w * m.h; ++k) {
            mn = std::min(mn, p[k]);
            mx = std::max(mx, p[k]);
            sum += p[k];
            ++n;
        }
    }
    std::cout << "  " << tag << " range: min=" << mn << " max=" << mx
              << " mean=" << (sum / n) << std::endl;
}

/// @brief 保存 ncnn::Mat (3通道 RGB float) 为 PNG
static void saveMatPNG(const ncnn::Mat& o, const std::string& path) {
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
    cv::imwrite(path, img);
}

// ============================================================================
// 主流程
// ============================================================================

/// @brief 加载音频并提取 Mel 特征 (完整 7 阶段流水线)
static cv::Mat extractMel(const std::string& path) {
    audio::AudioLoader loader;
    auto audio = loader.load(path);
    std::cout << "  audio: " << audio.duration << "s @ " << audio.sampleRate
              << "Hz, samples=" << audio.samples.size() << std::endl;

    audio::NoiseReduction nr(10, 0.02f);
    audio::RMSNormalize rn(0.056f);
    audio::PreEmphasis pe(0.97f);
    audio::AudioFramer fr;
    auto frames = fr.frame(pe.process(rn.process(nr.process(audio.samples, audio.sampleRate))),
                           {400, 160});
    audio::MelFeatureExtract me;
    // Wav2Lip 官方 Mel 参数：nFFT=800, nMels=80, fmin=55, fmax=7600
    // apply_minmax=true 做 symmetric 归一化到 [-4,4]（模型期望的输入格式）
    // 注意：不再用 CMVN —— Wav2Lip symmetric 归一化已是最终输入
    audio::MelConfig mc;
    mc.nFFT = 800; mc.nMels = 80; mc.sampleRate = audio.sampleRate;
    mc.fMin = 55.0f; mc.fMax = 7600.0f; mc.winSize = 800;
    auto feat = me.extract(frames, mc, /*apply_minmax=*/true);
    std::cout << "  mel feat: " << feat.rows << "x" << feat.cols << std::endl;
    return feat;
}

int main(int argc, char** argv) {
    std::string assets_dir = (argc > 1) ? argv[1] : ASSETS_DIR;
    std::string model_dir  = std::string(PROJECT_SOURCE_DIR) + "/models";
    std::string silent_wav = (argc > 2) ? argv[2] : (assets_dir + "/silence_30s_16k_mono.wav");
    std::string voice_wav  = (argc > 3) ? argv[3] : (assets_dir + "/voice_30s_16k_mono.wav");
    int N = (argc > 4) ? std::atoi(argv[4]) : 8;  // 采样窗口数
    int ctx = 16;  // Wav2Lip 时间帧上下文

    std::cout << "==============================================\n";
    std::cout << "  Wav2Lip 模型响应诊断 (静音 vs 语音)\n";
    std::cout << "==============================================\n";
    std::cout << "  silent: " << silent_wav << "\n";
    std::cout << "  voice:  " << voice_wav  << "\n";
    std::cout << "  windows: " << N << " (ctx=" << ctx << ")\n\n";

    // ---- 1. 加载人脸并对齐到 96x96 ----
    cv::Mat original = cv::imread(assets_dir + "/face.jpg");
    if (original.empty()) { std::cerr << "[ERR] no face.jpg\n"; return 1; }
    core::FaceDetector detector;
    detector.loadModel(model_dir + "/shape_predictor_68_face_landmarks.dat");
    auto faces = detector.detect(original);
    if (faces.empty()) { std::cerr << "[ERR] no face detected\n"; return 1; }
    auto pts = detector.getLandmarks(original, faces[0]);
    core::FaceAligner aligner;
    std::vector<cv::Point2f> lmf;
    for (auto& p : pts) lmf.emplace_back((float)p.x, (float)p.y);
    auto ar = aligner.alignByRect(original, lmf, 96, faces[0]);
    ncnn::Mat face_ncnn = faceToNCNN(ar.aligned_face, 96, 96);
    std::cout << "[face] aligned 96x96, face rect=" << ar.face_rect << "\n\n";

    // ---- 2. 两组音频 Mel ----
    std::cout << "[silent mel]\n";
    cv::Mat mel_silent = extractMel(silent_wav);
    std::cout << "\n[voice mel]\n";
    cv::Mat mel_voice  = extractMel(voice_wav);

    // ---- 3. Mel 输入差异基线 ----
    // 取两组 Mel 在相同窗口位置的差异，证明输入确实不同
    int maxStart = std::min(mel_silent.rows, mel_voice.rows) - ctx;
    if (maxStart < 1) { std::cerr << "[ERR] mel too short\n"; return 1; }

    double mel_diff_sum = 0;
    for (int i = 0; i < N; ++i) {
        int start = (int)((long)i * maxStart / std::max(N - 1, 1));
        if (start < 0) start = 0;
        ncnn::Mat m_s = melToNCNN(mel_silent, start, ctx);
        ncnn::Mat m_v = melToNCNN(mel_voice,  start, ctx);
        mel_diff_sum += ncnnDiff(m_s, m_v);
    }
    double mel_diff_avg = mel_diff_sum / N;
    std::cout << "\n[基线] Mel 输入平均差异 (silent vs voice): " << mel_diff_avg << "\n";
    if (mel_diff_avg < 0.1) {
        std::cout << "[WARN] Mel 差异过小，两组音频可能没区别，诊断无效\n";
    }

    // ---- 4. 模型推理 ----
    model::ModelInferencer inf;
    if (!inf.Init(model_dir)) { std::cerr << "[ERR] model init fail\n"; return 1; }
    std::cout << "[model] init ok\n\n";

    std::vector<ncnn::Mat> out_silent, out_voice;
    for (int i = 0; i < N; ++i) {
        int start = (int)((long)i * maxStart / std::max(N - 1, 1));
        if (start < 0) start = 0;

        ncnn::Mat m_s = melToNCNN(mel_silent, start, ctx);
        ncnn::Mat m_v = melToNCNN(mel_voice,  start, ctx);

        ncnn::Mat o_s = inf.Infer(m_s, face_ncnn);
        ncnn::Mat o_v = inf.Infer(m_v, face_ncnn);
        if (o_s.empty() || o_v.empty()) {
            std::cerr << "[ERR] infer fail at window " << i << "\n";
            return 1;
        }
        out_silent.push_back(o_s);
        out_voice.push_back(o_v);
    }

    // 打印第一帧值域，判断输出是否正常 (tanh[-1,1] / sigmoid[0,1] / 全黑)
    std::cout << "[输出值域检查]\n";
    printStats(out_silent[0], "silent[0]");
    printStats(out_voice[0],  "voice[0]");
    std::cout << "\n";

    // ---- 5. 模型输出差异 (核心指标) ----
    // Wav2Lip 标准对齐 96x96 中嘴部 ROI: x=[24,72], y=[48,80]
    const int MX0 = 24, MY0 = 48, MX1 = 72, MY1 = 80;

    double out_diff_full = 0;     // 全图差异
    double out_diff_mouth = 0;    // 嘴部 ROI 差异
    for (int i = 0; i < N; ++i) {
        out_diff_full  += ncnnDiff(out_silent[i], out_voice[i]);
        out_diff_mouth += ncnnDiffROI(out_silent[i], out_voice[i], MX0, MY0, MX1, MY1);
    }
    double diff_full_avg  = out_diff_full  / N;
    double diff_mouth_avg = out_diff_mouth / N;

    // 同组内帧间差异 (判断输出是否随时间变化)
    double silent_intra = 0, voice_intra = 0;
    for (int i = 1; i < N; ++i) {
        silent_intra += ncnnDiff(out_silent[i - 1], out_silent[i]);
        voice_intra  += ncnnDiff(out_voice[i - 1],  out_voice[i]);
    }
    silent_intra /= std::max(N - 1, 1);
    voice_intra  /= std::max(N - 1, 1);

    std::cout << "==============================================\n";
    std::cout << "  诊断结果\n";
    std::cout << "==============================================\n";
    std::cout << "  [输入] Mel 差异 (silent vs voice): " << mel_diff_avg << "\n";
    std::cout << "  [输出] 全图差异 (silent vs voice): " << diff_full_avg << "\n";
    std::cout << "  [输出] 嘴部ROI差异 (silent vs voice): " << diff_mouth_avg << "\n";
    std::cout << "  [输出] 静音组帧间差异: " << silent_intra << "\n";
    std::cout << "  [输出] 语音组帧间差异: " << voice_intra << "\n";
    std::cout << "  [响应比] 输出差异/输入差异: "
              << (mel_diff_avg > 1e-6 ? diff_full_avg / mel_diff_avg : 0) << "\n\n";

    // ---- 6. 保存对比图 ----
    std::string out_dir = assets_dir + "/output/diag_response";
    // 用 cv::imwrite 前确保目录存在 (C++17 filesystem)
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    // 拼接: 上排静音 N 帧, 下排语音 N 帧
    int tile_w = 96, tile_h = 96, gap = 4;
    cv::Mat montage(2 * tile_h + gap, N * tile_w + (N - 1) * gap, CV_8UC3, cv::Scalar(64, 64, 64));
    for (int i = 0; i < N; ++i) {
        cv::Mat s(out_silent[0].h, out_silent[0].w, CV_8UC3), v(out_voice[0].h, out_voice[0].w, CV_8UC3);
        // 复用 saveMatPNG 的转换逻辑直接写 cv::Mat
        for (int y = 0; y < 96; ++y) for (int x = 0; x < 96; ++x) {
            int idx = y * 96 + x;
            s.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (uchar)std::clamp(out_silent[i].channel(2)[idx] * 255.f, 0.f, 255.f),
                (uchar)std::clamp(out_silent[i].channel(1)[idx] * 255.f, 0.f, 255.f),
                (uchar)std::clamp(out_silent[i].channel(0)[idx] * 255.f, 0.f, 255.f));
            v.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (uchar)std::clamp(out_voice[i].channel(2)[idx] * 255.f, 0.f, 255.f),
                (uchar)std::clamp(out_voice[i].channel(1)[idx] * 255.f, 0.f, 255.f),
                (uchar)std::clamp(out_voice[i].channel(0)[idx] * 255.f, 0.f, 255.f));
        }
        cv::Rect rs(i * (tile_w + gap), 0, tile_w, tile_h);
        cv::Rect rv(i * (tile_w + gap), tile_h + gap, tile_w, tile_h);
        s.copyTo(montage(rs));
        v.copyTo(montage(rv));
    }
    std::string montage_path = out_dir + "/montage_silent_vs_voice.png";
    cv::imwrite(montage_path, montage);
    std::cout << "[saved] " << montage_path << " (上=静音, 下=语音)\n";

    // 单独保存首末对比
    saveMatPNG(out_silent[0], out_dir + "/silent_w0.png");
    saveMatPNG(out_silent[N - 1], out_dir + "/silent_wlast.png");
    saveMatPNG(out_voice[0],  out_dir + "/voice_w0.png");
    saveMatPNG(out_voice[N - 1],  out_dir + "/voice_wlast.png");

    // ---- 7. 判定 ----
    std::cout << "\n==============================================\n";
    std::cout << "  结论\n";
    std::cout << "==============================================\n";
    if (mel_diff_avg < 0.1) {
        std::cout << "  [无效] Mel 输入差异过小，无法诊断\n";
        return 3;
    }
    if (diff_full_avg < 0.002 && diff_mouth_avg < 0.005) {
        std::cout << "  [无响应] 模型输出几乎不随音频变化\n";
        std::cout << "  -> 问题在模型权重或输入格式 (mel 布局/归一化/face c=6)\n";
        return 2;
    }
    if (diff_full_avg > 0.01 || diff_mouth_avg > 0.02) {
        std::cout << "  [有响应] 模型输出随音频显著变化\n";
        std::cout << "  -> 推理有效, 问题在后处理/融合/测量指标\n";
        return 0;
    }
    std::cout << "  [弱响应] 模型输出有变化但不明显\n";
    std::cout << "  -> 可能是模型质量差或 mel 预处理与训练不匹配\n";
    return 1;
}
