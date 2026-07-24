/**
 * @file video_output_test.cpp
 * @brief 音视频输出测试 — 生成 MP4 视频 + 音频
 *
 * 流程:
 *   1. 加载 face.jpg → 对齐 (96x96)
 *   2. 加载音频 → Mel 特征
 *   3. 逐段推理生成唇形帧
 *   4. 写入 MP4 视频文件
 *   5. 输出原始音频供合并
 *
 * 用法: ./bin/video_output_test [帧数] [音频文件名 (相对 assets/)]
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include <filesystem>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <ncnn/mat.h>

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

// ============================================================================
// BGR → ncnn::Mat (96x96x6)
// 模型输入: ch0-2 = 下半脸遮罩的人脸, ch3-5 = 完整人脸 (Wav2Lip 标准格式)
// ============================================================================
static ncnn::Mat faceToNCNN(const cv::Mat& bgr, int w, int h) {
    ncnn::Mat out(w, h, 6);
    const int mask_from = h / 2;  // 下半部分 (y >= 48) 在 ch0-2 置零
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            auto px = bgr.at<cv::Vec3b>(y, x);
            float r = px[2] / 255.0f, g = px[1] / 255.0f, b = px[0] / 255.0f;
            bool masked = (y >= mask_from);
            out.channel(0).row(y)[x] = masked ? 0.0f : r;
            out.channel(1).row(y)[x] = masked ? 0.0f : g;
            out.channel(2).row(y)[x] = masked ? 0.0f : b;
            out.channel(3).row(y)[x] = r;
            out.channel(4).row(y)[x] = g;
            out.channel(5).row(y)[x] = b;
        }
    }
    return out;
}

// ============================================================================
// Mel 特征行 → ncnn::Mat
// 模型输入布局: w=时间帧数(16), h=mel bins(80), c=1
// row(bin)[frame] = feat(frame, bin)
// ============================================================================
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

// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "  音视频输出测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    std::string assets_dir = ASSETS_DIR;
    std::string model_dir  = std::string(PROJECT_SOURCE_DIR) + "/models";
    std::string out_dir    = assets_dir + "/output";
    std::filesystem::create_directories(out_dir);

    // 生成帧数 (默认 1500 帧 = 60秒 @25fps)
    int target_frames = 1500;
    if (argc > 1) target_frames = std::stoi(argv[1]);
    target_frames = std::min(target_frames, 2000);

    // 音频文件名 (相对 assets/, 默认 zw_trimmed.mp3)
    std::string audio_name = "zw_trimmed.mp3";
    if (argc > 2) audio_name = argv[2];

    double fps = 25.0;
    int hop_frames = 4;  // 25fps = 每帧 4 Mel 帧

    // ====================================================================
    // 1. 人脸处理
    // ====================================================================
    std::cout << "\n[1/5] 人脸处理..." << std::endl;
    cv::Mat original = cv::imread(assets_dir + "/face.jpg");
    if (original.empty()) { std::cerr << "FAIL: face.jpg\n"; return 1; }

    digital_human::core::FaceDetector detector;
    detector.loadModel(model_dir + "/shape_predictor_68_face_landmarks.dat");
    auto faces = detector.detect(original);
    if (faces.empty()) { std::cerr << "FAIL: no face\n"; return 1; }

    auto pts = detector.getLandmarks(original, faces[0]);
    digital_human::core::FaceAligner aligner;
    std::vector<cv::Point2f> lmf;
    for (auto& p : pts) lmf.emplace_back((float)p.x, (float)p.y);
    auto ar = aligner.alignByRect(original, lmf, 96, faces[0]);
    if (!ar.valid) { std::cerr << "FAIL: align\n"; return 1; }

    ncnn::Mat face_ncnn = faceToNCNN(ar.aligned_face, 96, 96);
    digital_human::core::FaceMaskGenerator mask_gen;
    cv::Mat mask = mask_gen.generateMouthMask(original.size(), pts);

    std::cout << "  [OK] 人脸 " << faces[0].width << "x" << faces[0].height
              << " → 96x96" << std::endl;

    // ====================================================================
    // 2. 音频 Mel 特征
    // ====================================================================
    std::cout << "[2/5] Mel 特征..." << std::endl;
    audio::AudioLoader loader;
    auto audio = loader.load(assets_dir + "/" + audio_name);

    audio::NoiseReduction nr(10, 0.02f);
    audio::RMSNormalize rn(0.056f);
    audio::PreEmphasis pe(0.97f);
    audio::AudioFramer fr;
    auto frames = fr.frame(
        pe.process(rn.process(nr.process(audio.samples, audio.sampleRate))),
        {400, 160});
    audio::MelFeatureExtract me;
    audio::CMVN cmvn;
    auto feat = cmvn.process(
        me.extract(frames, {512, 80, audio.sampleRate, 0, 8000}));

    std::cout << "  [OK] " << feat.rows << "x" << feat.cols << std::endl;

    // ====================================================================
    // 3. 模型
    // ====================================================================
    std::cout << "[3/5] 模型..." << std::endl;
    model::ModelInferencer inf;
    if (!inf.Init(model_dir)) { std::cerr << "FAIL: model\n"; return 1; }
    model::OutputProcessor op;
    std::cout << "  [OK] " << inf.GetThreadCount() << "t" << std::endl;

    // ====================================================================
    // 4. 推理
    // ====================================================================
    // 模型音频窗口 = 16 个 Mel 帧 (Wav2Lip syncnet_mel_step_size)
    // 25fps 视频每帧对应 4 个 Mel 帧 (hop=160 @16kHz, 10ms/帧)
    int ctx = 16;
    int total = std::min(target_frames,
                         (feat.rows - ctx) / hop_frames);
    total = std::max(total, 1);

    std::cout << "[4/5] 推理 " << total << " 帧..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    // 准备输出目录（保存 PNG 帧到磁盘，避免内存 OOM）
    std::string frames_dir = out_dir + "/frames";
    std::filesystem::create_directories(frames_dir);
    int saved_count = 0;

    for (int i = 0; i < total; ++i) {
        int start = i * hop_frames;
        if (start > feat.rows - ctx) start = feat.rows - ctx;
        if (start < 0) start = 0;

        ncnn::Mat output = inf.Infer(melToNCNN(feat, start, ctx), face_ncnn);
        if (!output.empty()) {
            cv::Mat frame = op.Process(output, original, mask, ar.M_inv);
            if (!frame.empty()) {
                // 立即保存到磁盘，避免内存累积
                char fname[64];
                std::snprintf(fname, sizeof(fname), "frame_%04d.png", i);
                cv::imwrite(frames_dir + "/" + fname, frame);
                saved_count++;
            }
        }

        double pct = 100.0 * (i + 1) / total;
        double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        double eta = (el / (i + 1)) * (total - i - 1);
        printf("\r  [%4.0f%%] %d/%d  %.0fms/f ETA=%.0fs   ",
               pct, i+1, total, el*1000/(i+1), eta);
        std::cout << std::flush;
    }

    double total_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("\n  [OK] %d 帧, %.1fs, %.0fms/帧\n",
           saved_count, total_s, total_s * 1000 / std::max(total, 1));

    // ====================================================================
    // 5. 视频（从磁盘 PNG 帧序列合成）
    // ====================================================================
    std::cout << "[5/5] 生成视频..." << std::endl;
    cv::Size sz(original.cols, original.rows);

    // 使用 OpenCV VideoWriter 尝试生成视频（如果后端可用）
    std::string vpath = out_dir + "/digital_human.mp4";
    cv::VideoWriter w;
    if (w.open(vpath, cv::VideoWriter::fourcc('m','p','4','v'),
               fps, sz, true) ||
        w.open(vpath, cv::VideoWriter::fourcc('M','J','P','G'),
               fps, sz, true)) {
        int written = 0;
        for (int i = 0; i < saved_count; ++i) {
            char fname[64];
            std::snprintf(fname, sizeof(fname), "%s/frame_%04d.png",
                         frames_dir.c_str(), i);
            cv::Mat f = cv::imread(fname);
            if (!f.empty()) { w.write(f); written++; }
        }
        w.release();
        auto fsize = std::filesystem::file_size(vpath) / 1024;
        std::cout << "  视频: " << vpath << " (" << fsize << "KB, "
                  << written << "帧)" << std::endl;
    } else {
        std::cout << "  视频: OpenCV VideoWriter 不可用" << std::endl;
    }

    std::cout << "  帧保存至: " << frames_dir << " (" << saved_count << " 帧)" << std::endl;
    std::cout << "  使用 ffmpeg 合成: ffmpeg -framerate " << fps
              << " -i " << frames_dir << "/frame_%04d.png"
              << " -i " << assets_dir << "/" << audio_name
              << " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "
              << out_dir << "/final_with_audio.mp4" << std::endl;
}
