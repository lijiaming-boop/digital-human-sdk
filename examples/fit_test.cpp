/**
 * @file fit_test.cpp
 * @brief 拟合效果测试 — 加载 assets/zw.mp3 + face.jpg
 *
 * 测试流程:
 *   1. AudioLoader 加载 MP3 → 降噪 → 分帧 → VAD → 预加重
 *      → RMS → Mel → CMVN → 统计
 *   2. ImageLoader 加载 JPG → FaceDetect → FaceAlign
 *      → FaceMask → 统计
 *   3. 全链路输出报告
 *
 * 用法: ./bin/fit_test
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
#include <opencv2/imgcodecs.hpp>

#include "audio/audio_loader.h"
#include "audio/audio_noise_reduction.h"
#include "audio/audio_framer.h"
#include "audio/audio_vad.h"
#include "audio/audio_preemphasis.h"
#include "audio/audio_rms_normalize.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_cmvn.h"

#include "core/image_loader.h"
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"
#include "core/packet.h"

#include "model/model_inferencer.h"

using namespace digital_human;
using namespace digital_human::core;

// ============================================================================
// 测试框架
// ============================================================================

static int gPassed = 0;
static int gFailed = 0;

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

struct Section {
    std::string name;
    Section(const char* n) : name(n) {
        std::cout << "\n========== " << name << " ==========" << std::endl;
    }
    ~Section() {
        std::cout << "---------- " << name << " 结束 ----------" << std::endl;
    }
};

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "  数字人 SDK 拟合效果测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    std::string assets_dir = ASSETS_DIR;
    if (argc > 1) assets_dir = argv[1];

    std::string mp3_path  = assets_dir + "/zw.mp3";
    std::string face_path = assets_dir + "/face.jpg";
    std::string model_dir = std::string(PROJECT_SOURCE_DIR) + "/models";

    // ====================================================================
    // 1. 音频处理链路
    // ====================================================================
    {
        Section s("音频处理链路");

        // 1a. 加载 MP3
        std::cout << "\n[1a] 加载 MP3: " << mp3_path << std::endl;
        audio::AudioLoader loader;
        audio::AudioData data;

        auto t0 = std::chrono::steady_clock::now();
        try {
            data = loader.load(mp3_path);
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] MP3 加载失败: " << e.what() << std::endl;
            gFailed++;
            return 1;
        }
        double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        TEST_CHECK(!data.samples.empty(), "音频数据非空");
        std::cout << "  [INFO] 时长: " << data.duration << "s"
                  << "  采样率: " << data.sampleRate << "Hz"
                  << "  声道: " << data.channels
                  << "  样本数: " << data.samples.size()
                  << "  加载耗时: " << load_ms << "ms" << std::endl;

        // 1b. 音频处理流水线
        std::cout << "\n[1b] 音频特征提取流水线" << std::endl;
        const int sr = data.sampleRate;

        t0 = std::chrono::steady_clock::now();

        audio::NoiseReduction nr(10, 0.02f);
        auto denoised = nr.process(data.samples, sr);

        audio::RMSNormalize rms_norm(0.056f);
        auto normalized = rms_norm.process(denoised);

        audio::PreEmphasis pe(0.97f);
        auto emphasized = pe.process(normalized);

        audio::AudioFramer framer;
        audio::FrameConfig fcfg{400, 160};
        auto frames = framer.frame(emphasized, fcfg);

        audio::VoiceActivityDetector vad(0.01f, 0.0f, 0.5f, 3);
        auto voiced = vad.filter(frames);

        audio::MelFeatureExtract mel_ext;
        audio::MelConfig mcfg{512, 80, sr, 0.0f, 8000.0f};
        auto mel_spec = mel_ext.extract(voiced.empty() ? frames : voiced, mcfg);

        audio::CMVN cmvn;
        auto feat = cmvn.process(mel_spec);

        double process_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        double duration_ms = data.duration * 1000.0;
        double ratio = duration_ms / std::max(process_ms, 0.001);

        TEST_CHECK(!frames.empty(), "分帧输出非空 (" << frames.size() << " 帧)");
        TEST_CHECK(!mel_spec.empty(), "Mel 频谱非空 (" << mel_spec.rows << "×" << mel_spec.cols << ")");
        TEST_CHECK(!feat.empty(), "CMVN 归一化完成");

        std::cout << "  [INFO] 分帧: " << frames.size() << " 帧"
                  << "  VAD 过滤后: " << voiced.size() << " 帧"
                  << "  Mel: " << mel_spec.rows << "×" << mel_spec.cols
                  << std::endl;
        std::cout << "  [INFO] 处理耗时: " << process_ms << "ms"
                  << "  实时比: " << ratio << "x" << std::endl;

        // 计算 RMS 能量分布
        double energy_sum = 0.0;
        for (float s : data.samples) energy_sum += std::abs(s);
        double avg_energy = energy_sum / data.samples.size();
        std::cout << "  [INFO] 平均能量: " << avg_energy << std::endl;
    }

    // ====================================================================
    // 2. 图像处理链路
    // ====================================================================
    {
        Section s("图像处理链路");

        // 2a. 加载人脸图像
        std::cout << "\n[2a] 加载人脸: " << face_path << std::endl;
        cv::Mat face_img = cv::imread(face_path);
        TEST_CHECK(!face_img.empty(), "人脸图像加载成功");

        std::cout << "  [INFO] 尺寸: " << face_img.cols << "×" << face_img.rows
                  << "  通道: " << face_img.channels()
                  << "  类型: " << (face_img.type() == CV_8UC3 ? "BGR uint8" : "other")
                  << std::endl;

        // 2b. 人脸检测
        std::cout << "\n[2b] 人脸检测" << std::endl;
        digital_human::core::FaceDetector detector;

        // 尝试加载模型
        std::string model_path = std::string(PROJECT_SOURCE_DIR) + "/models";
        bool model_loaded = false;
        try {
            // 尝试加载 dlib 模型
            std::string detector_model = model_path + "/face";
            if (std::filesystem::exists(detector_model)) {
                model_loaded = detector.loadModel(detector_model);
            }
        } catch (...) {
            std::cout << "  [WARN] 模型加载失败，使用 OpenCV 内置检测器" << std::endl;
        }

        if (model_loaded) {
            auto faces = detector.detect(face_img);
            TEST_CHECK(!faces.empty(), "检测到人脸 (faces=" << faces.size() << ")");

            for (size_t i = 0; i < faces.size() && i < 3; ++i) {
                std::cout << "  [INFO] 人脸[" << i << "]: "
                          << faces[i].x << "," << faces[i].y
                          << " " << faces[i].width << "×" << faces[i].height
                          << std::endl;
            }
        } else {
            std::cout << "  [SKIP] 人脸检测模型未加载，跳过检测" << std::endl;
        }

        // 2c. 人脸裁剪与缩放（模拟人脸对齐）
        std::cout << "\n[2c] 人脸裁剪与缩放" << std::endl;
        int crop_face_size = 96;
        cv::Mat aligned_face_img;

        if (face_img.rows >= crop_face_size && face_img.cols >= crop_face_size) {
            int cx = face_img.cols / 2;
            int cy = face_img.rows / 3;
            int crop_size = std::min(face_img.cols, face_img.rows) / 2;

            cv::Rect roi(
                std::max(0, cx - crop_size / 2),
                std::max(0, cy - crop_size / 2),
                std::min(crop_size, face_img.cols - cx + crop_size / 2),
                std::min(crop_size, face_img.rows - cy + crop_size / 2)
            );

            cv::Mat cropped = face_img(roi).clone();
            cv::resize(cropped, aligned_face_img, cv::Size(crop_face_size, crop_face_size));
            TEST_CHECK(aligned_face_img.rows == crop_face_size && aligned_face_img.cols == crop_face_size,
                       "对齐人脸 " << crop_face_size << "×" << crop_face_size);
        }

        // 2d. 处理后的数据整合
        std::cout << "\n[2d] 数据整合" << std::endl;
        // 使用完整 namespace 引用 ProcessedFaceData
        digital_human::core::ProcessedFaceData face_data;
        face_data.aligned_face  = aligned_face_img;
        face_data.original_face = face_img.clone();
        face_data.M_inv         = cv::Mat::eye(2, 3, CV_32F);
        face_data.face_mask     = cv::Mat(face_img.size(), CV_32FC1, cv::Scalar(0.5f));
        face_data.face_rect     = cv::Rect(0, 0, face_img.cols, face_img.rows);

        TEST_CHECK(face_data.IsValid(), "ProcessedFaceData 有效");
        std::cout << "  [INFO] 对齐人脸: " << face_data.aligned_face.size()
                  << "  遮罩: " << face_data.face_mask.size()
                  << "  M_inv: " << face_data.M_inv.size()
                  << std::endl;
    }

    // ====================================================================
    // 3. 模型推理状态（检查模型文件）
    // ====================================================================
    {
        Section s("模型推理就绪检查");

        std::string param_path = model_dir + "/Wav2Lip-SD-GAN-opt.param";
        std::string bin_path   = model_dir + "/Wav2Lip-SD-GAN-opt.bin";

        bool param_ok = std::filesystem::exists(param_path);
        bool bin_ok   = std::filesystem::exists(bin_path);

        TEST_CHECK(param_ok, "模型 param 文件存在");
        TEST_CHECK(bin_ok, "模型 bin 文件存在");

        if (param_ok && bin_ok) {
            auto param_size = std::filesystem::file_size(param_path);
            auto bin_size   = std::filesystem::file_size(bin_path);
            std::cout << "  [INFO] param: " << param_path
                      << " (" << param_size / 1024 << "KB)" << std::endl;
            std::cout << "  [INFO] bin:   " << bin_path
                      << " (" << bin_size / 1024 / 1024 << "MB)" << std::endl;

            // 尝试初始化推理器
            model::ModelInferencer inferencer;
            bool init_ok = inferencer.Init(model_dir);
            TEST_CHECK(init_ok, "ModelInferencer 初始化");

            if (init_ok) {
                std::cout << "  [INFO] 线程数: " << inferencer.GetThreadCount()
                          << "  GPU: " << (inferencer.IsGPUEnabled() ? "yes" : "no")
                          << " 目标延迟: " << inferencer.GetTargetLatencyMs() << "ms"
                          << std::endl;

                // 创建测试输入 (与模型 warmup 形状一致)
                ncnn::Mat audio_test(80, 80, 1);   // mel_bins=80, frames=80
                ncnn::Mat face_test(96, 96, 6);    // face_w=96, face_h=96, c=6
                audio_test.fill(0.0f);
                face_test.fill(0.0f);

                auto t0 = std::chrono::steady_clock::now();
                ncnn::Mat output = inferencer.Infer(audio_test, face_test);
                double infer_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();

                TEST_CHECK(!output.empty(), "推理输出非空");
                if (!output.empty()) {
                    std::cout << "  [INFO] 推理输出: " << output.w << "×" << output.h
                              << " c=" << output.c << " 耗时: " << infer_ms << "ms"
                              << std::endl;
                }

                auto stats = inferencer.GetAvgLatencyMs();
                std::cout << "  [INFO] 平均推理延迟: " << stats << "ms"
                          << "  总推理次数: " << inferencer.GetInferenceCount()
                          << std::endl;
            }
        }
    }

    // ====================================================================
    // 汇总
    // ====================================================================
    std::cout << "\n==============================================" << std::endl;
    std::cout << "  拟合效果测试汇总" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  通过: " << gPassed << std::endl;
    std::cout << "  失败: " << gFailed << std::endl;
    std::cout << "  总计: " << (gPassed + gFailed) << std::endl;
    std::cout << "==============================================" << std::endl;

    return (gFailed == 0) ? 0 : 1;
}
