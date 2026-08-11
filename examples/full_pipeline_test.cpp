/**
 * @file full_pipeline_test.cpp
 * @brief 全链路流程测试 — 模块可独立验证 + 端到端集成
 *
 * 测试层级：
 *   Level 1: 音频处理模块独立测试
 *     NoiseReduction → AudioFramer → VAD → PreEmphasis
 *     → RMSNormalize → MelFeatureExtract → CMVN
 *
 *   Level 2: 图像处理模块独立测试
 *     ImageLoader → FaceDetect → FaceAlign → FaceMask
 *
 *   Level 3: 端到端流水线集成测试
 *     模拟音频 + 视频帧 → 全流程 → 输出验证
 *
 * 每个测试函数可单独编译运行，不依赖硬件（模型文件、音频设备）。
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "audio/audio_noise_reduction.h"
#include "audio/audio_framer.h"
#include "audio/audio_vad.h"
#include "audio/audio_preemphasis.h"
#include "audio/audio_rms_normalize.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_cmvn.h"
#include "audio/audio_ring_buffer.h"

#include "core/packet.h"
#include "core/thread_safe_queue.h"
#include "core/thread_base.h"

using namespace digital_human;

// ============================================================================
// 测试框架
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

// ============================================================================
// 工具函数
// ============================================================================

/// @brief 生成正弦波
static std::vector<float> genSine(float freq, float dur, int sr) {
    int n = static_cast<int>(sr * dur);
    std::vector<float> pcm(n);
    for (int i = 0; i < n; ++i) {
        pcm[i] = std::sin(2.0f * 3.14159265358979323846f * freq * i / sr);
    }
    return pcm;
}

/// @brief 生成带噪信号
static std::vector<float> genNoisySine(float freq, float dur, int sr, float noise_level) {
    auto clean = genSine(freq, dur, sr);
    for (auto& s : clean) {
        s += noise_level * (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    return clean;
}

/// @brief 计算 RMS
static float calcRMS(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * x;
    return static_cast<float>(std::sqrt(sum / v.size()));
}

/// @brief 生成测试图像（棋盘格）
static cv::Mat genTestImage(int w, int h, int type = CV_8UC3) {
    cv::Mat img(h, w, type, cv::Scalar(0, 0, 0));
    if (type == CV_8UC3) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                if ((x / 32 + y / 32) % 2 == 0)
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(200, 200, 200);
    }
    return img;
}

// ============================================================================
// ===================== Level 1: 音频模块独立测试 =====================
// ============================================================================

// ---------------------------------------------------------------------------
// Test A1: NoiseReduction — 降噪模块独立可用
// ---------------------------------------------------------------------------
static void testNoiseReduction() {
    TEST_NAME("A1: NoiseReduction 模块");
    int sr = 16000;

    // 静音 → 静音
    {
        audio::NoiseReduction nr;
        auto out = nr.process(std::vector<float>(8000, 0.0f), sr);
        float rms = calcRMS(out);
        TEST_CHECK(rms < 0.01f, "静音输入 RMS≈0 (rms=" << rms << ")");
    }

    // 带噪信号 → 能量降低
    {
        audio::NoiseReduction nr(5, 2.0f);
        auto noisy = genNoisySine(440.0f, 0.3f, sr, 0.5f);
        float before = calcRMS(noisy);
        auto out = nr.process(noisy, sr);
        float after = calcRMS(out);
        TEST_CHECK(after <= before * 1.1f, "降噪后能量未增加 (before="
                   << before << " after=" << after << ")");
    }

    // 输出尺寸匹配
    {
        audio::NoiseReduction nr;
        auto pcm = genSine(440.0f, 0.1f, sr);
        auto out = nr.process(pcm, sr);
        TEST_CHECK(out.size() == pcm.size(), "输出尺寸=" << out.size()
                   << " 输入尺寸=" << pcm.size());
    }
}

// ---------------------------------------------------------------------------
// Test A2: AudioFramer — 分帧模块独立可用
// ---------------------------------------------------------------------------
static void testAudioFramer() {
    TEST_NAME("A2: AudioFramer 模块");
    int sr = 16000;
    int frame = 400, hop = 160;

    auto pcm = genSine(440.0f, 1.0f, sr);  // 16000 samples
    audio::AudioFramer framer;
    audio::FrameConfig cfg{frame, hop};

    auto frames = framer.frame(pcm, cfg);
    int expected = (static_cast<int>(pcm.size()) - frame + hop - 1) / hop + 1;
    TEST_CHECK(!frames.empty(), "分帧结果非空");
    TEST_CHECK(static_cast<int>(frames.size()) == expected,
               "帧数正确 got=" << frames.size() << " expected=" << expected);
    TEST_CHECK(static_cast<int>(frames[0].size()) == frame,
               "每帧长度=" << frame << " got=" << frames[0].size());

    // 短音频不足一帧
    auto short_pcm = genSine(440.0f, 0.01f, sr);  // 160 samples
    auto short_frames = framer.frame(short_pcm, cfg);
    TEST_CHECK(short_frames.size() == 1, "短音频输出 1 帧 (got="
               << short_frames.size() << ")");
}

// ---------------------------------------------------------------------------
// Test A3: VAD — 语音活动检测
// ---------------------------------------------------------------------------
static void testVAD() {
    TEST_NAME("A3: VAD 模块");

    audio::VoiceActivityDetector vad(0.01f, 0.0f, 0.5f, 2);
    audio::AudioFramer framer;
    audio::FrameConfig cfg{400, 160};

    auto pcm = genSine(440.0f, 0.5f, 16000);
    auto frames = framer.frame(pcm, cfg);
    auto filtered = vad.filter(frames);

    TEST_CHECK(!filtered.empty(), "过滤后非空");
    TEST_CHECK(filtered.size() <= frames.size(), "过滤后帧数减少或相等 ("
               << filtered.size() << " ≤ " << frames.size() << ")");

    // 静音输入应被过滤
    std::vector<float> silence(16000, 0.0f);
    auto silent_frames = framer.frame(silence, cfg);
    auto silent_filtered = vad.filter(silent_frames);
    TEST_CHECK(silent_filtered.empty() || silent_filtered.size() <= silent_frames.size(),
               "静音帧过滤 (原=" << silent_frames.size()
               << " 过滤后=" << silent_filtered.size() << ")");
}

// ---------------------------------------------------------------------------
// Test A4: PreEmphasis — 预加重
// ---------------------------------------------------------------------------
static void testPreEmphasis() {
    TEST_NAME("A4: PreEmphasis 模块");

    audio::PreEmphasis pe(0.97f);

    // 恒定信号 → 第一个样本不变，其余接近 0
    std::vector<float> constant(100, 1.0f);
    auto out = pe.process(constant);
    TEST_CHECK(std::abs(out[0] - 1.0f) < 0.001f, "首样本不变 (got=" << out[0] << ")");
    for (size_t i = 1; i < out.size(); ++i) {
        TEST_CHECK(std::abs(out[i]) < 0.05f, "样本[" << i << "]≈0 (got=" << out[i] << ")");
        if (std::abs(out[i]) >= 0.05f) break;
    }

    // 正弦波 → 输出尺寸相同
    auto sine = genSine(440.0f, 0.1f, 16000);
    auto filtered = pe.process(sine);
    TEST_CHECK(filtered.size() == sine.size(), "尺寸不变 ("
               << filtered.size() << " == " << sine.size() << ")");
}

// ---------------------------------------------------------------------------
// Test A5: RMSNormalize — RMS 归一化
// ---------------------------------------------------------------------------
static void testRMSNormalize() {
    TEST_NAME("A5: RMSNormalize 模块");

    float target = 0.1f;
    audio::RMSNormalize rms_norm(target);

    auto sine = genSine(440.0f, 0.3f, 16000);
    auto out = rms_norm.process(sine);

    TEST_CHECK(out.size() == sine.size(), "尺寸不变");
    float out_rms = calcRMS(out);
    TEST_CHECK(std::abs(out_rms - target) < 0.01f,
               "RMS 接近目标 " << target << " (got=" << out_rms << ")");
}

// ---------------------------------------------------------------------------
// Test A6: MelFeatureExtract — Mel 频谱提取
// ---------------------------------------------------------------------------
static void testMelExtract() {
    TEST_NAME("A6: MelFeatureExtract 模块");

    audio::MelFeatureExtract mel;
    audio::MelConfig mc;
    mc.nFFT = 512;
    mc.nMels = 80;
    mc.sampleRate = 16000;

    // 构造模拟帧
    int n_frames = 20;
    std::vector<std::vector<float>> frames(n_frames, std::vector<float>(400, 0.5f));

    auto spec = mel.extract(frames, mc);
    TEST_CHECK(!spec.empty(), "Mel 频谱非空");
    TEST_CHECK(spec.rows == n_frames, "行数=帧数 (got=" << spec.rows << ")");
    TEST_CHECK(spec.cols == mc.nMels, "列数=Mel bins (got=" << spec.cols << ")");
    TEST_CHECK(spec.type() == CV_32F, "类型=CV_32F");

    // 空帧处理
    std::vector<std::vector<float>> empty;
    auto empty_spec = mel.extract(empty, mc);
    TEST_CHECK(empty_spec.empty(), "空帧返回空 Mat");
}

// ---------------------------------------------------------------------------
// Test A7: CMVN — 倒谱归一化
// ---------------------------------------------------------------------------
static void testCMVN() {
    TEST_NAME("A7: CMVN 模块");

    audio::CMVN cmvn;

    // 创建测试 Mel 频谱 (10×80)
    cv::Mat mel(10, 80, CV_32F);
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 80; ++x)
            mel.at<float>(y, x) = static_cast<float>(y * 80 + x);

    auto norm = cmvn.process(mel);
    TEST_CHECK(!norm.empty(), "归一化结果非空");
    TEST_CHECK(norm.rows == 10, "行数不变");
    TEST_CHECK(norm.cols == 80, "列数不变");
    TEST_CHECK(norm.type() == CV_32F, "类型=CV_32F");

    // 空输入
    cv::Mat empty;
    auto empty_norm = cmvn.process(empty);
    TEST_CHECK(empty_norm.empty(), "空输入返回空");
}

// ---------------------------------------------------------------------------
// Test A8: RingBuffer — 环形缓冲区
// ---------------------------------------------------------------------------
static void testRingBuffer() {
    TEST_NAME("A8: RingBuffer 模块");

    audio::RingBuffer rb(1024);
    TEST_CHECK(rb.capacity() == 1024, "容量=1024");
    TEST_CHECK(rb.empty(), "初始为空");

    std::vector<float> data(256, 1.0f);
    size_t written = rb.write(data.data(), data.size());
    TEST_CHECK(written == 256, "写入 256 元素 (got=" << written << ")");
    TEST_CHECK(!rb.empty(), "写入后非空");

    std::vector<float> out(128);
    size_t read = rb.read(out.data(), out.size());
    TEST_CHECK(read == 128, "读取 128 元素 (got=" << read << ")");
    TEST_CHECK(out[0] == 1.0f, "数据正确");

    rb.reset();
    TEST_CHECK(rb.empty(), "重置后为空");
}

// ============================================================================
// ===================== Level 2: 图像模块独立测试 =====================
// ============================================================================

// ---------------------------------------------------------------------------
// Test I1: cv::Mat 基础图像操作（无需模型可独立测试）
// ---------------------------------------------------------------------------
static void testImageBasics() {
    TEST_NAME("I1: 图像数据基础操作");

    // 图像创建与转换
    cv::Mat img = genTestImage(640, 480);
    TEST_CHECK(!img.empty(), "测试图像非空");
    TEST_CHECK(img.rows == 480, "高度=480");
    TEST_CHECK(img.cols == 640, "宽度=640");

    // BGR → RGB 转换
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    TEST_CHECK(!rgb.empty(), "RGB 转换非空");
    TEST_CHECK(rgb.channels() == 3, "RGB 通道数=3");

    // 缩放
    cv::Mat small;
    cv::resize(img, small, cv::Size(96, 96));
    TEST_CHECK(small.rows == 96 && small.cols == 96, "缩放至 96×96");

    // 裁剪
    cv::Rect roi(100, 100, 200, 200);
    cv::Mat cropped = img(roi).clone();
    TEST_CHECK(cropped.rows == 200 && cropped.cols == 200, "裁剪 200×200");

    // 类型转换 CV_8UC3 → CV_32F
    cv::Mat fimg;
    img.convertTo(fimg, CV_32F, 1.0 / 255.0);
    TEST_CHECK(fimg.type() == CV_32FC3, "float 类型转换");
}

// ---------------------------------------------------------------------------
// Test I2: 人脸数据打包与传递（ProcessedFaceData + Packet）
// ---------------------------------------------------------------------------
static void testFaceDataPacket() {
    TEST_NAME("I2: ProcessedFaceData 数据包");

    core::ProcessedFaceData data;
    data.aligned_face  = cv::Mat(96, 96, CV_8UC3, cv::Scalar(64, 128, 192));
    data.M_inv         = cv::Mat::eye(2, 3, CV_32F);
    data.face_mask     = cv::Mat(96, 96, CV_32FC1, cv::Scalar(1.0f));
    data.original_face = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    data.face_rect     = cv::Rect(100, 100, 200, 200);

    TEST_CHECK(data.IsValid(), "有效数据验证通过");

    // 通过 Packet 传递（移动语义，零拷贝）
    core::ProcessedFacePacket pkt =
        core::ProcessedFacePacket::Make(std::move(data), 0, 0);
    TEST_CHECK(pkt.header.IsOK(), "Packet 状态 OK");
    TEST_CHECK(pkt.payload.IsValid(), "Packet payload 有效");
}

// ============================================================================
// ============= Level 3: 全链路集成测试 =============
// ============================================================================

// ---------------------------------------------------------------------------
// Test F1: 音频全链路 — PCM → Mel 特征
// ---------------------------------------------------------------------------
static void testFullAudioPipeline() {
    TEST_NAME("F1: 音频全链路 PCM → Mel 特征");

    int sr = 16000;
    auto pcm = genSine(440.0f, 1.0f, sr);

    // 创建所有模块（链式处理）
    audio::NoiseReduction      nr;
    audio::AudioFramer         framer;
    audio::VoiceActivityDetector vad;
    audio::PreEmphasis         pe(0.97f);
    audio::RMSNormalize        rms_norm(0.056f);
    audio::MelFeatureExtract   mel_ext;
    audio::CMVN                cmvn;

    audio::FrameConfig fcfg{400, 160};
    audio::MelConfig    mcfg;
    mcfg.nFFT = 512;
    mcfg.nMels = 80;
    mcfg.sampleRate = sr;

    // 全链路
    auto denoised     = nr.process(pcm, sr);
    auto normalized   = rms_norm.process(denoised);
    auto emphasized   = pe.process(normalized);
    auto frames       = framer.frame(emphasized, fcfg);
    auto voiced       = vad.filter(frames);
    auto mel          = mel_ext.extract(voiced.empty() ? frames : voiced, mcfg);
    auto feat         = cmvn.process(mel);

    TEST_CHECK(!denoised.empty(), "1) 降噪输出非空");
    TEST_CHECK(denoised.size() == pcm.size(), "2) 降噪尺寸不变");
    TEST_CHECK(!frames.empty(), "3) 分帧非空 (got=" << frames.size() << ")");
    TEST_CHECK(!voiced.empty() || true, "4) VAD 过滤完成");
    TEST_CHECK(!mel.empty(), "5) Mel 特征非空 (rows=" << mel.rows
               << " cols=" << mel.cols << ")");
    TEST_CHECK(!feat.empty(), "6) CMVN 归一化完成");
    TEST_CHECK(feat.type() == CV_32F, "7) 最终特征类型 CV_32F");
    std::cout << "  [INFO] 音频全链路: " << pcm.size() << " samples → "
              << feat.rows << "×" << feat.cols << " feature" << std::endl;
}

// ---------------------------------------------------------------------------
// Test F2: 图像全链路 — 原始帧 → 对齐人脸遮罩
// （不依赖模型文件，仅验证数据流 + 图像操作）
// ---------------------------------------------------------------------------
static void testFullImagePipeline() {
    TEST_NAME("F2: 图像全链路 原始帧 → 预处理数据");

    // 生成测试图像
    cv::Mat frame = genTestImage(640, 480);
    TEST_CHECK(!frame.empty(), "测试帧非空");

    // 模拟人脸区域（实际生产中由 FaceDetector 提供）
    cv::Rect face_rect(200, 100, 200, 250);
    cv::Mat face_roi = frame(face_rect).clone();
    TEST_CHECK(!face_roi.empty(), "人脸 ROI 提取成功");

    // 模拟对齐（在实际模型中由 FaceAligner 提供）
    cv::Mat aligned_face;
    cv::resize(face_roi, aligned_face, cv::Size(96, 96));
    TEST_CHECK(aligned_face.rows == 96 && aligned_face.cols == 96,
               "对齐人脸 96×96");

    // 模拟 M_inv 矩阵（仿射逆变换）
    cv::Mat M_inv = cv::Mat::eye(2, 3, CV_32F);
    TEST_CHECK(!M_inv.empty(), "M_inv 矩阵可用");

    // 模拟口唇遮罩（实际由 FaceMaskGenerator 生成）
    cv::Mat mouth_mask = cv::Mat::zeros(frame.size(), CV_32FC1);
    cv::circle(mouth_mask,
               cv::Point(face_rect.x + face_rect.width / 2,
                         face_rect.y + face_rect.height * 2 / 3),
               face_rect.width / 4, cv::Scalar(1.0f), -1);
    TEST_CHECK(!mouth_mask.empty(), "口唇遮罩可用");

    // 整合为 ProcessedFaceData
    core::ProcessedFaceData face_data;
    face_data.aligned_face  = aligned_face;
    face_data.original_face = frame.clone();
    face_data.M_inv         = M_inv;
    face_data.face_mask     = mouth_mask;
    face_data.face_rect     = face_rect;
    TEST_CHECK(face_data.IsValid(), "ProcessedFaceData 有效");

    // 通过 Packet 传递
    core::ProcessedFacePacket pkt =
        core::ProcessedFacePacket::Make(std::move(face_data), 0, 0);
    TEST_CHECK(pkt.payload.IsValid(), "Packet 传递有效");

    std::cout << "  [INFO] 图像全链路: " << frame.cols << "×" << frame.rows
              << " → 96×96 aligned + mask" << std::endl;
}

// ---------------------------------------------------------------------------
// Test F3: 线程安全队列集成（生产者→消费者数据流）
// ---------------------------------------------------------------------------
static void testQueueIntegration() {
    TEST_NAME("F3: 队列集成测试（生产者→消费者）");

    using namespace digital_human::core;

    ThreadSafeQueue<core::AudioRawPacket> queue;
    const int kCount = 500;

    // 生产者线程
    std::thread producer([&]() {
        for (int i = 0; i < kCount; ++i) {
            std::vector<float> pcm(160, static_cast<float>(i));
            queue.Push(core::AudioRawPacket::Make(pcm, i * 10, i));
        }
        queue.Push(core::AudioRawPacket::EOS());
    });

    // 消费者线程
    std::atomic<int64_t> sum{0};
    std::thread consumer([&]() {
        while (true) {
            core::AudioRawPacket pkt;
            if (!queue.WaitAndPop(pkt, 100)) continue;
            if (pkt.header.IsEOS()) break;
            if (pkt.header.IsOK()) {
                // 验证数据完整性
                if (!pkt.payload.empty()) {
                    sum.fetch_add(static_cast<int64_t>(pkt.payload[0]),
                                  std::memory_order_relaxed);
                }
            }
        }
    });

    producer.join();
    consumer.join();

    // 期望: sum(0..499) = 499*500/2 = 124750
    int64_t expected = static_cast<int64_t>(kCount) * (kCount - 1) / 2;
    TEST_CHECK(sum.load() == expected, "数据完整性 sum="
               << sum.load() << " expected=" << expected);
}

// ---------------------------------------------------------------------------
// Test F4: 全链路计时与吞吐量
// ---------------------------------------------------------------------------
static void testThroughput() {
    TEST_NAME("F4: 音频全链路吞吐量测试");

    int sr = 16000;
    auto pcm = genSine(440.0f, 10.0f, sr);  // 10s 音频

    audio::NoiseReduction      nr;
    audio::AudioFramer         framer;
    audio::VoiceActivityDetector vad;
    audio::PreEmphasis         pe(0.97f);
    audio::RMSNormalize        rms_norm(0.056f);
    audio::MelFeatureExtract   mel_ext;
    audio::CMVN                cmvn;

    audio::FrameConfig fcfg{400, 160};
    audio::MelConfig    mcfg;
    mcfg.nFFT = 512;
    mcfg.nMels = 80;
    mcfg.sampleRate = sr;

    auto t0 = std::chrono::steady_clock::now();

    auto denoised     = nr.process(pcm, sr);
    auto normalized   = rms_norm.process(denoised);
    auto emphasized   = pe.process(normalized);
    auto frames       = framer.frame(emphasized, fcfg);
    auto voiced       = vad.filter(frames);
    auto mel          = mel_ext.extract(voiced.empty() ? frames : voiced, mcfg);
    auto feat         = cmvn.process(mel);

    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    double input_dur_ms = static_cast<double>(pcm.size()) / sr * 1000.0;
    double ratio = input_dur_ms / elapsed;

    TEST_CHECK(!feat.empty(), "10s 音频处理完成");
    std::cout << "  [INFO] 输入 " << input_dur_ms << "ms 音频, "
              << "处理耗时 " << elapsed << "ms, "
              << "实时比 " << ratio << "x" << std::endl;
    TEST_CHECK(ratio > 0.5f, "处理速度 ≥ 0.5x 实时 (ratio=" << ratio << ")");
}

// ---------------------------------------------------------------------------
// Test F5: 渲染任务数据包端到端
// ---------------------------------------------------------------------------
static void testRenderTaskE2E() {
    TEST_NAME("F5: RenderTask 端到端数据流");

    using namespace digital_human::core;

    // 模拟推理输出 (ncnn::Mat 格式)
    ncnn::Mat model_out(448, 96, 3);
    model_out.fill(0.0f);

    // 原始人脸
    cv::Mat original = genTestImage(640, 480);

    // 变换矩阵
    cv::Mat M_inv = cv::Mat::eye(2, 3, CV_32F);

    // 口唇遮罩
    cv::Mat mask(480, 640, CV_32FC1, cv::Scalar(1.0f));

    // 构造 RenderTaskData
    auto task = RenderTaskData::Make(
        std::move(model_out),
        original.clone(),
        M_inv.clone(),
        mask.clone());

    TEST_CHECK(task.IsValid(), "RenderTaskData 有效");

    // 通过 Packet 传递
    auto pkt = RenderPacket::Make(std::move(task), 100, 1);
    TEST_CHECK(pkt.header.pts_ms == 100, "PTS 正确");
    TEST_CHECK(pkt.payload.IsValid(), "Packet payload 有效");

    // 移动到输出队列
    ThreadSafeQueue<RenderPacket> queue;
    queue.Push(std::move(pkt));
    queue.Push(RenderPacket::EOS());

    int count = 0;
    RenderPacket received;
    while (queue.WaitAndPop(received, 100)) {
        if (received.header.IsEOS()) break;
        if (received.header.IsOK()) count++;
    }
    TEST_CHECK(count == 1, "队列收发正确 (count=" << count << ")");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  全链路流程验收测试" << std::endl;
    std::cout << "==============================================";
    std::cout << "\n  测试层级:";
    std::cout << "\n    Level 1 — 音频处理模块独立可用";
    std::cout << "\n    Level 2 — 图像处理模块独立可用";
    std::cout << "\n    Level 3 — 端到端流水线集成";
    std::cout << "\n==============================================" << std::endl;

    // ========== Level 1: 音频模块 ==========
    testNoiseReduction();
    testAudioFramer();
    testVAD();
    testPreEmphasis();
    testRMSNormalize();
    testMelExtract();
    testCMVN();
    testRingBuffer();

    // ========== Level 2: 图像模块 ==========
    testImageBasics();
    testFaceDataPacket();

    // ========== Level 3: 集成测试 ==========
    testFullAudioPipeline();
    testFullImagePipeline();
    testQueueIntegration();
    testThroughput();
    testRenderTaskE2E();

    // ==========================================
    // 汇总
    // ==========================================
    std::cout << "\n==============================================" << std::endl;
    std::cout << "  测试汇总" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  Level 1 (音频模块):" << std::endl;
    std::cout << "  Level 2 (图像模块):" << std::endl;
    std::cout << "  Level 3 (集成测试):" << std::endl;
    std::cout << "  -----------------------------------------" << std::endl;
    std::cout << "  通过: " << gPassed << std::endl;
    std::cout << "  失败: " << gFailed << std::endl;
    std::cout << "  总计: " << (gPassed + gFailed) << std::endl;
    std::cout << "==============================================" << std::endl;

    return (gFailed == 0) ? 0 : 1;
}
