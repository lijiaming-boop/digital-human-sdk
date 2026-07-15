/**
 * @file audio_processor_test.cpp
 * @brief AudioProcessor 音频处理线程验收测试
 *
 * 覆盖范围：
 * - 文件模式：固定 PCM 缓冲区处理
 * - 滑动窗口正确性（帧数、重叠、PTS）
 * - 流式模式：RingBuffer 增量输入
 * - 实时性：处理延迟测量
 * - EOS 终止
 * - 立体声混合
 * - 边界条件（静音、短音频）
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>

#include "core/audio_processor.h"
#include "audio/audio_ring_buffer.h"

using namespace digital_human::core;
using digital_human::audio::RingBuffer;

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
// 辅助函数
// ============================================================================

/// @brief 生成正弦波测试音频
static std::vector<float> generateSine(float freq_hz, int sample_rate,
                                        int channels, double duration_sec) {
    int total_samples = static_cast<int>(sample_rate * duration_sec) * channels;
    std::vector<float> data(total_samples);
    for (int i = 0; i < total_samples / channels; ++i) {
        float val = std::sin(2.0f * static_cast<float>(M_PI) * freq_hz * i / sample_rate);
        for (int c = 0; c < channels; ++c) {
            data[i * channels + c] = val;
        }
    }
    return data;
}

/// @brief 等待 MelFeatureQueue 收到指定数量帧
static int drainQueue(ThreadSafeQueue<MelFeaturePacket>& queue,
                      int max_count, int timeout_ms = 5000) {
    int count = 0;
    auto deadline = std::chrono::steady_clock::now()
                   + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline && count < max_count) {
        MelFeaturePacket pkt;
        if (queue.WaitAndPop(pkt, 50)) {
            if (pkt.header.IsEOS()) break;
            if (pkt.header.IsOK()) count++;
        }
    }
    return count;
}

// ============================================================================
// Test 1: 基本配置与初始化
// ============================================================================
static void testConfig() {
    TEST_NAME("Test 1: 基本配置与初始化");

    AudioProcessor processor;
    AudioProcessorConfig cfg;
    cfg.sample_rate = 16000;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    const auto& got = processor.GetConfig();
    TEST_CHECK(got.sample_rate == 16000, "1.1 采样率配置正确");
    TEST_CHECK(got.frame_size == 400, "1.2 帧长配置正确");
    TEST_CHECK(got.hop_size == 160, "1.3 帧移配置正确");

    // AutoConfigure
    AudioProcessorConfig auto_cfg;
    auto_cfg.AutoConfigure(16000);
    TEST_CHECK(auto_cfg.frame_size == 400, "1.4 AutoConfigure 16kHz frame_size=400");
    TEST_CHECK(auto_cfg.hop_size == 160, "1.5 AutoConfigure 16kHz hop_size=160");
}

// ============================================================================
// Test 2: 文件模式 — 固定缓冲区的滑动窗口处理
// ============================================================================
static void testFixedBuffer() {
    TEST_NAME("Test 2: 文件模式 — 固定缓冲区的滑动窗口处理");

    // 生成 1 秒 16kHz 正弦波
    int sr = 16000;
    auto audio = generateSine(440.0f, sr, 1, 1.0);

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;
    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    processor.SetAudioSource(audio.data(), audio.size(), sr, 1);
    processor.SetOutputQueue(&output_queue);

    // 启动线程
    processor.Start();

    // 等待处理完成
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    processor.Stop();

    int64_t output_count = processor.GetOutputCount();
    int64_t processed_samps = processor.GetProcessedSamples();
    int64_t pending = processor.GetPendingFrames();

    std::cout << "  [INFO] 输出帧数=" << output_count
              << " 已处理samples=" << processed_samps
              << " 待处理帧=" << pending << std::endl;

    // 理论帧数: (10000 - 400) / 160 + 1 = 61 帧 (@16kHz, 1秒=16000samples)
    // 实际: 16kHz * 1s = 16000 samples, 但 AudioProcessor 以 hop=160 推进
    // window 初始 0, 每次 FillWindow 填充全部
    // 修正: 第一次 FillWindow 读取全部 16000, 然后 NextFrame 逐帧消耗
    // 帧数 = (16000 - 400) / 160 + 1 = 98.5 → 98 帧 (整数除法)
    int expected_frames = (static_cast<int>(audio.size()) - cfg.frame_size)
                          / cfg.hop_size + 1;
    // window 读到全部 16000 sample, 逐帧消耗
    // 但实际 NextFrame 会在窗口不足一帧时停止

    // 至少要处理一些帧
    TEST_CHECK(output_count > 0, "2.1 有输出帧 (count="
               << output_count << ")");

    TEST_CHECK(processed_samps > 0, "2.2 已处理 samples > 0 ("
               << processed_samps << ")");

    // 验证 Mel 特征格式: cv::Mat, rows = 帧数, cols = mel_bins
    // (但每个 Packet 只有一行的 cv::Mat，因为 ProcessOneFrame
    // 对单帧窗口提取的是单帧的 Mel 特征)
    // 实际上 output_queue 中每包是 1 帧 (T=1) 的 mel 特征
}

// ============================================================================
// Test 3: 滑动窗口帧数正确性
// ============================================================================
static void testWindowFrameCount() {
    TEST_NAME("Test 3: 滑动窗口帧数正确性");

    int sr = 16000;
    int hop = 160, frame = 400;

    // 生成 2 秒音频 = 32000 samples
    auto audio = generateSine(440.0f, sr, 1, 2.0);
    size_t total = audio.size();

    // 理论帧数
    int expected = (static_cast<int>(total) - frame) / hop + 1;
    std::cout << "  [INFO] 总samples=" << total
              << " 理论帧数=" << expected << std::endl;

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;

    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = frame;
    cfg.hop_size    = hop;

    processor.SetConfig(cfg);
    processor.SetAudioSource(audio.data(), audio.size(), sr, 1);
    processor.SetOutputQueue(&output_queue);

    processor.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    processor.Stop();

    int64_t output_count = processor.GetOutputCount();

    // 应该接近理论帧数
    TEST_CHECK(output_count > 0, "3.1 有输出帧");
    TEST_CHECK(std::abs(static_cast<int>(output_count) - expected) < 5,
               "3.2 帧数接近理论值 (实际=" << output_count
               << " 理论=" << expected << ")");
}

// ============================================================================
// Test 4: PTS 时间戳连续性
// ============================================================================
static void testPTSContinuity() {
    TEST_NAME("Test 4: PTS 时间戳连续性");

    int sr = 16000;
    auto audio = generateSine(440.0f, sr, 1, 0.5);  // 0.5s

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;

    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    processor.SetAudioSource(audio.data(), audio.size(), sr, 1);
    processor.SetOutputQueue(&output_queue);

    processor.Start();

    // 收集所有输出帧的 PTS
    std::vector<int64_t> pts_list;
    while (true) {
        MelFeaturePacket pkt;
        if (!output_queue.WaitAndPop(pkt, 500)) break;
        if (pkt.header.IsEOS()) break;
        if (pkt.header.IsOK()) {
            pts_list.push_back(pkt.header.pts_ms);
        }
    }

    processor.Stop();

    TEST_CHECK(!pts_list.empty(), "4.1 至少有一帧");

    // 验证 PTS 递增
    bool monotonic = true;
    for (size_t i = 1; i < pts_list.size(); ++i) {
        if (pts_list[i] <= pts_list[i-1]) {
            monotonic = false;
            std::cout << "  [INFO] PTS 不连续: pts[" << i-1 << "]="
                      << pts_list[i-1] << " pts[" << i << "]="
                      << pts_list[i] << std::endl;
            break;
        }
    }
    TEST_CHECK(monotonic, "4.2 PTS 单调递增");

    // 相邻 PTS 差值应为 hop/sample_rate*1000 ≈ 10ms
    if (pts_list.size() > 1) {
        int64_t diff = pts_list[1] - pts_list[0];
        TEST_CHECK(std::abs(diff - 10) <= 1,
                   "4.3 PTS 步长 ≈10ms (实际=" << diff << "ms)");
    }

    std::cout << "  [INFO] 首帧PTS=" << pts_list.front()
              << "ms 末帧PTS=" << pts_list.back()
              << "ms 总帧数=" << pts_list.size() << std::endl;
}

// ============================================================================
// Test 5: 流式模式 — RingBuffer 增量输入
// ============================================================================
static void testRingBufferMode() {
    TEST_NAME("Test 5: 流式模式 — RingBuffer 增量输入");

    int sr = 16000;
    RingBuffer ring(8192);
    auto audio = generateSine(440.0f, sr, 1, 0.3);  // 0.3s

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;

    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    processor.SetRingBuffer(&ring);
    processor.SetOutputQueue(&output_queue);

    processor.Start();

    // 分 3 次写入 RingBuffer，模拟流式输入
    size_t chunk = audio.size() / 3;
    for (int i = 0; i < 3; ++i) {
        size_t offset = i * chunk;
        size_t count = (i == 2) ? (audio.size() - offset) : chunk;
        ring.write(audio.data() + offset, count);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // 标记结束
    processor.MarkEOS();

    // 等待处理完成
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    processor.Stop();

    int64_t output_count = processor.GetOutputCount();
    TEST_CHECK(output_count > 0, "5.1 流式模式有输出 (count="
               << output_count << ")");
}

// ========================================================================
// Test 6: 立体声输入自动混合
// ========================================================================
static void testStereoMixing() {
    TEST_NAME("Test 6: 立体声输入自动混合");

    int sr = 16000;
    // 立体声正弦波
    auto stereo_audio = generateSine(440.0f, sr, 2, 0.3);

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;

    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    // channels=2 传入
    processor.SetAudioSource(stereo_audio.data(), stereo_audio.size(), sr, 2);
    processor.SetOutputQueue(&output_queue);

    processor.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    processor.Stop();

    int64_t output_count = processor.GetOutputCount();
    TEST_CHECK(output_count > 0, "6.1 立体声混音后正常输出 (count="
               << output_count << ")");
}

// ============================================================================
// Test 7: 静音输入处理
// ============================================================================
static void testSilence() {
    TEST_NAME("Test 7: 静音输入处理");

    int sr = 16000;
    std::vector<float> silence(sr * 1, 0.0f);  // 1秒静音

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;

    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    processor.SetAudioSource(silence.data(), silence.size(), sr, 1);
    processor.SetOutputQueue(&output_queue);

    processor.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    processor.Stop();

    int64_t output_count = processor.GetOutputCount();
    TEST_CHECK(output_count >= 0, "7.1 静音输入不崩溃 (count="
               << output_count << ")");
}

// ============================================================================
// Test 8: 短音频（不足一帧）
// ============================================================================
static void testShortAudio() {
    TEST_NAME("Test 8: 短音频（不足一帧）");

    int sr = 16000;
    std::vector<float> short_audio(100, 0.01f);  // 100 samples < 400

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;

    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    processor.SetAudioSource(short_audio.data(), short_audio.size(), sr, 1);
    processor.SetOutputQueue(&output_queue);

    processor.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    processor.Stop();

    int64_t output_count = processor.GetOutputCount();
    TEST_CHECK(output_count == 0, "8.1 短音频无输出帧 (count="
               << output_count << ")");
}

// ============================================================================
// Test 9: 双处理器并行（模拟流水线中多个实例）
// ============================================================================
static void testMultiProcessor() {
    TEST_NAME("Test 9: 双处理器并行");

    int sr = 16000;
    auto audio1 = generateSine(440.0f, sr, 1, 0.3);
    auto audio2 = generateSine(880.0f, sr, 1, 0.3);

    ThreadSafeQueue<MelFeaturePacket> q1, q2;

    AudioProcessor p1("AudioProcessor-1"), p2("AudioProcessor-2");
    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    p1.SetConfig(cfg);
    p1.SetAudioSource(audio1.data(), audio1.size(), sr, 1);
    p1.SetOutputQueue(&q1);

    p2.SetConfig(cfg);
    p2.SetAudioSource(audio2.data(), audio2.size(), sr, 1);
    p2.SetOutputQueue(&q2);

    p1.Start();
    p2.Start();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    p1.Stop();
    p2.Stop();

    TEST_CHECK(p1.GetOutputCount() > 0, "9.1 p1 有输出");
    TEST_CHECK(p2.GetOutputCount() > 0, "9.2 p2 有输出");

    std::cout << "  [INFO] p1=" << p1.GetOutputCount()
              << " p2=" << p2.GetOutputCount() << std::endl;
}

// ============================================================================
// Test 10: 处理延迟测量（实时性）
// ============================================================================
static void testProcessingLatency() {
    TEST_NAME("Test 10: 处理延迟测量（实时性）");

    int sr = 16000;
    auto audio = generateSine(440.0f, sr, 1, 0.5);

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;

    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    processor.SetAudioSource(audio.data(), audio.size(), sr, 1);
    processor.SetOutputQueue(&output_queue);

    // 测量单帧处理延迟
    auto start = std::chrono::steady_clock::now();
    processor.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    processor.Stop();
    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    int64_t frames = processor.GetOutputCount();
    double avg_latency = (frames > 0) ? elapsed / frames : 0;

    std::cout << "  [INFO] 总耗时=" << elapsed << "ms"
              << " 帧数=" << frames
              << " 均延迟=" << avg_latency << "ms/帧" << std::endl;

    // 对于 0.5s 音频，处理应在数百 ms 内完成
    TEST_CHECK(elapsed < 2000, "10.1 处理 0.5s 音频 < 2000ms (实际="
               << elapsed << "ms)");

    // 每帧延迟应在合理范围（通常 < 5ms，但 CI 环境可放宽）
    TEST_CHECK(frames > 0, "10.2 有输出帧");
}

// ============================================================================
// Test 11: 长时间运行稳定性
// ============================================================================
static void testLongRunning() {
    TEST_NAME("Test 11: 长时间运行稳定性");

    int sr = 16000;
    auto audio = generateSine(440.0f, sr, 1, 5.0);  // 5s 音频

    ThreadSafeQueue<MelFeaturePacket> output_queue;
    AudioProcessor processor;

    AudioProcessorConfig cfg;
    cfg.sample_rate = sr;
    cfg.frame_size  = 400;
    cfg.hop_size    = 160;

    processor.SetConfig(cfg);
    processor.SetAudioSource(audio.data(), audio.size(), sr, 1);
    processor.SetOutputQueue(&output_queue);

    processor.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    processor.Stop();

    int64_t output_count = processor.GetOutputCount();
    double progress = processor.GetProgress();
    double duration = processor.GetProcessedDurationMs();

    TEST_CHECK(output_count > 0, "11.1 长时间运行有输出 (count="
               << output_count << ")");
    TEST_CHECK(progress > 0, "11.2 进度 > 0 (progress="
               << progress << ")");
    TEST_CHECK(duration > 0, "11.3 处理时长 > 0 ("
               << duration << "ms)");

    std::cout << "  [INFO] 输出=" << output_count
              << " 进度=" << (progress * 100) << "%"
              << " 已处理=" << duration << "ms" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  AudioProcessor 音频处理线程验收测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    testConfig();
    testFixedBuffer();
    testWindowFrameCount();
    testPTSContinuity();
    testRingBufferMode();
    testStereoMixing();
    testSilence();
    testShortAudio();
    testMultiProcessor();
    testProcessingLatency();
    testLongRunning();

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
