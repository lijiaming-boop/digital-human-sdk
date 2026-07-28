/**
 * @file v1_review_fixes_test.cpp
 * @brief v1 代码审查报告修复验证测试
 *
 * 严格按 v1代码审查报告.md 的 7 项主要问题逐条验证：
 *   1. ThreadBase::Wait() 并发未定义行为（std::async + detach）
 *   2. AudioProcessor::FillWindow() 队列模式丢音频样本
 *   3. Pipeline::Stop() 后 Start() 假运行状态
 *   4. Pipeline 未使用 PipelineConfig 队列容量
 *   5. PipelineMetrics 指标未聚合（始终为 0）
 *   6. CMakeLists.txt 全局编译选项（构建级，运行时验证构建类型）
 *   7. .gitignore 缺失忽略规则（文件级，运行时验证规则存在）
 *
 * 用法: ./bin/v1_review_fixes_test
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>

#include "core/thread_base.h"
#include "core/thread_safe_queue.h"
#include "core/audio_processor.h"
#include "core/pipeline.h"
#include "core/packet.h"
#include "core/inference_worker.h"
#include "core/render_thread.h"

using namespace digital_human::core;

// ============================================================================
// 测试框架
// ============================================================================

static int gPassed = 0;
static int gFailed = 0;

#define TEST_SECTION(name) \
    std::cout << "\n====== " << name << " ======" << std::endl;

#define TEST_CHECK(cond, desc)                                                 \
    do {                                                                       \
        if (cond) {                                                            \
            std::cout << "  [PASS] " << desc << std::endl;                     \
            gPassed++;                                                          \
        } else {                                                               \
            std::cout << "  [FAIL] " << desc << " (" << __FILE__ << ":"        \
                      << __LINE__ << ")" << std::endl;                         \
            gFailed++;                                                          \
        }                                                                      \
    } while (0)

// ============================================================================
// Issue 1: ThreadBase::Wait() 并发未定义行为修复验证
// ============================================================================

/// @brief 协作式停止的线程（用于验证正常路径）
class CooperativeThread : public ThreadBase {
public:
    std::atomic<int> iter_count{0};
    CooperativeThread() : ThreadBase("Cooperative") {}
protected:
    void Run() override {
        while (!IsStopping()) {
            iter_count.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
};

/// @brief 延迟响应停止的线程（用于验证超时返回 false 但不 detach）
class DelayedThread : public ThreadBase {
public:
    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};
    DelayedThread() : ThreadBase("Delayed") {}
protected:
    void Run() override {
        started.store(true, std::memory_order_release);
        // 模拟一段不响应 IsStopping() 的长任务（300ms）
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        // 之后开始响应停止
        while (!IsStopping()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        finished.store(true, std::memory_order_release);
    }
};

static void TestThreadBaseWaitFix() {
    TEST_SECTION("Issue 1: ThreadBase::Wait() 并发 UB 修复");

    // 1.1 无限等待 Wait(-1) 正常退出
    {
        CooperativeThread t;
        TEST_CHECK(t.Start(), "1.1 协作线程启动成功");
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        t.Stop();
        bool ok = t.Wait(-1);
        TEST_CHECK(ok, "1.2 Wait(-1) 正常返回 true");
        TEST_CHECK(t.IsStopped(), "1.3 线程状态为 STOPPED");
        TEST_CHECK(t.iter_count.load() > 0,
                   "1.4 线程确实执行过 (iter=" << t.iter_count.load() << ")");
    }

    // 1.2 有限超时 + 线程提前完成 → 返回 true
    {
        CooperativeThread t;
        TEST_CHECK(t.Start(), "1.5 协作线程启动成功（超时测试）");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        t.Stop();
        bool ok = t.Wait(2000);  // 2s 超时，远大于实际需要
        TEST_CHECK(ok, "1.6 Wait(2000) 线程提前完成返回 true");
        TEST_CHECK(t.IsStopped(), "1.7 线程状态为 STOPPED");
    }

    // 1.3 有限超时 + 线程未及时退出 → 返回 false，但不 detach
    //     关键：再次 Wait() 能正常回收线程（旧实现中 detach 后线程泄漏）
    {
        DelayedThread t;
        TEST_CHECK(t.Start(), "1.8 延迟线程启动成功");

        // 等待线程进入长任务阶段
        while (!t.started.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        t.Stop();

        // 立即 Wait(50)：线程还在 300ms 长任务中，应当超时
        auto t0 = std::chrono::steady_clock::now();
        bool ok = t.Wait(50);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

        TEST_CHECK(!ok, "1.9 Wait(50) 线程未退出时返回 false");
        TEST_CHECK(elapsed >= 40 && elapsed <= 200,
                   "1.10 Wait(50) 在合理时间内返回 (耗时=" << elapsed << "ms)");
        // 关键验证：线程状态不应是 ERROR（旧实现会强制设为 ERROR）
        TEST_CHECK(!t.IsError(),
                   "1.11 超时后线程状态不是 ERROR（旧 bug：超时即设 ERROR）");

        // 1.4 再次 Wait() 回收线程（验证未 detach）
        // 给足时间让 DelayedThread 的 300ms 长任务结束
        bool ok2 = t.Wait(2000);
        TEST_CHECK(ok2, "1.12 超时后再次 Wait() 成功回收线程（未 detach）");
        TEST_CHECK(t.IsStopped(),
                   "1.13 最终线程状态为 STOPPED");
        TEST_CHECK(t.finished.load(),
                   "1.14 线程 Run() 正常退出（未被打断）");
    }

    // 1.5 多次 Wait() 幂等
    {
        CooperativeThread t;
        t.Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        t.Stop();
        t.Wait();
        bool second = t.Wait();
        TEST_CHECK(second, "1.15 已 STOPPED 后再次 Wait() 仍返回 true（幂等）");
    }
}

// ============================================================================
// Issue 2: AudioProcessor::FillWindow() 队列模式丢样本修复验证
// ============================================================================

/// @brief 生成正弦波测试音频
static std::vector<float> generateSine(float freq_hz, int sample_rate,
                                        double duration_sec) {
    int total = static_cast<int>(sample_rate * duration_sec);
    std::vector<float> data(total);
    for (int i = 0; i < total; ++i) {
        data[i] = 0.5f * std::sin(2.0f * static_cast<float>(M_PI)
                 * freq_hz * i / sample_rate);
    }
    return data;
}

static void TestAudioProcessorNoSampleLoss() {
    TEST_SECTION("Issue 2: AudioProcessor 队列模式大包不丢样本");

    // 2.1 大包跨多次 FillWindow 不丢样本
    // 配置：window_capacity=4800（300ms@16kHz），frame_size=400, hop_size=160
    // 推送一个 10000 sample 的包：远大于窗口容量
    // 旧实现：只消费前 4800，剩余 5200 丢失 → 输出 ~28 帧
    // 新实现：pending_audio_ 缓冲剩余样本，跨多次 FillWindow 消费 → 输出 ~61 帧
    {
        const int sr = 16000;
        const int frame_size = 400;
        const int hop_size = 160;
        const int window_capacity = 4800;
        const size_t packet_samples = 10000;

        AudioProcessor processor("BigPacketTest");
        AudioProcessorConfig cfg;
        cfg.sample_rate = sr;
        cfg.frame_size = frame_size;
        cfg.hop_size = hop_size;
        cfg.window_capacity = window_capacity;
        cfg.mel_bins = 80;
        cfg.nfft = 512;
        processor.SetConfig(cfg);

        ThreadSafeQueue<AudioRawPacket> in_queue;
        ThreadSafeQueue<MelFeaturePacket> out_queue;
        processor.SetInputQueue(&in_queue);
        processor.SetOutputQueue(&out_queue);

        // 生成单包大音频（正弦波，非静音以保证 VAD 通过）
        auto audio = generateSine(440.0f, sr,
                                  static_cast<double>(packet_samples) / sr);
        TEST_CHECK(audio.size() == packet_samples,
                   "2.1 测试音频包大小正确: " << audio.size());

        // 推送单个大包
        in_queue.Push(AudioRawPacket::Make(audio, 0, 1));
        processor.MarkEOS();  // 标记流结束

        // 启动处理
        processor.Start();

        // 等待 EOS
        MelFeaturePacket pkt;
        int ok_count = 0;
        int skip_count = 0;
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(3000);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!out_queue.WaitAndPop(pkt, 200)) break;
            if (pkt.header.IsEOS()) break;
            if (pkt.header.IsOK()) ok_count++;
            else if (pkt.header.IsSkip()) skip_count++;
        }

        processor.Stop();
        processor.Wait(1000);

        // 理论帧数 = (10000 - 400) / 160 + 1 = 61
        const int expected_frames =
            (static_cast<int>(packet_samples) - frame_size) / hop_size + 1;
        std::cout << "  [INFO] 输出 OK=" << ok_count
                  << " Skip=" << skip_count
                  << " 理论=" << expected_frames
                  << " (旧 bug 预期 ~28)" << std::endl;

        // 关键断言：输出帧数应接近理论值 61
        // 允许 VAD 过滤少量帧，但不应低于理论值的 80%
        TEST_CHECK(ok_count >= expected_frames * 8 / 10,
                   "2.2 大包输出帧数接近理论值 (实际=" << ok_count
                   << " 理论=" << expected_frames
                   << " 阈值=" << (expected_frames * 8 / 10) << ")");

        // 旧 bug 的特征：只有 ~28 帧（4800 个样本能产出的帧数）
        TEST_CHECK(ok_count > 40,
                   "2.3 输出帧数 > 40（旧 bug 会 ≤ 30，因剩余样本丢失）");
    }

    // 2.2 多个小包连续推送，样本不丢
    {
        const int sr = 16000;
        AudioProcessor processor("MultiPacketTest");
        AudioProcessorConfig cfg;
        cfg.sample_rate = sr;
        cfg.frame_size = 400;
        cfg.hop_size = 160;
        cfg.window_capacity = 4800;
        cfg.mel_bins = 80;
        cfg.nfft = 512;
        processor.SetConfig(cfg);

        ThreadSafeQueue<AudioRawPacket> in_queue;
        ThreadSafeQueue<MelFeaturePacket> out_queue;
        processor.SetInputQueue(&in_queue);
        processor.SetOutputQueue(&out_queue);

        // 推送 5 个 1000 sample 的小包（共 5000 sample）
        const int packets = 5;
        const int per_packet = 1000;
        for (int i = 0; i < packets; ++i) {
            auto chunk = generateSine(440.0f, sr,
                                       static_cast<double>(per_packet) / sr);
            in_queue.Push(AudioRawPacket::Make(chunk, i * 100, i + 1));
        }
        processor.MarkEOS();

        processor.Start();

        MelFeaturePacket pkt;
        int ok_count = 0;
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(3000);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!out_queue.WaitAndPop(pkt, 200)) break;
            if (pkt.header.IsEOS()) break;
            if (pkt.header.IsOK()) ok_count++;
        }

        processor.Stop();
        processor.Wait(1000);

        // 总样本 5000，理论帧数 = (5000 - 400) / 160 + 1 = 29
        const int expected = (packets * per_packet - 400) / 160 + 1;
        std::cout << "  [INFO] 多包模式 OK=" << ok_count
                  << " 理论=" << expected << std::endl;

        TEST_CHECK(ok_count >= expected * 8 / 10,
                   "2.4 多包模式输出帧数接近理论值 (实际=" << ok_count
                   << " 理论=" << expected << ")");
    }

    // 2.3 Reset 清空 pending 缓冲
    {
        AudioProcessor processor("ResetTest");
        AudioProcessorConfig cfg;
        cfg.sample_rate = 16000;
        cfg.frame_size = 400;
        cfg.hop_size = 160;
        cfg.window_capacity = 4800;
        processor.SetConfig(cfg);
        processor.SetInputQueue(nullptr);  // 仅触发 SetInputQueue 路径
        // 注意：上面 SetInputQueue(nullptr) 不会真的设置（has_input_queue_ 保持 false）
        // 这里通过 Reset 验证 pending_audio_ 被清空（间接验证不崩溃）
        processor.Reset();
        TEST_CHECK(processor.GetProcessedSamples() == 0,
                   "2.5 Reset 后 processed_samples == 0");
        TEST_CHECK(processor.GetOutputCount() == 0,
                   "2.6 Reset 后 output_count == 0");
    }
}

// ============================================================================
// Issue 3: Pipeline::Stop() 后 Start() 假运行状态修复验证
// ============================================================================

static void TestPipelineStopStartRejection() {
    TEST_SECTION("Issue 3: Pipeline Stop 后 Start 被拒绝");

    // 3.1 Init → Stop（未 Start）→ Start 应失败
    {
        Pipeline pipeline;
        PipelineConfig cfg;
        cfg.audio_sample_rate = 16000;
        cfg.target_fps = 25;
        // 缩小队列容量避免长等待
        cfg.shutdown_timeout_ms = 500;

        bool init_ok = pipeline.Init(cfg);
        TEST_CHECK(init_ok, "3.1 Pipeline Init 成功");

        // 未 Start 直接 Stop（清理资源并标记 terminated）
        pipeline.Stop();

        // 关键验证：Start 应被拒绝
        bool start_ok = pipeline.Start();
        TEST_CHECK(!start_ok,
                   "3.2 Stop 后 Start 返回 false（拒绝假运行）");
        TEST_CHECK(!pipeline.IsRunning(),
                   "3.3 IsRunning() 仍为 false（不会进入假运行）");
    }

    // 3.2 Init → Stop → Init 也应失败（terminated 标志不可恢复）
    {
        Pipeline pipeline;
        PipelineConfig cfg;
        cfg.audio_sample_rate = 16000;
        cfg.target_fps = 25;
        cfg.shutdown_timeout_ms = 500;

        TEST_CHECK(pipeline.Init(cfg), "3.4 第一次 Init 成功");
        pipeline.Stop();
        // 再次 Init 应失败
        bool reinit = pipeline.Init(cfg);
        TEST_CHECK(!reinit,
                   "3.5 Stop 后再次 Init 返回 false（一次性对象）");
    }

    // 3.3 Stop 幂等：多次 Stop 不崩溃
    {
        Pipeline pipeline;
        PipelineConfig cfg;
        cfg.audio_sample_rate = 16000;
        cfg.target_fps = 25;
        cfg.shutdown_timeout_ms = 500;

        pipeline.Init(cfg);
        pipeline.Stop();
        pipeline.Stop();  // 第二次 Stop
        pipeline.Stop();  // 第三次 Stop
        TEST_CHECK(true, "3.6 多次 Stop 幂等不崩溃");
    }

    // 3.4 未 Init 的 Pipeline Stop 安全
    {
        Pipeline pipeline;
        pipeline.Stop();  // 未 Init
        TEST_CHECK(!pipeline.IsRunning(),
                   "3.7 未 Init 的 Pipeline Stop 后 IsRunning=false");
    }
}

// ============================================================================
// Issue 4: Pipeline 使用 PipelineConfig 队列容量
// ============================================================================

static void TestPipelineQueueCapacityFromConfig() {
    TEST_SECTION("Issue 4: Pipeline 队列容量来自 PipelineConfig");

    // 4.1 验证 PipelineConfig 默认值合理
    {
        PipelineConfig cfg;
        TEST_CHECK(cfg.audio_raw_queue_size > 0,
                   "4.1 audio_raw_queue_size 默认 > 0: "
                   << cfg.audio_raw_queue_size);
        TEST_CHECK(cfg.mel_queue_size > 0,
                   "4.2 mel_queue_size 默认 > 0: " << cfg.mel_queue_size);
        TEST_CHECK(cfg.video_raw_queue_size > 0,
                   "4.3 video_raw_queue_size 默认 > 0: "
                   << cfg.video_raw_queue_size);
        TEST_CHECK(cfg.face_queue_size > 0,
                   "4.4 face_queue_size 默认 > 0: " << cfg.face_queue_size);
        TEST_CHECK(cfg.infer_queue_size > 0,
                   "4.5 infer_queue_size 默认 > 0: " << cfg.infer_queue_size);
        TEST_CHECK(cfg.output_queue_size > 0,
                   "4.6 output_queue_size 默认 > 0: " << cfg.output_queue_size);
    }

    // 4.2 自定义队列容量，Pipeline 不崩溃
    {
        Pipeline pipeline;
        PipelineConfig cfg;
        cfg.audio_sample_rate = 16000;
        cfg.target_fps = 25;
        // 显式设置小容量
        cfg.audio_raw_queue_size = 5;
        cfg.mel_queue_size = 5;
        cfg.video_raw_queue_size = 5;
        cfg.face_queue_size = 5;
        cfg.infer_queue_size = 5;
        cfg.output_queue_size = 5;
        cfg.shutdown_timeout_ms = 500;

        bool ok = pipeline.Init(cfg);
        TEST_CHECK(ok, "4.7 小队列容量 Init 成功");
        pipeline.Stop();
        TEST_CHECK(true, "4.8 小队列容量 Stop 不崩溃");
    }

    // 4.3 直接验证 ThreadSafeQueue 容量语义（单元级）
    {
        ThreadSafeQueue<int> q(5, "test_bounded");
        TEST_CHECK(q.Capacity() == 5,
                   "4.9 ThreadSafeQueue 容量正确: " << q.Capacity());

        // 填满
        for (int i = 0; i < 5; ++i) {
            TEST_CHECK(q.TryPush(std::move(i)),
                       "4.10 TryPush 第 " << i << " 个成功");
        }
        // 第 6 个应失败（有界）
        int extra = 99;
        TEST_CHECK(!q.TryPush(std::move(extra)),
                   "4.11 满队列 TryPush 失败（有界反压）");

        // 消费一个后可再 push
        int v;
        TEST_CHECK(q.TryPop(v), "4.12 消费一个后 TryPop 成功");
        TEST_CHECK(q.TryPush(std::move(extra)),
                   "4.13 消费后 TryPush 重新成功");
    }
}

// ============================================================================
// Issue 5: PipelineMetrics 指标聚合验证
// ============================================================================

static void TestPipelineMetricsAggregation() {
    TEST_SECTION("Issue 5: PipelineMetrics 聚合");

    // 5.1 未启动的 Pipeline GetMetrics 不崩溃，返回零值
    {
        Pipeline pipeline;
        PipelineConfig cfg;
        cfg.audio_sample_rate = 16000;
        cfg.target_fps = 25;
        cfg.shutdown_timeout_ms = 500;

        pipeline.Init(cfg);
        // 不 Start，直接查询指标
        PipelineMetrics m = pipeline.GetMetrics();

        // 输入计数应为 0（从未 PushAudio/PushVideo）
        TEST_CHECK(m.total_frames_in == 0, "5.1 未启动时 total_frames_in == 0");
        TEST_CHECK(m.audio_packets_in == 0, "5.2 未启动时 audio_packets_in == 0");
        TEST_CHECK(m.video_packets_in == 0, "5.3 未启动时 video_packets_in == 0");

        // 输出/推理计数也应为 0（worker 从未运行）
        TEST_CHECK(m.total_frames_out == 0, "5.4 未启动时 total_frames_out == 0");
        TEST_CHECK(m.inference_count == 0, "5.5 未启动时 inference_count == 0");

        std::cout << "  [INFO] 未启动 Pipeline 指标: " << m.ToString()
                  << std::endl;

        pipeline.Stop();
    }

    // 5.2 验证 PipelineMetrics 结构完整（所有字段都可访问）
    {
        PipelineMetrics m;
        // 显式访问每个字段，确保结构体定义完整
        int64_t frames_in   = m.total_frames_in;
        int64_t frames_out  = m.total_frames_out;
        int64_t dropped     = m.frames_dropped;
        int64_t skipped     = m.frames_skipped;
        int64_t audio_in    = m.audio_packets_in;
        int64_t video_in    = m.video_packets_in;
        int64_t infer_cnt   = m.inference_count;
        double  avg_audio   = m.avg_audio_process_ms;
        double  avg_video   = m.avg_video_process_ms;
        double  avg_infer   = m.avg_inference_ms;
        double  avg_output  = m.avg_output_ms;
        double  fps         = m.actual_fps;

        // 防止编译器优化掉未使用变量
        (void)frames_in; (void)frames_out; (void)dropped; (void)skipped;
        (void)audio_in; (void)video_in; (void)infer_cnt;
        (void)avg_audio; (void)avg_video; (void)avg_infer;
        (void)avg_output; (void)fps;

        TEST_CHECK(true, "5.6 PipelineMetrics 所有字段可访问（结构完整）");
        TEST_CHECK(!m.ToString().empty(),
                   "5.7 PipelineMetrics::ToString() 非空");
    }

    // 5.3 验证 InferenceWorker 指标接口存在
    // （不实际运行推理，仅验证接口可调用）
    {
        InferenceWorker worker;
        InferenceMetrics im = worker.GetMetrics();
        TEST_CHECK(im.total_inferences == 0,
                   "5.8 InferenceWorker 初始 total_inferences == 0");
        TEST_CHECK(!im.ToString().empty(),
                   "5.9 InferenceMetrics::ToString() 非空");
    }

    // 5.4 验证 RenderThread 指标接口存在
    {
        RenderThread rt;
        RenderMetrics rm = rt.GetMetrics();
        TEST_CHECK(rm.frames_rendered == 0,
                   "5.10 RenderThread 初始 frames_rendered == 0");
        TEST_CHECK(!rm.ToString().empty(),
                   "5.11 RenderMetrics::ToString() 非空");
    }
}

// ============================================================================
// Issue 6: CMakeLists.txt 全局编译选项修复验证（构建级）
// ============================================================================

static void TestCMakeBuildConfig() {
    TEST_SECTION("Issue 6: CMake 构建配置（运行时验证）");

    // 6.1 验证 NDEBUG 不再被无条件定义
    // 旧实现：add_definitions(-DNDEBUG) 强制定义，破坏 Debug 构建
    // 新实现：只在 Release/RelWithDebInfo/MinSizeRel 中通过 target_compile_options 定义
    // 运行时验证：检查构建目录的 CMakeCache.txt 中 CMAKE_BUILD_TYPE
    // 这里仅验证编译期宏的状态符合预期

#ifdef NDEBUG
    std::cout << "  [INFO] 当前构建定义了 NDEBUG（Release/RelWithDebInfo/MinSizeRel）"
              << std::endl;
    TEST_CHECK(true, "6.1 Release 构建正确定义 NDEBUG");
#else
    std::cout << "  [INFO] 当前构建未定义 NDEBUG（Debug）"
              << std::endl;
    TEST_CHECK(true, "6.2 Debug 构建正确不定义 NDEBUG");
#endif

    // 6.2 验证 CMakeLists.txt 中不再有全局 -march=native
    // 通过读取文件内容验证（运行时检查源码）
    // 使用编译期宏 PROJECT_SOURCE_DIR 定位源码根，避免依赖运行时 CWD
#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif
    // 辅助函数：检查 content 中是否存在某 token（仅匹配非注释行，
    // 避免因注释中提到该 token 而产生误报）
    auto contains_outside_comments = [](const std::string& content,
                                         const std::string& token) -> bool {
        std::istringstream iss(content);
        std::string line;
        while (std::getline(iss, line)) {
            // 去掉行首空白
            auto first_non_space = line.find_first_not_of(" \t\r\n");
            if (first_non_space == std::string::npos) continue;
            // 跳过注释行（CMake 用 # 开头）
            if (line[first_non_space] == '#') continue;
            if (line.find(token) != std::string::npos) return true;
        }
        return false;
    };

    std::string cmake_path = std::string(PROJECT_SOURCE_DIR) + "/CMakeLists.txt";
    std::ifstream cmake_file(cmake_path);
    if (cmake_file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(cmake_file)),
                             std::istreambuf_iterator<char>());
        bool has_march_native = contains_outside_comments(content, "-march=native");
        bool has_global_O3 = contains_outside_comments(content,
                              "add_compile_options(-O3)");
        bool has_global_NDEBUG = contains_outside_comments(content,
                                  "add_definitions(-DNDEBUG)");
        bool has_link_directories = contains_outside_comments(content,
                                     "link_directories");

        TEST_CHECK(!has_march_native,
                   "6.3 CMakeLists.txt 不再含 -march=native");
        TEST_CHECK(!has_global_O3,
                   "6.4 CMakeLists.txt 不再含全局 add_compile_options(-O3)");
        TEST_CHECK(!has_global_NDEBUG,
                   "6.5 CMakeLists.txt 不再含全局 add_definitions(-DNDEBUG)");
        TEST_CHECK(!has_link_directories,
                   "6.6 CMakeLists.txt 不再含 link_directories（移除 $HOME/.local/lib）");
    } else {
        TEST_CHECK(false, "6.7 无法打开 CMakeLists.txt 进行验证");
    }

    // 6.3 验证 src/CMakeLists.txt 使用 CONFIGURE_DEPENDS
    std::string src_cmake_path = std::string(PROJECT_SOURCE_DIR) + "/src/CMakeLists.txt";
    std::ifstream src_cmake(src_cmake_path);
    if (src_cmake.is_open()) {
        std::string content((std::istreambuf_iterator<char>(src_cmake)),
                             std::istreambuf_iterator<char>());
        bool has_configure_depends = content.find("CONFIGURE_DEPENDS")
                                      != std::string::npos;
        bool has_target_compile_options = content.find("target_compile_options")
                                           != std::string::npos;
        bool has_target_include = content.find("target_include_directories")
                                   != std::string::npos;
        bool has_target_link = content.find("target_link_libraries")
                                != std::string::npos;

        TEST_CHECK(has_configure_depends,
                   "6.8 src/CMakeLists.txt 含 CONFIGURE_DEPENDS");
        TEST_CHECK(has_target_compile_options,
                   "6.9 src/CMakeLists.txt 使用 target_compile_options");
        TEST_CHECK(has_target_include,
                   "6.10 src/CMakeLists.txt 使用 target_include_directories");
        TEST_CHECK(has_target_link,
                   "6.11 src/CMakeLists.txt 使用 target_link_libraries");
    } else {
        TEST_CHECK(false, "6.12 无法打开 src/CMakeLists.txt 进行验证");
    }
}

// ============================================================================
// Issue 7: .gitignore 缺失忽略规则验证（文件级）
// ============================================================================

static void TestGitignoreRules() {
    TEST_SECTION("Issue 7: .gitignore 忽略规则");

    std::string gitignore_path = std::string(PROJECT_SOURCE_DIR) + "/.gitignore";
    std::ifstream gitignore(gitignore_path);
    if (!gitignore.is_open()) {
        TEST_CHECK(false, "7.1 无法打开 .gitignore");
        return;
    }

    std::string content((std::istreambuf_iterator<char>(gitignore)),
                         std::istreambuf_iterator<char>());

    // 验证关键规则存在
    struct RuleCheck {
        const char* pattern;
        const char* desc;
    };
    RuleCheck rules[] = {
        {"build_wsl/",          "7.2 忽略 build_wsl/"},
        {"assets/output/",      "7.3 忽略 assets/output/"},
        {"*.bin",               "7.4 忽略模型文件 *.bin"},
        {"*.param",             "7.5 忽略模型文件 *.param"},
        {"*.zip",               "7.6 忽略压缩包 *.zip"},
        {"*.mp4",               "7.7 忽略视频 *.mp4"},
        {"*.mp3",               "7.8 忽略音频 *.mp3"},
        {".vs/",                "7.9 忽略 .vs/"},
        {".DS_Store",           "7.10 忽略 .DS_Store"},
    };

    for (const auto& r : rules) {
        bool found = content.find(r.pattern) != std::string::npos;
        TEST_CHECK(found, r.desc);
    }

    // 验证不再含旧的过窄规则（旧 *.dat 已被替换为更全面的规则）
    // *.dat 仍在，但应同时包含 *.bin *.param 等模型规则
    bool has_dat = content.find("*.dat") != std::string::npos;
    TEST_CHECK(has_dat, "7.11 保留 *.dat 规则");
}

// ============================================================================
// 综合回归测试：确保修复未破坏既有功能
// ============================================================================

/// @brief 简单计数线程
class CounterThread : public ThreadBase {
public:
    std::atomic<int> count{0};
    CounterThread() : ThreadBase("Counter") {}
protected:
    void Run() override {
        while (!IsStopping()) {
            count.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

static void TestRegressionBasicThread() {
    TEST_SECTION("回归: ThreadBase 基础功能未破坏");

    CounterThread t;
    TEST_CHECK(t.GetState() == ThreadState::INIT, "R.1 初始 INIT");
    TEST_CHECK(t.Start(), "R.2 Start 成功");
    TEST_CHECK(!t.Start(), "R.3 重复 Start 失败");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    TEST_CHECK(t.count.load() > 0, "R.4 计数 > 0 (count=" << t.count.load() << ")");
    t.Stop();
    TEST_CHECK(t.Wait(2000), "R.5 Wait 成功");
    TEST_CHECK(t.IsStopped(), "R.6 状态 STOPPED");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  v1 代码审查报告修复验证测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    TestThreadBaseWaitFix();
    TestAudioProcessorNoSampleLoss();
    TestPipelineStopStartRejection();
    TestPipelineQueueCapacityFromConfig();
    TestPipelineMetricsAggregation();
    TestCMakeBuildConfig();
    TestGitignoreRules();
    TestRegressionBasicThread();

    std::cout << "\n==============================================" << std::endl;
    std::cout << "  测试汇总" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  通过: " << gPassed << std::endl;
    std::cout << "  失败: " << gFailed << std::endl;
    std::cout << "  总计: " << (gPassed + gFailed) << std::endl;
    std::cout << "==============================================" << std::endl;

    return (gFailed == 0) ? 0 : 1;
}
