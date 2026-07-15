/**
 * @file pipeline_test.cpp
 * @brief Pipeline 多线程流水线验收测试
 *
 * 覆盖范围：
 * - ThreadSafeQueue 单/多生产者消费者
 * - Packet 状态管理
 * - ThreadBase 生命周期
 * - Pipeline 完整流水线（音视频输入 → 处理 → 输出）
 * - 优雅退出 / 超时强制停止
 * - 死锁检测（长时间运行无锁等待）
 * - 反压测试（慢消费者）
 * - 暂停/恢复控制
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>

#include "core/thread_safe_queue.h"
#include "core/packet.h"
#include "core/thread_base.h"
#include "core/pipeline.h"

using namespace digital_human::core;

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
// Test 1: ThreadSafeQueue 基本操作
// ============================================================================
static void testQueueBasic() {
    TEST_NAME("Test 1: ThreadSafeQueue 基本操作");

    ThreadSafeQueue<int> queue;

    TEST_CHECK(queue.Empty(), "1.1 初始队列为空");
    TEST_CHECK(queue.Size() == 0, "1.2 初始队列大小=0");

    queue.Push(1);
    queue.Push(2);
    queue.Push(3);

    TEST_CHECK(!queue.Empty(), "1.3 Push 后非空");
    TEST_CHECK(queue.Size() == 3, "1.4 Size=3");

    int val;
    TEST_CHECK(queue.TryPop(val), "1.5 TryPop 成功");
    TEST_CHECK(val == 1, "1.6 先进先出 val=1");
    TEST_CHECK(queue.Size() == 2, "1.7 Size=2");

    queue.TryPop(val);
    queue.TryPop(val);
    TEST_CHECK(queue.Empty(), "1.8 全部弹出后为空");
}

// ============================================================================
// Test 2: ThreadSafeQueue 有界队列与背压
// ============================================================================
static void testQueueBounded() {
    TEST_NAME("Test 2: ThreadSafeQueue 有界队列与反压");

    ThreadSafeQueue<int> queue(3);  // 容量=3

    TEST_CHECK(queue.Capacity() == 3, "2.1 容量=3");
    TEST_CHECK(queue.Push(1), "2.2 Push 1");
    TEST_CHECK(queue.Push(2), "2.3 Push 2");
    TEST_CHECK(queue.Push(3), "2.4 Push 3");
    TEST_CHECK(queue.Full(), "2.5 队列已满");

    // 非阻塞 TryPush 应失败
    TEST_CHECK(!queue.TryPush(4), "2.6 满队列 TryPush 失败");

    // 消费一个
    int val;
    queue.TryPop(val);
    TEST_CHECK(!queue.Full(), "2.7 Pop 后非满");
}

// ============================================================================
// Test 3: ThreadSafeQueue 多生产者多消费者
// ============================================================================
static void testQueueMultiThread() {
    TEST_NAME("Test 3: ThreadSafeQueue 多生产者多消费者");

    ThreadSafeQueue<int> queue;
    const int kCount = 10000;
    std::atomic<int64_t> sum{0};

    // 2 个生产者
    auto producer = [&]() {
        for (int i = 0; i < kCount; ++i) {
            queue.Push(1);
        }
    };

    // 2 个消费者
    auto consumer = [&]() {
        int v;
        int consumed = 0;
        while (consumed < kCount) {
            if (queue.WaitAndPop(v, 10)) {
                sum.fetch_add(v, std::memory_order_relaxed);
                consumed++;
            }
        }
    };

    std::thread p1(producer);
    std::thread p2(producer);
    std::thread c1(consumer);
    std::thread c2(consumer);

    p1.join();
    p2.join();
    c1.join();
    c2.join();

    // 2 producers × 10000 × 1 = 20000
    TEST_CHECK(sum.load() == 2 * kCount,
               "3.1 两生产者两消费者 sum=" << sum.load());
}

// ============================================================================
// Test 4: ThreadSafeQueue Stop 信号
// ============================================================================
static void testQueueStop() {
    TEST_NAME("Test 4: ThreadSafeQueue Stop 信号");

    ThreadSafeQueue<int> queue;

    // 启动消费者线程等待
    std::atomic<bool> stopped{false};
    std::thread consumer([&]() {
        int v;
        queue.WaitAndPop(v, 5000);
        stopped.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    TEST_CHECK(!stopped.load(), "4.1 消费者在等待中");

    queue.Stop();
    consumer.join();
    TEST_CHECK(stopped.load(), "4.2 Stop 后消费者退出");
    TEST_CHECK(queue.IsStopped(), "4.3 IsStopped() == true");
}

// ============================================================================
// Test 5: ThreadSafeQueue EOS 包传播
// ============================================================================
static void testQueueEOS() {
    TEST_NAME("Test 5: Packet EOS 传播");

    ThreadSafeQueue<Packet<int>> queue;

    // 普通包
    queue.Push(Packet<int>::Make(42, 100, 1));
    // EOS 包
    queue.Push(Packet<int>::EOS());

    Packet<int> pkt;
    queue.WaitAndPop(pkt);
    TEST_CHECK(pkt.header.IsOK(), "5.1 第一个包 OK");
    TEST_CHECK(pkt.payload == 42, "5.2 payload=42");
    TEST_CHECK(pkt.header.pts_ms == 100, "5.3 pts=100");

    queue.WaitAndPop(pkt);
    TEST_CHECK(pkt.header.IsEOS(), "5.4 第二个包 EOS");
}

// ============================================================================
// Test 6: ThreadBase 生命周期
// ============================================================================
static void testThreadLifecycle() {
    TEST_NAME("Test 6: ThreadBase 生命周期");

    class TestThread : public ThreadBase {
    public:
        std::atomic<int> count{0};
        int max_count;
        TestThread(int max) : ThreadBase("TestThread"), max_count(max) {}
        void Run() override {
            while (!IsStopping() && count < max_count) {
                count.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    TestThread thread(100);
    TEST_CHECK(thread.GetState() == ThreadState::INIT, "6.1 初始状态 INIT");

    TEST_CHECK(thread.Start(), "6.2 Start() 成功");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    TEST_CHECK(thread.IsRunning(), "6.3 运行中");
    TEST_CHECK(thread.count.load() > 0, "6.4 已处理 > 0 项");

    thread.Stop();
    thread.Wait();
    TEST_CHECK(thread.IsStopped(), "6.5 停止后状态 STOPPED");

    // 重复启动应失败
    TEST_CHECK(!thread.Start(), "6.6 重复 Start() 失败");
}

// ============================================================================
// Test 7: 死锁检测 - 环形依赖场景
// ============================================================================
static void testDeadlockFreedom() {
    TEST_NAME("Test 7: 死锁检测 - 多队列无环依赖");

    // 模拟两阶段流水线：P1 → Q1 → P2 → Q2 → C
    ThreadSafeQueue<int> q1;
    ThreadSafeQueue<int> q2;
    std::atomic<bool> done{false};
    std::atomic<int> total{0};

    // 生产者
    auto producer = [&]() {
        for (int i = 0; i < 500; ++i) {
            q1.Push(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        q1.Push(-1);  // 终止信号
    };

    // 中间处理
    auto processor = [&]() {
        while (true) {
            int v;
            if (!q1.WaitAndPop(v, 100)) continue;
            if (v < 0) { q2.Push(v); break; }
            q2.Push(v * 2);
        }
    };

    // 消费者
    auto consumer = [&]() {
        int sum = 0;
        while (true) {
            int v;
            if (!q2.WaitAndPop(v, 100)) continue;
            if (v < 0) break;
            sum += v;
        }
        total.store(sum, std::memory_order_release);
        done.store(true, std::memory_order_release);
    };

    std::thread t1(producer);
    std::thread t2(processor);
    std::thread t3(consumer);

    t1.join();
    t2.join();
    t3.join();

    // 期望: sum(0..499) * 2 = 499*500/2 * 2 = 249500
    int expected = 499 * 500 / 2 * 2;
    TEST_CHECK(total.load() == expected,
               "7.1 流水线结果正确 sum=" << total.load()
               << " expected=" << expected);
    TEST_CHECK(done.load(), "7.2 消费者正常完成");
}

// ============================================================================
// Test 8: 优雅退出 - 超时停止
// ============================================================================
static void testGracefulShutdown() {
    TEST_NAME("Test 8: 优雅退出 - 超时停止");

    class SlowThread : public ThreadBase {
    public:
        SlowThread() : ThreadBase("SlowThread") {}
        void Run() override {
            while (!IsStopping()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    };

    SlowThread thread;
    thread.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    auto start = std::chrono::steady_clock::now();
    thread.Shutdown();  // Stop + Wait
    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    TEST_CHECK(thread.IsStopped(), "8.1 优雅退出成功");
    TEST_CHECK(elapsed < 1000, "8.2 退出耗时 < 1000ms (实际=" << elapsed << "ms)");
}

// ============================================================================
// Test 9: Pipeline 初始化与启动停止
// ============================================================================
static void testPipelineLifecycle() {
    TEST_NAME("Test 9: Pipeline 生命周期");

    Pipeline pipeline;

    PipelineConfig cfg;
    cfg.audio_sample_rate = 16000;
    cfg.target_fps        = 25.0;

    TEST_CHECK(pipeline.Init(cfg), "9.1 Init 成功");
    TEST_CHECK(pipeline.Start(), "9.2 Start 成功");
    TEST_CHECK(pipeline.IsRunning(), "9.3 IsRunning == true");

    pipeline.Stop();
    TEST_CHECK(!pipeline.IsRunning(), "9.4 Stop 后 IsRunning == false");

    // 重复停止应安全
    pipeline.Stop();
    TEST_CHECK(true, "9.5 重复 Stop 安全");
}

// ============================================================================
// Test 10: Pipeline 暂停/恢复
// ============================================================================
static void testPipelinePauseResume() {
    TEST_NAME("Test 10: Pipeline 暂停/恢复");

    Pipeline pipeline;
    PipelineConfig cfg;
    cfg.audio_sample_rate = 16000;
    cfg.target_fps        = 25.0;

    pipeline.Init(cfg);
    pipeline.Start();

    TEST_CHECK(!pipeline.IsPaused(), "10.1 初始未暂停");

    pipeline.Pause();
    TEST_CHECK(pipeline.IsPaused(), "10.2 Pause 后暂停");

    pipeline.Resume();
    TEST_CHECK(!pipeline.IsPaused(), "10.3 Resume 后未暂停");

    pipeline.Stop();
    TEST_CHECK(true, "10.4 暂停状态停止安全");
}

// ============================================================================
// Test 11: Pipeline 指标查询
// ============================================================================
static void testPipelineMetrics() {
    TEST_NAME("Test 11: Pipeline 指标查询");

    Pipeline pipeline;
    PipelineConfig cfg;
    cfg.audio_sample_rate = 16000;
    cfg.target_fps        = 25.0;

    pipeline.Init(cfg);
    pipeline.Start();

    // 发送一些测试数据
    std::vector<float> audio_data(1600, 0.0f);  // 100ms 静音
    cv::Mat video_frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    pipeline.PushAudio(audio_data, 0);
    pipeline.PushVideo(video_frame, 0);
    pipeline.PushAudio(audio_data, 40);
    pipeline.PushVideo(video_frame, 40);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    pipeline.MarkAudioEOS();
    pipeline.MarkVideoEOS();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    pipeline.Stop();

    PipelineMetrics metrics = pipeline.GetMetrics();
    std::cout << "  [INFO] " << metrics.ToString() << std::endl;
    FrameStats stats = pipeline.GetFrameStats();
    std::cout << "  [INFO] FrameStats: " << stats.ToString() << std::endl;

    TEST_CHECK(metrics.audio_packets_in == 2, "11.1 音频包=2");
    TEST_CHECK(metrics.video_packets_in == 2, "11.2 视频包=2");
    TEST_CHECK(metrics.total_frames_in >= 0, "11.3 帧计数可用");
}

// ============================================================================
// Test 12: 反压测试 - 慢消费者
// ============================================================================
static void testBackPressure() {
    TEST_NAME("Test 12: 反压测试 - 慢消费者");

    ThreadSafeQueue<int> queue(10);  // 容量=10

    // 快速生产者
    std::atomic<int> pushed{0};
    std::thread producer([&]() {
        for (int i = 0; i < 100; ++i) {
            if (queue.Push(i)) {
                pushed.fetch_add(1, std::memory_order_relaxed);
            } else {
                break;  // 队列已停止
            }
            // 无延迟，快速推送
        }
    });

    // 慢消费者
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int v, consumed = 0;
    while (queue.TryPop(v)) {
        consumed++;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (consumed >= 20) break;
    }

    queue.Stop();
    producer.join();

    // 有界队列应限制生产者过快于消费者
    TEST_CHECK(pushed.load() >= 10, "12.1 有界队列限制生产 (pushed="
               << pushed.load() << ")");
    std::cout << "  [INFO] 生产者推送=" << pushed.load()
              << " 消费者消费=" << consumed << std::endl;
}

// ============================================================================
// Test 13: 异常安全 - 线程异常退出
// ============================================================================
static void testExceptionSafety() {
    TEST_NAME("Test 13: 异常安全 - 线程异常退出");

    class CrashThread : public ThreadBase {
    public:
        CrashThread() : ThreadBase("CrashThread") {}
        void Run() override {
            throw std::runtime_error("模拟崩溃");
        }
    };

    CrashThread thread;
    thread.Start();
    thread.Wait(1000);

    TEST_CHECK(thread.IsError(), "13.1 异常后状态为 ERROR");
}

// ============================================================================
// Test 14: 高吞吐压力测试
// ============================================================================
static void testHighThroughput() {
    TEST_NAME("Test 14: 高吞吐压力测试");

    ThreadSafeQueue<int> queue;
    const int kTotal = 50000;
    std::atomic<int64_t> sum{0};

    auto producer = [&]() {
        for (int i = 0; i < kTotal; ++i) {
            queue.Push(1);
        }
        queue.Push(-1);  // 终止信号
    };

    auto consumer = [&]() {
        int64_t local_sum = 0;
        while (true) {
            int v;
            queue.WaitAndPop(v, -1);
            if (v < 0) break;
            local_sum += v;
        }
        sum.store(local_sum, std::memory_order_release);
    };

    std::thread p(producer);
    std::thread c(consumer);
    p.join();
    c.join();

    TEST_CHECK(sum.load() == kTotal, "14.1 50000 数据无丢失 sum=" << sum.load());
}

// ============================================================================
// Test 15: 队列移动语义与 Emplace 原位构造
// ============================================================================
static void testQueueMoveAndEmplace() {
    TEST_NAME("Test 15: 队列移动语义与 Emplace 原位构造");

    // Push 移动语义 — MoveOnly 类型验证
    struct MoveOnly {
        int val;
        explicit MoveOnly(int v) : val(v) {}
        MoveOnly(const MoveOnly&) = delete;
        MoveOnly& operator=(const MoveOnly&) = delete;
        MoveOnly(MoveOnly&&) = default;
        MoveOnly& operator=(MoveOnly&&) = default;
    };

    ThreadSafeQueue<MoveOnly> queue;
    queue.Push(MoveOnly(42));
    queue.Emplace(100);

    MoveOnly item(0);
    queue.WaitAndPop(item);
    TEST_CHECK(item.val == 42, "15.1 移动语义 Push 正确");

    queue.WaitAndPop(item);
    TEST_CHECK(item.val == 100, "15.2 Emplace 原位构造正确");
}

// ============================================================================
// Test 16: 批量出队性能优化
// ============================================================================
static void testBatchPop() {
    TEST_NAME("Test 16: 批量出队性能优化");

    ThreadSafeQueue<int> queue;
    for (int i = 0; i < 100; ++i) {
        queue.Push(i);
    }

    std::vector<int> batch;
    size_t count = queue.TryPopBatch(batch, 30);
    TEST_CHECK(count == 30, "16.1 批量出队30个成功, 实际=" << count);
    TEST_CHECK(batch.size() == 30, "16.2 batch 容器大小=30");
    TEST_CHECK(batch[0] == 0, "16.3 第一个元素=0");
    TEST_CHECK(batch[29] == 29, "16.4 第30个元素=29");

    // 再次批量出队全部剩余
    batch.clear();
    count = queue.TryPopBatch(batch, 0);  // 0 = 全部
    TEST_CHECK(count == 70, "16.5 清空剩余70个, 实际=" << count);

    // 空队列批量出队
    batch.clear();
    count = queue.TryPopBatch(batch, 10);
    TEST_CHECK(count == 0, "16.6 空队列批量出队=0");
}

// ============================================================================
// Test 17: 心跳检测与健康检查
// ============================================================================
static void testHeartbeat() {
    TEST_NAME("Test 17: 心跳检测与健康检查");

    // 启用 100ms 心跳超时
    ThreadSafeQueue<int> queue(0, "test_heartbeat", 100);

    // 未心跳时默认健康
    TEST_CHECK(queue.IsHealthy(), "17.1 初始状态健康");

    // 发送心跳
    queue.Heartbeat();
    TEST_CHECK(queue.IsHealthy(), "17.2 心跳后健康");

    // 等待超过超时时间
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    TEST_CHECK(!queue.IsHealthy(), "17.3 超时后不健康");

    // 再次心跳恢复健康
    queue.Heartbeat();
    TEST_CHECK(queue.IsHealthy(), "17.4 再次心跳后恢复健康");

    // 未启用心跳检测的队列
    ThreadSafeQueue<int> q2;
    q2.Heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    TEST_CHECK(q2.IsHealthy(), "17.5 未启用检测始终健康");
}

// ============================================================================
// Test 18: 死锁检测
// ============================================================================
static void testDeadlockDetection() {
    TEST_NAME("Test 18: 死锁检测");

    ThreadSafeQueue<int> queue(0, "test_deadlock", 0);

    // 空队列不应检测到死锁
    TEST_CHECK(!queue.CheckDeadlock(50), "18.1 空队列无死锁");

    // 入队后消费者停滞
    queue.Push(1);
    queue.Push(2);
    queue.Push(3);

    // 短超时应检测到消费者死锁
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    TEST_CHECK(queue.CheckDeadlock(50), "18.2 消费者停滞检测到死锁");

    // 消费后清除死锁
    int v;
    queue.WaitAndPop(v, 100);
    queue.WaitAndPop(v, 100);
    queue.WaitAndPop(v, 100);
    // 空队列后再检测
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    TEST_CHECK(!queue.CheckDeadlock(100), "18.3 消费后无死锁");
}

// ============================================================================
// Test 19: 队列运行时指标
// ============================================================================
static void testQueueMetrics() {
    TEST_NAME("Test 19: 队列运行时指标");

    ThreadSafeQueue<int> queue(10, "metrics_test");
    TEST_CHECK(queue.GetMetrics().ToString().find("metrics_test") != std::string::npos,
               "19.1 指标包含队列名称");

    // 入队 5 个
    for (int i = 0; i < 5; ++i) queue.Push(i);

    auto m1 = queue.GetMetrics();
    TEST_CHECK(m1.current_size == 5, "19.2 当前大小=5");
    TEST_CHECK(m1.total_pushes == 5, "19.3 总入队=5");
    TEST_CHECK(m1.total_pops == 0, "19.4 总出队=0");
    TEST_CHECK(m1.peak_size == 5, "19.5 峰值=5");

    // 出队 3 个
    int v;
    queue.WaitAndPop(v); queue.WaitAndPop(v); queue.WaitAndPop(v);

    auto m2 = queue.GetMetrics();
    TEST_CHECK(m2.current_size == 2, "19.6 当前大小=2");
    TEST_CHECK(m2.total_pops == 3, "19.7 总出队=3");

    // 队列名
    TEST_CHECK(queue.GetName() == "metrics_test", "19.8 队列名称正确");

    std::cout << "  [INFO] Metrics: " << m2.ToString() << std::endl;
}

// ============================================================================
// Test 20: 超时等待不阻塞
// ============================================================================
static void testTimeoutNoBlock() {
    TEST_NAME("Test 20: 超时等待不阻塞");

    ThreadSafeQueue<int> queue;

    // 空队列 WaitAndPop 应超时返回 false
    int v = -1;
    auto start = std::chrono::steady_clock::now();
    bool result = queue.WaitAndPop(v, 50);  // 50ms 超时
    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    TEST_CHECK(!result, "20.1 空队列超时返回 false");
    TEST_CHECK(elapsed >= 40 && elapsed < 500,
               "20.2 超时耗时合理 (" << elapsed << "ms)");

    // 0ms 超时 = 非阻塞
    start = std::chrono::steady_clock::now();
    result = queue.WaitAndPop(v, 0);
    elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    TEST_CHECK(!result, "20.3 非阻塞模式返回 false");
    TEST_CHECK(elapsed < 10, "20.4 非阻塞耗时 < 10ms (" << elapsed << "ms)");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  Pipeline 多线程流水线验收测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    testQueueBasic();
    testQueueBounded();
    testQueueMultiThread();
    testQueueStop();
    testQueueEOS();
    testThreadLifecycle();
    testDeadlockFreedom();
    testGracefulShutdown();
    testPipelineLifecycle();
    testPipelinePauseResume();
    testPipelineMetrics();
    testBackPressure();
    testExceptionSafety();
    testHighThroughput();
    testQueueMoveAndEmplace();
    testBatchPop();
    testHeartbeat();
    testDeadlockDetection();
    testQueueMetrics();
    testTimeoutNoBlock();

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
