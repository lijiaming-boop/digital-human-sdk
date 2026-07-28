/**
 * @file frame_scheduler_test.cpp
 * @brief FrameScheduler 帧调度模块验收测试
 *
 * 覆盖范围：
 * - 基础调度（25fps 正常递增）
 * - 掉帧处理（视频滞后）
 * - 重复帧处理（视频超前）
 * - 边界调度（阈值内偏移）
 * - 连续掉帧/重复
 * - 统计信息正确性
 * - 帧率配置和切换
 * - Reset/Move
 * - 大量帧压力测试
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include "core/frame_scheduler.h"

using namespace digital_human::core;

// ============================================================================
// 辅助函数
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

static SchedulerConfig defaultConfig() {
    SchedulerConfig cfg;
    cfg.target_fps        = 25.0;
    cfg.smoothing_factor  = 0.5;
    cfg.enable_smoothing  = true;
    cfg.max_pending_frames = 10;
    return cfg;
}

// ============================================================================
// 测试用例
// ============================================================================

// ---- Test 1: 基本调度 ----
static void testBasicSchedule() {
    TEST_NAME("Test 1: 基本调度 (25fps, 40ms间隔)");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    // 25fps 间隔 40ms，连续 5 帧
    for (int i = 0; i < 5; ++i) {
        double pts = i * 40.0;
        ScheduleResult r = sched.ScheduleFrame(i, pts);
        sched.OnFrameDisplayed(pts);
        TEST_CHECK(r.action == FrameAction::DISPLAY,
                   "1." << (i+1) << " 帧" << i << " PTS=" << pts << "ms → DISPLAY");
    }

    FrameStats stats = sched.GetStats();
    TEST_CHECK(stats.total_frames == 5,      "1.6 total_frames == 5");
    TEST_CHECK(stats.frames_displayed == 5,   "1.7 frames_displayed == 5");
    TEST_CHECK(stats.frames_dropped == 0,     "1.8 frames_dropped == 0");
    TEST_CHECK(stats.frames_duplicated == 0,  "1.9 frames_duplicated == 0");
}

// ---- Test 2: 掉帧处理 ----
static void testDropFrame() {
    TEST_NAME("Test 2: 掉帧处理 (视频滞后)");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    // 第一帧正常
    ScheduleResult r0 = sched.ScheduleFrame(0, 0.0);
    sched.OnFrameDisplayed(0.0);
    TEST_CHECK(r0.action == FrameAction::DISPLAY, "2.1 首帧 DISPLAY");

    // 下一帧 PTS=20ms，期望 PTS=40ms，滞后 20ms > 半帧 20ms
    // diff = 20 - 40 = -20, half=20, diff >= -half → 边界值显示
    // diff = 20 - 40 = -20, half=20: diff 不 < -20，所以不 DROP
    // 用 PTS=10ms: diff = 10 - 40 = -30 < -20 → DROP
    ScheduleResult r1 = sched.ScheduleFrame(1, 10.0);
    TEST_CHECK(r1.action == FrameAction::DROP, "2.2 PTS滞后30ms → DROP");

    // 统计
    FrameStats stats = sched.GetStats();
    TEST_CHECK(stats.frames_dropped == 1, "2.3 frames_dropped == 1");
    TEST_CHECK(stats.frames_displayed == 1, "2.4 frames_displayed == 1");
}

// ---- Test 3: 重复帧处理 ----
static void testDuplicateFrame() {
    TEST_NAME("Test 3: 重复帧处理 (视频超前)");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    // 第一帧正常
    ScheduleResult r0 = sched.ScheduleFrame(0, 0.0);
    sched.OnFrameDisplayed(0.0);
    TEST_CHECK(r0.action == FrameAction::DISPLAY, "3.1 首帧 DISPLAY");

    // 下一帧 PTS=80ms，期望 PTS=40ms，超前 40ms > 半帧 20ms → DUPLICATE
    ScheduleResult r1 = sched.ScheduleFrame(1, 80.0);
    TEST_CHECK(r1.action == FrameAction::DUPLICATE, "3.2 PTS超前40ms → DUPLICATE");

    // 统计
    FrameStats stats = sched.GetStats();
    TEST_CHECK(stats.frames_duplicated == 1, "3.3 frames_duplicated == 1");
    TEST_CHECK(stats.frames_displayed == 1, "3.4 frames_displayed == 1");
}

// ---- Test 4: 边界调度 ----
static void testBoundarySchedule() {
    TEST_NAME("Test 4: 边界调度 (阈值内偏移)");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    // 第一帧正常
    sched.ScheduleFrame(0, 0.0);
    sched.OnFrameDisplayed(0.0);

    // 下一帧 PTS=50ms，期望 PTS=40ms，超前 10ms < 半帧 20ms → DISPLAY
    ScheduleResult r1 = sched.ScheduleFrame(1, 50.0);
    sched.OnFrameDisplayed(50.0);
    TEST_CHECK(r1.action == FrameAction::DISPLAY,
               "4.1 PTS超前10ms (<半帧) → DISPLAY");

    // 下一帧 PTS=70ms，期望=40+40=80ms，滞后10ms < 半帧 → DISPLAY
    ScheduleResult r2 = sched.ScheduleFrame(2, 70.0);
    sched.OnFrameDisplayed(70.0);
    TEST_CHECK(r2.action == FrameAction::DISPLAY,
               "4.2 PTS滞后10ms (<半帧) → DISPLAY");
}

// ---- Test 5: 连续掉帧 ----
static void testContinuousDrop() {
    TEST_NAME("Test 5: 连续掉帧");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    // 首帧
    sched.ScheduleFrame(0, 0.0);
    sched.OnFrameDisplayed(0.0);

    // 连续 5 帧全部严重滞后（每帧都卡在同一 PTS=5ms，远落后于期望的 40/80/120...）
    for (int i = 1; i <= 5; ++i) {
        ScheduleResult r = sched.ScheduleFrame(i, 5.0);
        TEST_CHECK(r.action == FrameAction::DROP,
                   "5." << i << " 帧" << i << " PTS=5ms → DROP");
    }

    FrameStats stats = sched.GetStats();
    TEST_CHECK(stats.frames_dropped == 5, "5.6 连续丢弃 5 帧");
    TEST_CHECK(stats.frames_displayed == 1, "5.7 仅首帧显示");
}

// ---- Test 6: 连续重复 ----
static void testContinuousDuplicate() {
    TEST_NAME("Test 6: 连续重复帧");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    // 首帧
    sched.ScheduleFrame(0, 0.0);
    sched.OnFrameDisplayed(0.0);

    // 连续 5 帧全部严重超前（PTS 跳变 200ms）
    for (int i = 1; i <= 5; ++i) {
        ScheduleResult r = sched.ScheduleFrame(i, i * 200.0);
        TEST_CHECK(r.action == FrameAction::DUPLICATE,
                   "6." << i << " 帧" << i << " PTS=" << (i*200) << "ms → DUPLICATE");
    }

    FrameStats stats = sched.GetStats();
    TEST_CHECK(stats.frames_duplicated == 5, "6.6 连续重复 5 帧");
    TEST_CHECK(stats.frames_displayed == 1, "6.7 仅首帧显示");
}

// ---- Test 7: 统计信息 ----
static void testStats() {
    TEST_NAME("Test 7: 统计信息正确性");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    // 正常: 2 帧
    sched.ScheduleFrame(0, 0.0);
    sched.OnFrameDisplayed(0.0);
    sched.ScheduleFrame(1, 40.0);
    sched.OnFrameDisplayed(40.0);

    // 丢弃: 1 帧
    sched.ScheduleFrame(2, 30.0);

    // 重复: 1 帧
    sched.ScheduleFrame(3, 200.0);

    FrameStats stats = sched.GetStats();
    std::cout << "  [INFO] " << stats.ToString() << std::endl;

    TEST_CHECK(stats.total_frames == 4,       "7.1 total_frames == 4");
    TEST_CHECK(stats.frames_displayed == 2,   "7.2 frames_displayed == 2");
    TEST_CHECK(stats.frames_dropped == 1,     "7.3 frames_dropped == 1");
    TEST_CHECK(stats.frames_duplicated == 1,  "7.4 frames_duplicated == 1");
    TEST_CHECK(stats.actual_fps > 0,          "7.5 actual_fps > 0");
    TEST_CHECK(stats.avg_frame_interval_ms > 0, "7.6 avg_frame_interval_ms > 0");
}

// ---- Test 8: 帧率配置 ----
static void testFpsConfig() {
    TEST_NAME("Test 8: 帧率配置");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    TEST_CHECK(std::abs(sched.GetTargetFps() - 25.0) < 1e-9,    "8.1 默认 25fps");
    TEST_CHECK(std::abs(sched.GetFrameIntervalMs() - 40.0) < 1e-9, "8.2 间隔 40ms");

    sched.SetTargetFps(30.0);
    TEST_CHECK(std::abs(sched.GetTargetFps() - 30.0) < 1e-9,    "8.3 Set 30fps");
    TEST_CHECK(std::abs(sched.GetFrameIntervalMs() - 33.333) < 0.001, "8.4 间隔 ≈33.3ms");

    sched.SetTargetFps(60.0);
    TEST_CHECK(std::abs(sched.GetFrameIntervalMs() - 16.666) < 0.001, "8.5 60fps 间隔 ≈16.7ms");

    // 设置为 0（应忽略）
    sched.SetTargetFps(0.0);
    TEST_CHECK(std::abs(sched.GetTargetFps() - 60.0) < 1e-9, "8.6 fps=0 忽略，保持 60fps");
}

// ---- Test 9: 不同帧率调度 ----
static void testDifferentFps() {
    TEST_NAME("Test 9: 不同帧率调度");

    // 30fps = 33.33ms 间隔
    SchedulerConfig cfg30;
    cfg30.target_fps = 30.0;
    cfg30.smoothing_factor = 0.5;
    cfg30.enable_smoothing = true;

    FrameScheduler sched;
    sched.Init(cfg30);

    for (int i = 0; i < 3; ++i) {
        double pts = i * 33.333;
        ScheduleResult r = sched.ScheduleFrame(i, pts);
        sched.OnFrameDisplayed(pts);
        TEST_CHECK(r.action == FrameAction::DISPLAY,
                   "9." << (i+1) << " 30fps 帧" << i << " DISPLAY");
    }
}

// ---- Test 10: Reset ----
static void testReset() {
    TEST_NAME("Test 10: Reset");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    sched.ScheduleFrame(0, 0.0);
    sched.OnFrameDisplayed(0.0);
    sched.ScheduleFrame(1, 100.0); // 超前 → DUPLICATE

    FrameStats before = sched.GetStats();
    TEST_CHECK(before.total_frames > 0, "10.1 Reset 前有统计数据");

    sched.Reset();
    FrameStats after = sched.GetStats();
    TEST_CHECK(after.total_frames == 0,        "10.2 Reset 后 total=0");
    TEST_CHECK(after.frames_displayed == 0,    "10.3 Reset 后 displayed=0");
    TEST_CHECK(after.frames_dropped == 0,      "10.4 Reset 后 dropped=0");
    TEST_CHECK(after.frames_duplicated == 0,   "10.5 Reset 后 duplicated=0");

    // Reset 后重新调度
    ScheduleResult r = sched.ScheduleFrame(0, 0.0);
    TEST_CHECK(r.action == FrameAction::DISPLAY, "10.6 Reset 后首帧 DISPLAY");
}

// ---- Test 11: Move 语义 ----
static void testMoveSemantics() {
    TEST_NAME("Test 11: Move 语义");

    FrameScheduler sched1;
    sched1.Init(defaultConfig());
    sched1.ScheduleFrame(0, 0.0);
    sched1.OnFrameDisplayed(0.0);
    sched1.ScheduleFrame(1, 40.0);
    sched1.OnFrameDisplayed(40.0);

    FrameStats before = sched1.GetStats();
    TEST_CHECK(before.frames_displayed == 2, "11.1 sched1 显示 2 帧");

    // Move 构造
    FrameScheduler sched2(std::move(sched1));
    FrameStats after = sched2.GetStats();
    TEST_CHECK(after.frames_displayed == 2, "11.2 移动后统计保留");

    // 移动后仍可调度
    ScheduleResult r = sched2.ScheduleFrame(2, 80.0);
    sched2.OnFrameDisplayed(80.0);
    TEST_CHECK(r.action == FrameAction::DISPLAY, "11.3 移动后调度正常");

    // Move 赋值
    FrameScheduler sched3;
    sched3 = std::move(sched2);
    r = sched3.ScheduleFrame(3, 120.0);
    TEST_CHECK(r.action == FrameAction::DISPLAY, "11.4 赋值后调度正常");
}

// ---- Test 12: 大量帧压力测试 ----
static void testBulkFrames() {
    TEST_NAME("Test 12: 大量帧压力测试 (1000帧)");

    FrameScheduler sched;
    sched.Init(defaultConfig());

    const int kFrameCount = 1000;
    int displayed = 0, dropped = 0, duplicated = 0;

    for (int i = 0; i < kFrameCount; ++i) {
        double pts = i * 40.0;  // 完美同步
        ScheduleResult r = sched.ScheduleFrame(i, pts);
        sched.OnFrameDisplayed(pts);

        if (r.action == FrameAction::DISPLAY) displayed++;
        else if (r.action == FrameAction::DROP) dropped++;
        else if (r.action == FrameAction::DUPLICATE) duplicated++;
    }

    FrameStats stats = sched.GetStats();
    TEST_CHECK(stats.total_frames == kFrameCount, "12.1 总帧数 == 1000");
    TEST_CHECK(stats.frames_displayed > 900, "12.2 显示帧 > 900（首帧外全部显示）");
    TEST_CHECK(stats.frames_dropped == 0,   "12.3 无丢弃帧");
    TEST_CHECK(stats.frames_duplicated == 0, "12.4 无重复帧");

    std::cout << "  [INFO] 1000帧调度完成: displayed=" << stats.frames_displayed
              << " dropped=" << stats.frames_dropped
              << " duplicated=" << stats.frames_duplicated << std::endl;
    std::cout << "  [INFO] " << stats.ToString() << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  FrameScheduler 帧调度模块验收测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    testBasicSchedule();
    testDropFrame();
    testDuplicateFrame();
    testBoundarySchedule();
    testContinuousDrop();
    testContinuousDuplicate();
    testStats();
    testFpsConfig();
    testDifferentFps();
    testReset();
    testMoveSemantics();
    testBulkFrames();

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
