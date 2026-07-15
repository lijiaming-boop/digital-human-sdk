/**
 * @file av_sync_test.cpp
 * @brief AVSync 音视频同步模块验收测试
 *
 * 覆盖范围：
 * - 正常同步（drift=0）
 * - 视频滞后（drift < -threshold）
 * - 视频超前（drift > threshold）
 * - 严重偏移（|drift| >= max_drift）
 * - 边界阈值测试
 * - 音频时钟更新与设置
 * - 多次同步场景
 * - Reset/Move/配置修改
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>

#include "core/av_sync.h"

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

/// @brief 创建默认同步配置
static SyncConfig defaultConfig() {
    SyncConfig cfg;
    cfg.audio_sample_rate = 16000;
    cfg.sync_threshold_ms = 30.0;
    cfg.max_drift_ms      = 100.0;
    return cfg;
}

// ============================================================================
// 测试用例
// ============================================================================

// ---- Test 1: 基本同步 ----
static void testBasicSync() {
    TEST_NAME("Test 1: 基本同步 (drift=0)");

    AVSync sync;
    sync.Init(defaultConfig());

    // 音画完全同步
    sync.SetAudioClockMs(500.0);
    SyncResult r = sync.GetSyncStatus(500.0);

    TEST_CHECK(r.status == SyncStatus::SYNCED,       "1.1 状态 == SYNCED");
    TEST_CHECK(!r.should_drop_frame,                  "1.2 should_drop_frame == false");
    TEST_CHECK(std::abs(r.drift_ms) < 1e-9,           "1.3 drift == 0");
    TEST_CHECK(r.wait_time_ms == 0.0,                 "1.4 wait_time == 0");
    TEST_CHECK(r.audio_clock_ms == 500.0,             "1.5 audio_clock == 500");
    TEST_CHECK(r.video_pts_ms == 500.0,               "1.6 video_pts == 500");
}

// ---- Test 2: 视频超前 ----
static void testVideoAhead() {
    TEST_NAME("Test 2: 视频超前 (drift=+50ms)");

    AVSync sync;
    sync.Init(defaultConfig());
    sync.SetAudioClockMs(500.0);

    // 视频比音频快 50ms（> 30ms 阈值）
    SyncResult r = sync.GetSyncStatus(550.0);

    TEST_CHECK(r.status == SyncStatus::VIDEO_AHEAD,   "2.1 状态 == VIDEO_AHEAD");
    TEST_CHECK(!r.should_drop_frame,                  "2.2 should_drop_frame == false（等待即可）");
    TEST_CHECK(std::abs(r.drift_ms - 50.0) < 1e-9,    "2.3 drift == 50ms");
    TEST_CHECK(std::abs(r.wait_time_ms - 50.0) < 1e-9,"2.4 wait_time == 50ms");

    // 输出状态字符串
    std::cout << "  [INFO] StatusString: " << r.StatusString() << std::endl;
}

// ---- Test 3: 视频滞后 ----
static void testVideoBehind() {
    TEST_NAME("Test 3: 视频滞后 (drift=-50ms)");

    AVSync sync;
    sync.Init(defaultConfig());
    sync.SetAudioClockMs(500.0);

    // 视频比音频慢 50ms（> 30ms 阈值）
    SyncResult r = sync.GetSyncStatus(450.0);

    TEST_CHECK(r.status == SyncStatus::VIDEO_BEHIND,  "3.1 状态 == VIDEO_BEHIND");
    TEST_CHECK(r.should_drop_frame,                    "3.2 should_drop_frame == true（丢弃旧帧）");
    TEST_CHECK(std::abs(r.drift_ms - (-50.0)) < 1e-9, "3.3 drift == -50ms");
    TEST_CHECK(r.wait_time_ms == 0.0,                  "3.4 wait_time == 0（无需等待）");
}

// ---- Test 4: 严重偏移 ----
static void testSevereOffset() {
    TEST_NAME("Test 4: 严重偏移 (drift=+200ms)");

    AVSync sync;
    sync.Init(defaultConfig());
    sync.SetAudioClockMs(500.0);

    // 视频超前 200ms（> 100ms 严重偏移阈值）
    SyncResult r = sync.GetSyncStatus(700.0);

    TEST_CHECK(r.status == SyncStatus::SEVERE_OFFSET, "4.1 状态 == SEVERE_OFFSET");
    TEST_CHECK(r.should_drop_frame,                    "4.2 should_drop_frame == true");
    TEST_CHECK(std::abs(r.drift_ms - 200.0) < 1e-9,   "4.3 drift == 200ms");

    // 严重滞后方向
    SyncResult r2 = sync.GetSyncStatus(200.0);
    TEST_CHECK(r2.status == SyncStatus::SEVERE_OFFSET, "4.4 严重滞后 → SEVERE_OFFSET");
    TEST_CHECK(r2.should_drop_frame,                   "4.5 should_drop_frame == true");
    TEST_CHECK(std::abs(r2.drift_ms - (-300.0)) < 1e-9,"4.6 drift == -300ms");
}

// ---- Test 5: 边界同步（drift 略小于 threshold） ----
static void testBoundarySync() {
    TEST_NAME("Test 5: 边界同步 (drift=+29ms, <30ms)");

    AVSync sync;
    sync.Init(defaultConfig());
    sync.SetAudioClockMs(500.0);

    // drift = 29ms，小于阈值 30ms → SYNCED
    SyncResult r = sync.GetSyncStatus(529.0);

    TEST_CHECK(r.status == SyncStatus::SYNCED,        "5.1 drift=29ms → SYNCED");
    TEST_CHECK(!r.should_drop_frame,                   "5.2 should_drop_frame == false");
    TEST_CHECK(std::abs(r.drift_ms - 29.0) < 1e-9,    "5.3 drift == 29ms");
}

// ---- Test 6: 边界超前（drift 略大于 threshold） ----
static void testBoundaryAhead() {
    TEST_NAME("Test 6: 边界超前 (drift=+31ms, >30ms)");

    AVSync sync;
    sync.Init(defaultConfig());
    sync.SetAudioClockMs(500.0);

    // drift = 31ms，大于阈值 30ms → VIDEO_AHEAD
    SyncResult r = sync.GetSyncStatus(531.0);

    TEST_CHECK(r.status == SyncStatus::VIDEO_AHEAD,   "6.1 drift=31ms → VIDEO_AHEAD");
    TEST_CHECK(!r.should_drop_frame,                   "6.2 should_drop_frame == false");
    TEST_CHECK(std::abs(r.drift_ms - 31.0) < 1e-9,    "6.3 drift == 31ms");
    TEST_CHECK(std::abs(r.wait_time_ms - 31.0) < 1e-9,"6.4 wait_time == 31ms");
}

// ---- Test 7: UpdateAudioClock ----
static void testUpdateAudioClock() {
    TEST_NAME("Test 7: UpdateAudioClock 样本数→毫秒转换");

    AVSync sync;
    sync.Init(defaultConfig());

    // 7.1 16000 样本 = 1000ms
    sync.UpdateAudioClock(16000);
    TEST_CHECK(std::abs(sync.GetAudioClockMs() - 1000.0) < 1e-6,
               "7.1 16000 samples @ 16kHz = 1000ms");

    // 7.2 再追加 8000 样本 = 500ms，累计 1500ms
    sync.UpdateAudioClock(8000);
    TEST_CHECK(std::abs(sync.GetAudioClockMs() - 1500.0) < 1e-6,
               "7.2 +8000 samples = +500ms, 累计 1500ms");

    // 7.3 160 样本 = 10ms
    sync.Reset();
    sync.UpdateAudioClock(160);
    TEST_CHECK(std::abs(sync.GetAudioClockMs() - 10.0) < 1e-6,
               "7.3 160 samples @ 16kHz = 10ms");

    // 7.4 不同采样率
    SyncConfig cfg48k;
    cfg48k.audio_sample_rate = 48000;
    cfg48k.sync_threshold_ms = 30.0;
    cfg48k.max_drift_ms      = 100.0;
    AVSync sync48k;
    sync48k.Init(cfg48k);
    sync48k.UpdateAudioClock(48000);
    TEST_CHECK(std::abs(sync48k.GetAudioClockMs() - 1000.0) < 1e-6,
               "7.4 48000 samples @ 48kHz = 1000ms");

    // 7.5 负样本不更新
    double before = sync48k.GetAudioClockMs();
    sync48k.UpdateAudioClock(-100);
    TEST_CHECK(std::abs(sync48k.GetAudioClockMs() - before) < 1e-6,
               "7.5 负样本数不更新时钟");
}

// ---- Test 8: SetAudioClockMs ----
static void testSetAudioClockMs() {
    TEST_NAME("Test 8: SetAudioClockMs 手动设值");

    AVSync sync;
    sync.Init(defaultConfig());

    sync.SetAudioClockMs(1234.567);
    TEST_CHECK(std::abs(sync.GetAudioClockMs() - 1234.567) < 1e-6,
               "8.1 设置时钟 1234.567ms");

    // 设值后同步判断应基于新时钟
    SyncResult r = sync.GetSyncStatus(1234.567);
    TEST_CHECK(r.status == SyncStatus::SYNCED, "8.2 设值后 drift=0 → SYNCED");

    // 设回 0
    sync.SetAudioClockMs(0.0);
    TEST_CHECK(std::abs(sync.GetAudioClockMs()) < 1e-6, "8.3 设回 0");
}

// ---- Test 9: 多次同步（模拟播放过程） ----
static void testMultiSync() {
    TEST_NAME("Test 9: 多次同步（模拟播放）");

    AVSync sync;
    sync.Init(defaultConfig());

    // 模拟：音频消耗 1600 样本/帧 @ 25fps = 每帧 40ms
    // 视频帧率为 25fps = 每帧 40ms → 完全同步

    struct TestStep {
        int    samples;       // 本次消耗的音频样本数
        double video_pts_ms;  // 视频帧 PTS
    };

    // Sync() 先调用 UpdateAudioClock 再 GetSyncStatus，
    // 所以 video_pts 应匹配更新后的音频时钟。
    // 每帧消耗 1600 样本 @16kHz = 100ms
    TestStep steps[] = {
        //    样本   视频PTS（匹配更新后时钟）
        {   1600,   100.0 },   // 第 1 帧：audio=100ms, video=100ms
        {   1600,   200.0 },   // 第 2 帧：audio=200ms, video=200ms
        {   1600,   300.0 },   // 第 3 帧：audio=300ms, video=300ms
        {   1600,   400.0 },   // 第 4 帧：audio=400ms, video=400ms
        {   1600,   500.0 },   // 第 5 帧：audio=500ms, video=500ms
    };

    for (int i = 0; i < 5; ++i) {
        SyncResult r = sync.Sync(steps[i].samples, steps[i].video_pts_ms);
        TEST_CHECK(r.status == SyncStatus::SYNCED,
                   "9." << (i+1) << " 第 " << (i+1) << " 帧同步");
        TEST_CHECK(!r.should_drop_frame,
                   "9." << (i+1) << " 不丢弃帧");
    }

    // 此时 audio_clock = 500ms（5帧 × 100ms/帧）
    TEST_CHECK(std::abs(sync.GetAudioClockMs() - 500.0) < 1e-6,
               "9.6 5 帧后音频时钟 = 500ms");
}

// ---- Test 10: Reset ----
static void testReset() {
    TEST_NAME("Test 10: Reset");

    AVSync sync;
    sync.Init(defaultConfig());
    sync.UpdateAudioClock(16000);
    TEST_CHECK(sync.GetAudioClockMs() > 0, "10.1 更新后时钟 > 0");

    sync.Reset();
    TEST_CHECK(std::abs(sync.GetAudioClockMs()) < 1e-6, "10.2 Reset 后时钟 = 0");

    // 重置后同步仍正常工作
    SyncResult r = sync.GetSyncStatus(0.0);
    TEST_CHECK(r.status == SyncStatus::SYNCED, "10.3 Reset 后同步正常");
}

// ---- Test 11: Move 语义 ----
static void testMoveSemantics() {
    TEST_NAME("Test 11: Move 语义");

    AVSync sync1;
    sync1.Init(defaultConfig());
    sync1.UpdateAudioClock(16000);  // audio = 1000ms
    TEST_CHECK(sync1.IsInitialized(), "11.1 sync1 已初始化");

    // Move 构造
    AVSync sync2(std::move(sync1));
    TEST_CHECK(sync2.IsInitialized(), "11.2 移动后 sync2 已初始化");
    TEST_CHECK(std::abs(sync2.GetAudioClockMs() - 1000.0) < 1e-6,
               "11.3 移动后时钟保留 1000ms");

    // 移动后同步正常
    SyncResult r = sync2.GetSyncStatus(1000.0);
    TEST_CHECK(r.status == SyncStatus::SYNCED, "11.4 移动后同步正常");

    // Move 赋值
    AVSync sync3;
    sync3 = std::move(sync2);
    TEST_CHECK(sync3.IsInitialized(), "11.5 赋值后 sync3 已初始化");
    r = sync3.GetSyncStatus(1000.0);
    TEST_CHECK(r.status == SyncStatus::SYNCED, "11.6 赋值后同步正常");
}

// ---- Test 12: 配置修改 ----
static void testConfigChange() {
    TEST_NAME("Test 12: 配置修改");

    AVSync sync;
    sync.Init(defaultConfig());

    // 默认配置
    SyncConfig cfg = sync.GetConfig();
    TEST_CHECK(cfg.sync_threshold_ms == 30.0, "12.1 默认 threshold = 30ms");
    TEST_CHECK(cfg.max_drift_ms == 100.0,     "12.2 默认 max_drift = 100ms");

    // 修改阈值
    sync.SetSyncThresholdMs(50.0);
    TEST_CHECK(sync.GetConfig().sync_threshold_ms == 50.0, "12.3 threshold = 50ms");

    // 修改后判断逻辑随之变化
    sync.SetAudioClockMs(500.0);
    SyncResult r = sync.GetSyncStatus(530.0);  // drift=30ms < 50ms
    TEST_CHECK(r.status == SyncStatus::SYNCED,
               "12.4 threshold=50ms, drift=30ms → SYNCED");

    r = sync.GetSyncStatus(560.0);  // drift=60ms > 50ms
    TEST_CHECK(r.status == SyncStatus::VIDEO_AHEAD,
               "12.5 threshold=50ms, drift=60ms → VIDEO_AHEAD");

    // 修改最大漂移
    sync.SetMaxDriftMs(200.0);
    TEST_CHECK(sync.GetConfig().max_drift_ms == 200.0, "12.6 max_drift = 200ms");

    // 最小值 clamp
    sync.SetSyncThresholdMs(0.0);
    TEST_CHECK(sync.GetConfig().sync_threshold_ms == 1.0, "12.7 threshold clamp 为 1ms");

    sync.SetMaxDriftMs(-10.0);
    TEST_CHECK(sync.GetConfig().max_drift_ms == 1.0, "12.8 max_drift clamp 为 1ms");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  AVSync 音视频同步模块验收测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    testBasicSync();
    testVideoAhead();
    testVideoBehind();
    testSevereOffset();
    testBoundarySync();
    testBoundaryAhead();
    testUpdateAudioClock();
    testSetAudioClockMs();
    testMultiSync();
    testReset();
    testMoveSemantics();
    testConfigChange();

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
