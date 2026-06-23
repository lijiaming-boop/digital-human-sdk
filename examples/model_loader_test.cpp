#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <filesystem>
#include <atomic>

#include "model/model_loader.h"

namespace fs = std::filesystem;
using namespace digital_human::model;

std::string resolvePath(const fs::path& relative) {
    fs::path dir = fs::current_path();
    while (true) {
        fs::path candidate = dir / relative;
        if (fs::exists(candidate)) return candidate.string();
        fs::path parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return relative.string();
}

// Test helpers
static std::atomic<int> g_pass{0};
static std::atomic<int> g_fail{0};

#define TEST(name) std::cout << "\n[Test " << name << "] "
#define PASS() do { std::cout << "PASS" << std::endl; g_pass++; } while(0)
#define FAIL(msg) do { std::cout << "FAIL: " << msg << std::endl; g_fail++; } while(0)
#define CHECK(cond, msg) if (!(cond)) { FAIL(msg); return; }

// ---------------------------------------------------------
// Test 1: LoadAsync rejects non-existent directory
// ---------------------------------------------------------
void test_nonexistent_dir() {
    TEST("non-existent directory");
    ModelLoader loader;
    std::atomic<bool> called{false};

    loader.LoadAsync("/nonexistent/path/models", [&](ncnn::Net* net, float io, float wu) {
        called = true;
        CHECK(net == nullptr, "net should be null for failed load");
        CHECK(io == 0.0f, "io cost should be 0 for failed load");
        CHECK(wu == 0.0f, "warmup cost should be 0 for failed load");
    });

    loader.Wait();
    CHECK(!loader.IsLoaded(), "IsLoaded should be false");
    CHECK(loader.GetNet() == nullptr, "GetNet should return nullptr");
    CHECK(loader.GetIOCostMs() == 0.0f, "IOCostMs should be 0");
    CHECK(loader.GetWarmupCostMs() == 0.0f, "WarmupCostMs should be 0");
    PASS();
}

// ---------------------------------------------------------
// Test 2: LoadAsync rejects missing param file
// ---------------------------------------------------------
void test_missing_param() {
    TEST("missing .param file");
    ModelLoader loader;

    std::string valid_dir = resolvePath("models");
    std::string bin = valid_dir + "/Wav2Lip-SD-GAN-opt.bin";
    std::string param = valid_dir + "/nonexistent.param";

    std::atomic<bool> called{false};
    loader.LoadAsync(param, bin, [&](ncnn::Net* net, float io, float wu) {
        called = true;
        CHECK(net == nullptr, "net should be null");
    });

    loader.Wait();
    CHECK(!loader.IsLoaded(), "IsLoaded should be false");
    CHECK(called.load(), "callback should have been called");
    PASS();
}

// ---------------------------------------------------------
// Test 3: LoadAsync rejects missing bin file
// ---------------------------------------------------------
void test_missing_bin() {
    TEST("missing .bin file");
    ModelLoader loader;

    std::string valid_dir = resolvePath("models");
    std::string param = valid_dir + "/Wav2Lip-SD-GAN-opt.param";
    std::string bin = valid_dir + "/nonexistent.bin";

    std::atomic<bool> called{false};
    loader.LoadAsync(param, bin, [&](ncnn::Net* net, float io, float wu) {
        called = true;
        CHECK(net == nullptr, "net should be null");
    });

    loader.Wait();
    CHECK(!loader.IsLoaded(), "IsLoaded should be false");
    CHECK(called.load(), "callback should have been called");
    PASS();
}

// ---------------------------------------------------------
// Test 4: Successful async load with IO timing
// ---------------------------------------------------------
void test_successful_load() {
    TEST("successful async load with IO & warmup timing");
    ModelLoader loader;

    std::string model_dir = resolvePath("models");
    std::cout << " (using " << model_dir << ")" << std::endl;

    std::atomic<bool> called{false};
    float io_cost = -1.0f, warmup_cost = -1.0f;

    loader.LoadAsync(model_dir, [&](ncnn::Net* net, float io, float wu) {
        called = true;
        io_cost = io;
        warmup_cost = wu;
        CHECK(net != nullptr, "net should not be null on success");
    });

    // Busy-wait a bit to show non-blocking behavior
    std::cout << "  Main thread is free during async load..." << std::endl;
    while (!called.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    CHECK(called.load(), "callback should have been called");
    CHECK(io_cost > 0.0f, "IO cost should be > 0");
    std::cout << "  IO cost: " << io_cost << " ms" << std::endl;
    std::cout << "  Warmup cost: " << warmup_cost << " ms" << std::endl;

    CHECK(loader.IsLoaded(), "IsLoaded should return true");
    CHECK(loader.GetNet() != nullptr, "GetNet should return valid pointer");
    CHECK(loader.GetIOCostMs() == io_cost, "GetIOCostMs should match callback value");
    CHECK(loader.GetWarmupCostMs() == warmup_cost, "GetWarmupCostMs should match callback value");
    PASS();
}

// ---------------------------------------------------------
// Test 5: Wait() blocks until load completes
// ---------------------------------------------------------
void test_wait_blocks() {
    TEST("Wait() blocks until loading completes");
    ModelLoader loader;

    std::string model_dir = resolvePath("models");

    auto t0 = std::chrono::steady_clock::now();
    loader.LoadAsync(model_dir, nullptr);

    // Wait should block until loading is done
    loader.Wait();

    auto elapsed = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    CHECK(loader.IsLoaded(), "should be loaded after Wait() returns");
    std::cout << "  Wait blocked for " << elapsed << " ms (model loaded)" << std::endl;
    PASS();
}

// ---------------------------------------------------------
// Test 6: Double-load prevention
// ---------------------------------------------------------
void test_double_load_prevention() {
    TEST("double LoadAsync is ignored");
    ModelLoader loader;
    std::string model_dir = resolvePath("models");

    std::atomic<int> call_count{0};
    auto cb = [&](ncnn::Net*, float, float) { call_count++; };

    loader.LoadAsync(model_dir, cb);
    loader.LoadAsync(model_dir, cb); // should be silently ignored
    loader.Wait();

    CHECK(call_count.load() == 1, "callback should be called exactly once, got " + std::to_string(call_count.load()));
    PASS();
}

// ---------------------------------------------------------
// Test 7: Move semantics
// ---------------------------------------------------------
void test_move_semantics() {
    TEST("move semantics");
    ModelLoader loader1;
    std::string model_dir = resolvePath("models");

    std::atomic<bool> called{false};
    loader1.LoadAsync(model_dir, [&](ncnn::Net*, float, float) { called = true; });
    loader1.Wait();

    CHECK(loader1.IsLoaded(), "loader1 should be loaded");

    // Move construct
    ModelLoader loader2(std::move(loader1));
    CHECK(loader2.IsLoaded(), "loader2 should be loaded after move");
    CHECK(loader2.GetNet() != nullptr, "loader2 GetNet should be valid");
    CHECK(loader2.GetIOCostMs() > 0.0f, "loader2 cost should be preserved");

    // Move assign
    ModelLoader loader3;
    loader3 = std::move(loader2);
    CHECK(loader3.IsLoaded(), "loader3 should be loaded after move assign");
    CHECK(loader3.GetNet() != nullptr, "loader3 GetNet should be valid");
    PASS();
}

// ---------------------------------------------------------
// Test 8: Callback runs on background thread
// ---------------------------------------------------------
void test_callback_thread() {
    TEST("callback runs on background thread");
    ModelLoader loader;
    std::string model_dir = resolvePath("models");

    std::thread::id main_tid = std::this_thread::get_id();
    std::thread::id callback_tid;

    loader.LoadAsync(model_dir, [&](ncnn::Net*, float, float) {
        callback_tid = std::this_thread::get_id();
    });
    loader.Wait();

    CHECK(callback_tid != main_tid, "callback should run on a different thread");
    std::cout << "  main_tid=" << main_tid << " callback_tid=" << callback_tid << std::endl;
    PASS();
}

// ---------------------------------------------------------
// Test 9: SetWarmupShapes before load
// ---------------------------------------------------------
void test_custom_warmup_shapes() {
    TEST("custom warmup shapes before load");
    ModelLoader loader;
    std::string model_dir = resolvePath("models");

    // Set non-default shapes (larger audio, same face)
    loader.SetWarmupShapes(80, 80, 1, 96, 96, 6);

    std::atomic<bool> called{false};
    loader.LoadAsync(model_dir, [&](ncnn::Net* net, float io, float wu) {
        called = true;
        CHECK(net != nullptr, "net should be valid with custom shapes");
        std::cout << "  custom shapes — warmup: " << wu << " ms" << std::endl;
    });
    loader.Wait();
    CHECK(called.load(), "callback should be called");
    PASS();
}

// ---------------------------------------------------------
// Test 10: Explicit param/bin paths overload
// ---------------------------------------------------------
void test_explicit_paths() {
    TEST("explicit param/bin paths");
    ModelLoader loader;

    std::string model_dir = resolvePath("models");
    std::string param = model_dir + "/Wav2Lip-SD-GAN-opt.param";
    std::string bin   = model_dir + "/Wav2Lip-SD-GAN-opt.bin";

    std::atomic<bool> called{false};
    loader.LoadAsync(param, bin, [&](ncnn::Net* net, float, float) {
        called = true;
        CHECK(net != nullptr, "net should be valid with explicit paths");
    });
    loader.Wait();
    CHECK(loader.IsLoaded(), "IsLoaded should be true");
    CHECK(called.load(), "callback should be called");
    PASS();
}

// ---------------------------------------------------------
// main
// ---------------------------------------------------------
int main(int argc, char** argv) {
    std::cout << "===== ModelLoader Test Suite =====" << std::endl;

    // Check model files exist
    std::string model_dir = resolvePath("models");
    std::string param = model_dir + "/Wav2Lip-SD-GAN-opt.param";
    std::string bin   = model_dir + "/Wav2Lip-SD-GAN-opt.bin";

    bool model_available = fs::exists(param) && fs::exists(bin);
    std::cout << "Model: " << (model_available ? param : "NOT FOUND") << std::endl;
    std::cout << std::endl;

    // These tests don't require model files to exist
    test_nonexistent_dir();
    test_missing_param();
    test_missing_bin();

    if (model_available) {
        test_successful_load();
        test_wait_blocks();
        test_double_load_prevention();
        test_move_semantics();
        test_callback_thread();
        test_custom_warmup_shapes();
        test_explicit_paths();
    } else {
        std::cout << "\nSkipped model-dependent tests (models not found)." << std::endl;
    }

    std::cout << "\n===== Results: " << g_pass.load() << " passed, "
              << g_fail.load() << " failed =====" << std::endl;

    return g_fail.load() > 0 ? 1 : 0;
}
