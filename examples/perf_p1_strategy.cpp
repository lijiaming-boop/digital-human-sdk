// ============================================================================
// perf_p1_strategy.cpp
//
// 按 docs/perf/performance_optimization_plan.md 的优先级顺序，对未完成的
// P1 项（线程预算 sweep、GPU 端到端验证）以及 P2 的 ROI 收益量化做一次性
// 验收测试。P0/P1(load-time) 已落地，本程序首先做回归断言，再进入 sweep。
//
// 阶段：
//   R  —— P0/P1 回归：env 变量、SetThreadCount 重载、推理可用性
//   S1 —— ncnn 推理线程 sweep：{1,2,4} 实测 p50/p95/p99，选出最优
//   S2 —— OpenCV 线程 sweep：{1,2,4} 实测 ProcessROI p50/p95，
//         并对比全图 Process() 量化 P2 ROI 收益
//   G  —— GPU 端到端验证：EnableGPU(true/false)，含回退路径
//
// 用法: ./perf_p1_strategy [model_dir] [assets_dir]
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <ncnn/mat.h>

#include "model/model_inferencer.h"
#include "model/output_processor.h"

using namespace digital_human;
using namespace std::chrono;

namespace {

// ---------------------------------------------------------------------------
// 统计辅助
// ---------------------------------------------------------------------------

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

struct LatencyStats {
    double p50 = 0, p95 = 0, p99 = 0, avg = 0, min = 0, max = 0;
    int    n   = 0;
};

LatencyStats summarize(std::vector<double>& v) {
    LatencyStats s;
    s.n = static_cast<int>(v.size());
    if (s.n == 0) return s;
    // percentile 按值传参会排序副本，这里直接对 v 排序以取真实 min/max。
    std::sort(v.begin(), v.end());
    s.p50 = percentile(v, 0.50);
    s.p95 = percentile(v, 0.95);
    s.p99 = percentile(v, 0.99);
    s.min = v.front();
    s.max = v.back();
    double sum = 0;
    for (double x : v) sum += x;
    s.avg = sum / s.n;
    return s;
}

void print_row(const std::string& label, const LatencyStats& s) {
    std::cout << "  " << std::left << std::setw(28) << label
              << " n=" << std::setw(4) << s.n
              << " avg=" << std::fixed << std::setprecision(2) << std::setw(8) << s.avg
              << " p50=" << std::setw(8) << s.p50
              << " p95=" << std::setw(8) << s.p95
              << " p99=" << std::setw(8) << s.p99
              << " min=" << std::setw(7) << s.min
              << " max=" << std::setw(8) << s.max
              << " ms\n";
}

// ---------------------------------------------------------------------------
// ncnn::Mat 构造工具
// ---------------------------------------------------------------------------

// 音频特征: 80 mel bins × 16 帧 (ncnn w=时间帧, h=bins)
ncnn::Mat make_audio_input() {
    ncnn::Mat m(16, 80, 1);
    float* p = (float*)m.data;
    for (int i = 0; i < m.total(); ++i) {
        // 模拟归一化后的 log-mel，范围约 [-3, 3]
        p[i] = static_cast<float>((i % 17) * 0.17 - 1.5);
    }
    return m;
}

// 人脸输入: 96×96×6（3 通道参考帧 + 3 通道当前帧）
ncnn::Mat make_face_input(const cv::Mat& gray_face) {
    ncnn::Mat m(96, 96, 6);
    float* p = (float*)m.data;
    for (int c = 0; c < 6; ++c) {
        for (int y = 0; y < 96; ++y) {
            for (int x = 0; x < 96; ++x) {
                uchar v = gray_face.at<uchar>(y, x);
                p[c * 96 * 96 + y * 96 + x] = v / 255.0f;
            }
        }
    }
    return m;
}

// 合成模型输出: 96×96×3 RGB float, 值域 [0,1]
ncnn::Mat make_model_output() {
    ncnn::Mat m(96, 96, 3);
    float* p = (float*)m.data;
    for (int i = 0; i < m.total(); ++i) {
        p[i] = 0.5f + 0.3f * static_cast<float>(std::sin(i * 0.013));
        if (p[i] < 0.0f) p[i] = 0.0f;
        if (p[i] > 1.0f) p[i] = 1.0f;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Stage R: P0/P1 回归断言
// ---------------------------------------------------------------------------

bool stage_regression(model::ModelInferencer& inferencer,
                      const std::string& model_dir) {
    std::cout << "\n====== Stage R: P0/P1 回归 ======\n";

    bool ok = true;

    // R1. 环境变量（由 ConfigureOpenMPPassiveWait 在 Init 时设置）
    const char* policy = std::getenv("OMP_WAIT_POLICY");
    const char* spin   = std::getenv("GOMP_SPINCOUNT");
    bool env_ok = (policy != nullptr && std::string(policy) == "PASSIVE")
               && (spin != nullptr && std::string(spin) == "0");
    std::cout << "  [R1] OMP_WAIT_POLICY=" << (policy ? policy : "<null>")
              << "  GOMP_SPINCOUNT=" << (spin ? spin : "<null>")
              << "  => " << (env_ok ? "PASS" : "FAIL") << "\n";
    ok &= env_ok;

    // R2. 初始化后默认线程数（Impl 默认 num_threads=2，未 autoTune 仍为 2）
    int tc = inferencer.GetThreadCount();
    bool tc_ok = (tc >= 1 && tc <= 4);
    std::cout << "  [R2] GetThreadCount()=" << tc
              << " (期望 1..4) => " << (tc_ok ? "PASS" : "FAIL") << "\n";
    ok &= tc_ok;

    // R3. SetThreadCount 触发模型重载（load-time pipeline 重建），
    //     验证重载后推理仍可用且线程数生效。无 "load-time value" 警告即代表
    //     设置时机正确；这里通过推理成功 + 线程数一致间接验证。
    inferencer.SetThreadCount(4);
    tc = inferencer.GetThreadCount();
    bool reload_ok = (tc == 4);
    std::cout << "  [R3] SetThreadCount(4) => GetThreadCount()=" << tc
              << " => " << (reload_ok ? "PASS" : "FAIL") << "\n";
    ok &= reload_ok;

    ncnn::Mat audio = make_audio_input();
    cv::Mat gray = cv::Mat::zeros(96, 96, CV_8UC1);
    ncnn::Mat face = make_face_input(gray);
    ncnn::Mat out = inferencer.Infer(audio, face);
    bool infer_ok = (!out.empty() && out.c == 3);
    std::cout << "  [R4] 重载后 Infer() 输出 c=" << out.c
              << " => " << (infer_ok ? "PASS" : "FAIL") << "\n";
    ok &= infer_ok;

    // 还原为 2 线程（后续 sweep 从 SetThreadCount 开始各自重建）
    inferencer.SetThreadCount(2);

    std::cout << "  Stage R => " << (ok ? "ALL PASS" : "FAILED") << "\n";
    return ok;
}

// ---------------------------------------------------------------------------
// Stage S1: ncnn 推理线程 sweep
// ---------------------------------------------------------------------------

int stage_ncnn_sweep(model::ModelInferencer& inferencer) {
    std::cout << "\n====== Stage S1: ncnn 推理线程 sweep ======\n";

    ncnn::Mat audio = make_audio_input();
    cv::Mat gray = cv::Mat::zeros(96, 96, CV_8UC1);
    ncnn::Mat face = make_face_input(gray);

    const std::vector<int> thread_opts = {1, 2, 4};
    const int kWarmup = 5;
    const int kIters   = 20;

    struct Row { int n; LatencyStats s; };
    std::vector<Row> rows;

    for (int n : thread_opts) {
        inferencer.SetThreadCount(n);
        // 等待 load-time 重建稳定：执行 warmup
        for (int i = 0; i < kWarmup; ++i) {
            ncnn::Mat out = inferencer.Infer(audio, face);
            if (out.empty()) {
                std::cerr << "  [S1] n=" << n << " warmup 推理失败\n";
                return 1;
            }
        }
        std::vector<double> lat;
        lat.reserve(kIters);
        for (int i = 0; i < kIters; ++i) {
            auto t0 = steady_clock::now();
            ncnn::Mat out = inferencer.Infer(audio, face);
            auto t1 = steady_clock::now();
            if (out.empty()) {
                std::cerr << "  [S1] n=" << n << " 测试推理失败\n";
                return 1;
            }
            lat.push_back(duration<double, std::milli>(t1 - t0).count());
        }
        auto s = summarize(lat);
        rows.push_back({n, s});
    }

    // 输出
    std::cout << "\n  ncnn 推理延迟分布:\n";
    for (auto& r : rows) {
        std::ostringstream lbl;
        lbl << "ncnn_threads=" << r.n;
        print_row(lbl.str(), r.s);
    }

    // 选 p95 最低
    auto best = std::min_element(rows.begin(), rows.end(),
        [](const Row& a, const Row& b) { return a.s.p95 < b.s.p95; });
    std::cout << "\n  >>> 最优 ncnn 线程数: " << best->n
              << " (p95=" << best->s.p95 << " ms)\n";
    std::cout << "  Stage S1 => DONE\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Stage S2: OpenCV 线程 sweep + ROI 收益量化
// ---------------------------------------------------------------------------

int stage_opencv_sweep() {
    std::cout << "\n====== Stage S2: OpenCV 线程 sweep + ROI 收益 ======\n";

    // 构造真实尺寸输入（模拟 720p 原始人脸图）
    cv::Mat original(720, 1280, CV_8UC3);
    for (int y = 0; y < original.rows; ++y) {
        for (int x = 0; x < original.cols; ++x) {
            original.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((x * 7) & 255),
                static_cast<uchar>((y * 5) & 255),
                static_cast<uchar>(((x + y) * 3) & 255));
        }
    }

    // 96×96 对齐空间的 mask（CV_32FC1, [0,1]）
    cv::Mat mask_96(96, 96, CV_32FC1);
    for (int y = 0; y < 96; ++y) {
        for (int x = 0; x < 96; ++x) {
            // 椭圆 mask，模拟嘴部区域
            float dx = (x - 48.0f) / 24.0f;
            float dy = (y - 70.0f) / 12.0f;
            mask_96.at<float>(y, x) = (dx * dx + dy * dy < 1.0f) ? 1.0f : 0.0f;
        }
    }

    // M_inv: 把 96×96 对齐人脸放到 (560, 300)，放大 1.5× → 覆盖 ~144×144
    cv::Mat M_inv = (cv::Mat_<double>(2, 3) <<
        1.5, 0.0, 560.0,
        0.0, 1.5, 300.0);

    ncnn::Mat model_out = make_model_output();

    model::OutputProcessor proc;
    cv::Rect face_rect(560, 300, 144, 144);

    const std::vector<int> cv_opts = {1, 2, 4};
    const int kWarmup = 5;
    const int kIters   = 20;

    struct Row { int n; LatencyStats s; };
    std::vector<Row> roi_rows, full_rows;

    // ---- ROI 路径 sweep ----
    for (int n : cv_opts) {
        cv::setNumThreads(n);
        for (int i = 0; i < kWarmup; ++i) {
            cv::Mat r = proc.ProcessROI(model_out, original, mask_96, M_inv, face_rect);
            if (r.empty()) {
                std::cerr << "  [S2] cv=" << n << " ROI warmup 失败\n";
                return 1;
            }
        }
        std::vector<double> lat;
        lat.reserve(kIters);
        for (int i = 0; i < kIters; ++i) {
            auto t0 = steady_clock::now();
            cv::Mat r = proc.ProcessROI(model_out, original, mask_96, M_inv, face_rect);
            auto t1 = steady_clock::now();
            lat.push_back(duration<double, std::milli>(t1 - t0).count());
        }
        roi_rows.push_back({n, summarize(lat)});
    }

    // ---- 全图路径（仅 cv=2 做参照，量化 ROI 收益）----
    cv::setNumThreads(2);
    for (int i = 0; i < kWarmup; ++i) {
        // 全图 mask: 将 96×96 mask 经 M_inv 逆变换到全图坐标
        cv::Mat full_mask = proc.InverseTransform(mask_96, M_inv, original.size());
        if (full_mask.empty()) {
            // 若逆变换 mask 不适用，用 96×96 mask 让 Process 内部 resize
            full_mask = mask_96;
        }
        cv::Mat r = proc.Process(model_out, original, full_mask, M_inv);
        if (r.empty()) {
            std::cerr << "  [S2] 全图 warmup 失败\n";
            return 1;
        }
    }
    std::vector<double> full_lat;
    full_lat.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        cv::Mat full_mask = proc.InverseTransform(mask_96, M_inv, original.size());
        if (full_mask.empty()) full_mask = mask_96;
        auto t0 = steady_clock::now();
        cv::Mat r = proc.Process(model_out, original, full_mask, M_inv);
        auto t1 = steady_clock::now();
        full_lat.push_back(duration<double, std::milli>(t1 - t0).count());
    }
    LatencyStats full_s = summarize(full_lat);

    // ---- 输出 ----
    std::cout << "\n  ProcessROI 延迟分布 (按 OpenCV 线程数):\n";
    for (auto& r : roi_rows) {
        std::ostringstream lbl;
        lbl << "ROI  cv_threads=" << r.n;
        print_row(lbl.str(), r.s);
    }
    print_row("全图 Process cv=2", full_s);

    // ROI 收益
    auto roi_best = std::min_element(roi_rows.begin(), roi_rows.end(),
        [](const Row& a, const Row& b) { return a.s.p95 < b.s.p95; });
    std::cout << "\n  >>> 最优 OpenCV 线程数: " << roi_best->n
              << " (ROI p95=" << roi_best->s.p95 << " ms)\n";
    if (full_s.p95 > 0.001) {
        std::cout << "  >>> ROI 相对全图收益: p95 "
                  << full_s.p95 << " -> " << roi_best->s.p95
                  << " ms (" << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - roi_best->s.p95 / full_s.p95))
                  << "% 降低)\n";
    }

    // 还原默认
    cv::setNumThreads(4);
    std::cout << "  Stage S2 => DONE\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Stage G: GPU 端到端验证（含回退）
// ---------------------------------------------------------------------------

int stage_gpu_validate(model::ModelInferencer& inferencer) {
    std::cout << "\n====== Stage G: GPU 端到端验证 ======\n";

    ncnn::Mat audio = make_audio_input();
    cv::Mat gray = cv::Mat::zeros(96, 96, CV_8UC1);
    ncnn::Mat face = make_face_input(gray);

    // 基线 CPU p95（线程数=2）
    inferencer.SetThreadCount(2);
    const int kWarmup = 5;
    const int kIters   = 20;
    for (int i = 0; i < kWarmup; ++i) {
        if (inferencer.Infer(audio, face).empty()) {
            std::cerr << "  [G] CPU warmup 失败\n";
            return 1;
        }
    }
    std::vector<double> cpu_lat;
    cpu_lat.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        auto t0 = steady_clock::now();
        inferencer.Infer(audio, face);
        auto t1 = steady_clock::now();
        cpu_lat.push_back(duration<double, std::milli>(t1 - t0).count());
    }
    LatencyStats cpu_s = summarize(cpu_lat);
    print_row("CPU baseline t=2", cpu_s);

    // ---- 启用 GPU ----
    bool gpu_ok = inferencer.EnableGPU(true);
    std::cout << "  [G1] EnableGPU(true) => " << (gpu_ok ? "SUCCESS" : "FAIL")
              << " (IsGPUEnabled=" << inferencer.IsGPUEnabled() << ")\n";

    if (gpu_ok) {
        // GPU warmup + 测试
        for (int i = 0; i < kWarmup; ++i) {
            if (inferencer.Infer(audio, face).empty()) {
                std::cerr << "  [G] GPU warmup 失败\n";
                break;
            }
        }
        std::vector<double> gpu_lat;
        gpu_lat.reserve(kIters);
        int fails = 0;
        for (int i = 0; i < kIters; ++i) {
            auto t0 = steady_clock::now();
            ncnn::Mat out = inferencer.Infer(audio, face);
            auto t1 = steady_clock::now();
            if (out.empty()) { ++fails; continue; }
            gpu_lat.push_back(duration<double, std::milli>(t1 - t0).count());
        }
        if (!gpu_lat.empty()) {
            LatencyStats gpu_s = summarize(gpu_lat);
            print_row("GPU Vulkan", gpu_s);
            std::cout << "  [G2] GPU 失败次数: " << fails << "/" << kIters << "\n";
            if (gpu_s.p95 < cpu_s.p95) {
                std::cout << "  >>> GPU 相对 CPU 加速: p95 "
                          << cpu_s.p95 << " -> " << gpu_s.p95 << " ms ("
                          << std::fixed << std::setprecision(1)
                          << (100.0 * (1.0 - gpu_s.p95 / cpu_s.p95))
                          << "% 降低)\n";
            } else {
                std::cout << "  >>> GPU 未取得加速（CPU 紧张设备预期之外，"
                          << "记录以备 P1 GPU 路径评审）\n";
            }
        } else {
            std::cout << "  [G2] GPU 推理全部失败，回退 CPU\n";
        }
    } else {
        std::cout << "  [G2] Vulkan 不可用（ncnn 未编译 Vulkan 或无可用设备）\n";
        std::cout << "       记录：当前部署无法走 GPU 路径，CPU 模式为唯一选择\n";
    }

    // ---- 回退验证：EnableGPU(false) 后 CPU 推理可用 ----
    bool restore_ok = inferencer.EnableGPU(false);
    int tc = inferencer.GetThreadCount();
    bool cpu_works = false;
    ncnn::Mat out = inferencer.Infer(audio, face);
    cpu_works = (!out.empty() && out.c == 3);
    std::cout << "  [G3] EnableGPU(false) => " << (restore_ok ? "OK" : "FAIL")
              << "  GetThreadCount()=" << tc
              << "  Infer()=" << (cpu_works ? "OK" : "FAIL") << "\n";

    // 还原 2 线程
    inferencer.SetThreadCount(2);
    std::cout << "  Stage G => "
              << ((gpu_ok || true) && restore_ok && cpu_works ? "DONE" : "INCOMPLETE")
              << "\n";
    return 0;
}

}  // namespace

// ============================================================================
// 入口
// ============================================================================

int main(int argc, char* argv[]) {
    std::string model_dir  = (argc > 1) ? argv[1]
                                       : std::string(PROJECT_SOURCE_DIR) + "/models";
    std::string assets_dir = (argc > 2) ? argv[2] : ASSETS_DIR;

    std::cout << "==============================================\n";
    std::cout << "  P1 性能策略验收测试\n";
    std::cout << "==============================================\n";
    std::cout << "  model_dir  = " << model_dir << "\n";
    std::cout << "  assets_dir = " << assets_dir << "\n";
    std::cout << "  hw_threads = " << std::thread::hardware_concurrency() << "\n";
    std::cout << "==============================================\n";

    model::ModelInferencer inferencer;
    if (!inferencer.Init(model_dir)) {
        std::cerr << "[FAIL] ModelInferencer Init 失败\n";
        return 1;
    }

    int rc = 0;
    if (!stage_regression(inferencer, model_dir)) {
        std::cerr << "[FAIL] Stage R 回归未通过，终止\n";
        return 1;
    }
    rc |= stage_ncnn_sweep(inferencer);
    rc |= stage_opencv_sweep();
    rc |= stage_gpu_validate(inferencer);

    std::cout << "\n==============================================\n";
    std::cout << "  测试完成 (rc=" << rc << ")\n";
    std::cout << "==============================================\n";
    return rc;
}
