// ============================================================================
// perf_baseline_compare.cpp
//
// 在同一进程中对比 P0 优化前后两条路径：
//   - 路径 A（baseline，模拟优化前）：
//       每帧完整跑 dlib detect + landmarks + align + mask（无缓存）
//       每帧 frame.clone() 深拷贝
//   - 路径 B（optimized，P0 后）：
//       首帧跑完整流水线，后续命中缓存（指纹一致）
//       original_face = frame 浅拷贝
//   - 路径 C（baseline PaceFrame）：
//       sleep_for(milliseconds(int(wait_ms)))，模拟优化前的截断定时
//   - 路径 D（optimized PaceFrame）：
//       sleep_until + 末段 yield 忙等
// ============================================================================
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"

using namespace digital_human;
using namespace std::chrono;

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(p * (v.size() - 1))];
}

void print_stats(const std::string& label, std::vector<double>& v) {
    if (v.empty()) return;
    double sum = 0;
    for (double x : v) sum += x;
    std::cout << "  " << label << " (n=" << v.size() << "):\n";
    std::cout << "    min: " << *std::min_element(v.begin(), v.end()) << " ms\n";
    std::cout << "    p50: " << percentile(v, 0.50) << " ms\n";
    std::cout << "    p95: " << percentile(v, 0.95) << " ms\n";
    std::cout << "    p99: " << percentile(v, 0.99) << " ms\n";
    std::cout << "    max: " << *std::max_element(v.begin(), v.end()) << " ms\n";
    std::cout << "    avg: " << sum / v.size() << " ms\n";
    std::cout << "    sum: " << sum << " ms\n";
}

// ============================================================================
// B1: VideoProcessor 路径对比（缓存 vs 无缓存）
// ============================================================================
int benchmark_video_processor(const std::string& assets_dir,
                              const std::string& model_dir,
                              int num_frames = 300) {
    std::cout << "\n====== B1: VideoProcessor 缓存效果对比 ======\n";
    std::cout << "[INFO] 帧数: " << num_frames << "\n";

    cv::Mat face_img = cv::imread(assets_dir + "/face.jpg");
    if (face_img.empty()) {
        std::cerr << "[FAIL] 加载 face.jpg 失败\n";
        return 1;
    }
    std::cout << "[INFO] 人脸图像: " << face_img.cols << "x" << face_img.rows << "\n";

    core::FaceDetector detector;
    if (!detector.loadModel(model_dir + "/face")) {
        std::cerr << "[FAIL] dlib 模型加载失败\n";
        return 1;
    }
    core::FaceAligner aligner;
    core::FaceMaskGenerator mask_gen;

    // ========== 路径 A: Baseline（无缓存，每帧完整流水线 + clone） ==========
    std::vector<double> baseline_latencies;
    baseline_latencies.reserve(num_frames);

    for (int i = 0; i < num_frames; ++i) {
        auto t0 = steady_clock::now();

        // 完整 dlib 流水线（模拟优化前）
        auto faces = detector.detect(face_img);
        if (!faces.empty()) {
            auto max_face = std::max_element(faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b) {
                    return a.area() < b.area();
                });
            auto landmarks = detector.getLandmarks(face_img, *max_face);
            if (!landmarks.empty()) {
                std::vector<cv::Point2f> lmf;
                for (auto& p : landmarks) lmf.emplace_back((float)p.x, (float)p.y);
                auto ar = aligner.alignByRect(face_img, lmf, 96, *max_face);
                if (ar.valid) {
                    auto mask = mask_gen.generatePreciseMouthAlphaMask96(ar.landmarks);
                    // 模拟优化前：original_face = frame.clone()
                    cv::Mat original_copy = face_img.clone();
                    (void)original_copy;  // 防止编译器优化掉
                }
            }
        }

        baseline_latencies.push_back(
            duration<double, std::milli>(steady_clock::now() - t0).count());
    }

    // ========== 路径 B: Optimized（P0：缓存 + 浅拷贝） ==========
    struct FaceCache {
        cv::Size frame_size{0, 0};
        std::vector<uchar> signature;
        cv::Mat aligned_face, M_inv, face_mask;
        cv::Rect face_rect;
        std::vector<cv::Point2f> landmarks_96;
        bool valid = false;
    } cache;

    auto compute_sig = [](const cv::Mat& frame) {
        std::vector<uchar> sig;
        sig.reserve(16 * 3 * frame.channels());
        auto sample_row = [&](int y) {
            const uchar* row = frame.ptr<uchar>(y);
            for (int i = 0; i < 16; ++i) {
                int x = (i * frame.cols) / 16;
                const uchar* p = row + x * frame.channels();
                for (int c = 0; c < frame.channels(); ++c) sig.push_back(p[c]);
            }
        };
        sample_row(0);
        sample_row(frame.rows / 2);
        sample_row(frame.rows - 1);
        return sig;
    };

    std::vector<double> optimized_latencies;
    optimized_latencies.reserve(num_frames);
    int cache_hits = 0;

    for (int i = 0; i < num_frames; ++i) {
        auto t0 = steady_clock::now();

        auto sig = compute_sig(face_img);
        bool hit = cache.valid &&
                   cache.frame_size == face_img.size() &&
                   cache.signature.size() == sig.size() &&
                   std::memcmp(cache.signature.data(), sig.data(), sig.size()) == 0;

        if (hit) {
            // 缓存命中：仅浅拷贝
            cv::Mat original_ref = face_img;  // 浅拷贝
            (void)original_ref;
            cache_hits++;
        } else {
            // 缓存未命中：完整 dlib 流水线
            auto faces = detector.detect(face_img);
            if (!faces.empty()) {
                auto max_face = std::max_element(faces.begin(), faces.end(),
                    [](const cv::Rect& a, const cv::Rect& b) {
                        return a.area() < b.area();
                    });
                auto landmarks = detector.getLandmarks(face_img, *max_face);
                if (!landmarks.empty()) {
                    std::vector<cv::Point2f> lmf;
                    for (auto& p : landmarks) lmf.emplace_back((float)p.x, (float)p.y);
                    auto ar = aligner.alignByRect(face_img, lmf, 96, *max_face);
                    if (ar.valid) {
                        auto mask = mask_gen.generatePreciseMouthAlphaMask96(ar.landmarks);
                        cache.frame_size = face_img.size();
                        cache.signature = std::move(sig);
                        cache.aligned_face = ar.aligned_face;
                        cache.M_inv = ar.M_inv;
                        cache.face_mask = mask;
                        cache.face_rect = ar.face_rect;
                        cache.landmarks_96 = ar.landmarks;
                        cache.valid = true;
                    }
                }
            }
            // 模拟优化后：original_face = frame 浅拷贝
            cv::Mat original_ref = face_img;
            (void)original_ref;
        }

        optimized_latencies.push_back(
            duration<double, std::milli>(steady_clock::now() - t0).count());
    }

    // ========== 报告 ==========
    std::cout << "\n--- 路径 A: Baseline (无缓存 + 深拷贝) ---\n";
    print_stats("每帧耗时", baseline_latencies);

    std::cout << "\n--- 路径 B: Optimized (P0: 缓存 + 浅拷贝) ---\n";
    std::cout << "[INFO] 缓存命中: " << cache_hits << " / " << num_frames
              << " (" << (100.0 * cache_hits / num_frames) << "%)\n";
    print_stats("每帧耗时", optimized_latencies);

    // 对比加速比
    double base_sum = 0, opt_sum = 0;
    for (double v : baseline_latencies) base_sum += v;
    for (double v : optimized_latencies) opt_sum += v;
    std::cout << "\n--- 对比 ---\n";
    std::cout << "  Baseline 总耗时:    " << base_sum << " ms\n";
    std::cout << "  Optimized 总耗时:   " << opt_sum << " ms\n";
    std::cout << "  加速比:             " << base_sum / std::max(opt_sum, 0.001) << "x\n";
    std::cout << "  节省时间:           " << (base_sum - opt_sum) << " ms ("
              << (100.0 * (base_sum - opt_sum) / base_sum) << "%)\n";

    return 0;
}

// ============================================================================
// B2: PaceFrame 定时精度对比（截断 sleep_for vs sleep_until + 忙等）
// ============================================================================
int benchmark_pace_frame(double target_fps = 30.0, int num_frames = 150) {
    std::cout << "\n====== B2: PaceFrame 定时精度对比 (" << target_fps
              << " fps) ======\n";

    double target_interval_ms = 1000.0 / target_fps;

    // ========== 路径 C: Baseline (sleep_for + 整毫秒截断) ==========
    std::vector<double> baseline_intervals;
    baseline_intervals.reserve(num_frames);

    auto last = steady_clock::now();
    bool has_last = false;
    for (int i = 0; i < num_frames; ++i) {
        auto now = steady_clock::now();
        if (!has_last) { last = now; has_last = true; continue; }

        double elapsed = duration<double, std::milli>(now - last).count();
        double wait_ms = target_interval_ms - elapsed;
        if (wait_ms > 0) {
            // 模拟优化前：sleep_for + 整毫秒截断
            std::this_thread::sleep_for(
                milliseconds(static_cast<int>(wait_ms)));
        }
        auto t = steady_clock::now();
        baseline_intervals.push_back(duration<double, std::milli>(t - last).count());
        last = t;
    }

    // ========== 路径 D: Optimized (sleep_until + 末段忙等) ==========
    std::vector<double> optimized_intervals;
    optimized_intervals.reserve(num_frames);

    last = steady_clock::now();
    has_last = false;
    for (int i = 0; i < num_frames; ++i) {
        auto now = steady_clock::now();
        if (!has_last) { last = now; has_last = true; continue; }

        auto interval_us = microseconds(static_cast<int64_t>(target_interval_ms * 1000.0));
        auto target_time = last + interval_us;
        if (now < target_time) {
            auto busy_threshold = target_time - milliseconds(1);
            if (now < busy_threshold) {
                std::this_thread::sleep_until(busy_threshold);
            }
            while (steady_clock::now() < target_time) {
                std::this_thread::yield();
            }
        }
        auto t = steady_clock::now();
        optimized_intervals.push_back(duration<double, std::milli>(t - last).count());
        last = t;
    }

    std::cout << "[INFO] 目标间隔: " << target_interval_ms << " ms\n";

    std::cout << "\n--- 路径 C: Baseline (sleep_for + 整ms截断) ---\n";
    print_stats("实际间隔", baseline_intervals);
    double base_jitter = 0;
    for (double v : baseline_intervals) base_jitter += std::abs(v - target_interval_ms);
    std::cout << "    平均抖动: " << base_jitter / baseline_intervals.size() << " ms\n";
    double base_avg = 0;
    for (double v : baseline_intervals) base_avg += v;
    base_avg /= baseline_intervals.size();
    std::cout << "    实际 fps: " << 1000.0 / base_avg << "\n";

    std::cout << "\n--- 路径 D: Optimized (sleep_until + yield忙等) ---\n";
    print_stats("实际间隔", optimized_intervals);
    double opt_jitter = 0;
    for (double v : optimized_intervals) opt_jitter += std::abs(v - target_interval_ms);
    std::cout << "    平均抖动: " << opt_jitter / optimized_intervals.size() << " ms\n";
    double opt_avg = 0;
    for (double v : optimized_intervals) opt_avg += v;
    opt_avg /= optimized_intervals.size();
    std::cout << "    实际 fps: " << 1000.0 / opt_avg << "\n";

    std::cout << "\n--- 对比 ---\n";
    std::cout << "  抖动降低: " << (base_jitter / baseline_intervals.size())
              << " -> " << (opt_jitter / optimized_intervals.size())
              << " ms (" << (100.0 * (1 - opt_jitter / base_jitter)) << "% 降低)\n";

    return 0;
}

int main(int argc, char* argv[]) {
    std::string assets_dir = (argc > 1) ? argv[1] : ASSETS_DIR;
    std::string model_dir = std::string(PROJECT_SOURCE_DIR) + "/models";

    std::cout << "==============================================\n";
    std::cout << "  P0 优化前后对比基准\n";
    std::cout << "==============================================\n";
    std::cout << "  assets_dir = " << assets_dir << "\n";
    std::cout << "  model_dir  = " << model_dir << "\n";
    std::cout << "==============================================\n";

    benchmark_video_processor(assets_dir, model_dir, 300);
    benchmark_pace_frame(30.0, 150);
    benchmark_pace_frame(25.0, 150);

    std::cout << "\n==============================================\n";
    std::cout << "  对比结束\n";
    std::cout << "==============================================\n";
    return 0;
}
