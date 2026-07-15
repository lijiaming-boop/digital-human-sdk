#include "model/model_inferencer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include <ncnn/net.h>

namespace digital_human {
namespace model {

// ============================================================================
// 内部辅助类型
// ============================================================================

/// @brief Benchmark 结果
struct BenchmarkResult {
    int   num_threads  = 1;    ///< 使用的线程数
    bool  use_gpu      = false;///< 是否启用 GPU
    float avg_latency_ms = 0.0f; ///< 平均推理延迟（毫秒）
    bool  success      = false; ///< benchmark 是否成功完成
};

// ============================================================================
// Impl 结构体：封装 ModelInferencer 的所有内部实现细节
// ============================================================================
struct ModelInferencer::Impl {
    // ---- 模型网络（一次加载，全程复用） ----
    ncnn::Net net;

    // ---- 初始化状态 ----
    bool initialized = false;

    // ---- 性能配置 ----
    int   num_threads      = 2;      ///< CPU 线程数（autoTune 后确定）
    bool  use_gpu          = false;  ///< 是否启用 Vulkan GPU 加速
    float target_latency_ms = 50.0f; ///< 目标推理延迟阈值（毫秒）

    // ---- 模型 IO blob 名称（与 Wav2Lip-SD-GAN-opt.param 一致） ----
    static constexpr const char* kAudioInput = "audio_sequences";
    static constexpr const char* kFaceInput  = "face_sequences";
    static constexpr const char* kOutputName = "output";

    // ---- 默认输入形状（与 model_loader 保持一致） ----
    static constexpr int kAudioW = 80;
    static constexpr int kAudioH = 80;
    static constexpr int kAudioC = 1;
    static constexpr int kFaceW  = 96;
    static constexpr int kFaceH  = 96;
    static constexpr int kFaceC  = 6;

    // ---- Benchmark 参数 ----
    static constexpr int kBenchmarkWarmupIters = 5;    ///< 预热迭代数
    static constexpr int kBenchmarkTestIters   = 10;   ///< 测试迭代数

    // ---- 可尝试的线程配置 ----
    static constexpr int kThreadOptions[] = {1, 2, 4, 8, 16};

    // ---- 性能计数器 ----
    std::atomic<int64_t> inference_count{0};
    std::mutex  stats_mutex;                    ///< 保护延迟统计
    double      total_latency_ms   = 0.0;       ///< 累计延迟（毫秒）
    float       min_latency_ms     = std::numeric_limits<float>::max();
    float       max_latency_ms     = 0.0f;

    // ========================================================================
    // 初始化
    // ========================================================================

    /**
     * @brief 执行真正的初始化流程
     *
     * @param param_path .param 文件路径
     * @param bin_path   .bin 文件路径
     * @return true  初始化成功
     */
    bool doInit(const std::string& param_path, const std::string& bin_path) {
        // ---- 加载模型 ----
        if (net.load_param(param_path.c_str()) != 0) {
            std::cerr << "[ModelInferencer] 加载 param 失败: " << param_path << std::endl;
            return false;
        }
        if (net.load_model(bin_path.c_str()) != 0) {
            std::cerr << "[ModelInferencer] 加载 bin 失败: " << bin_path << std::endl;
            return false;
        }

        // ---- warmup ----
        ncnn::Mat audio_in(kAudioW, kAudioH, kAudioC);
        ncnn::Mat face_in(kFaceW, kFaceH, kFaceC);
        audio_in.fill(0.0f);
        face_in.fill(0.0f);

        net.opt.num_threads = 2;
        ncnn::Extractor ex = net.create_extractor();
        ex.input(kAudioInput, audio_in);
        ex.input(kFaceInput, face_in);
        ncnn::Mat warmup_out;
        int ret = ex.extract(kOutputName, warmup_out);
        if (ret != 0) {
            std::cerr << "[ModelInferencer] warmup 推理失败 (ret="
                      << ret << ")，模型可能不兼容" << std::endl;
            return false;
        }

        // ---- 自动调优 ----
        autoTune();

        initialized = true;
        return true;
    }

    // ========================================================================
    // 自动线程 / GPU 调优
    // ========================================================================

    /**
     * @brief 自动 benchmark 各线程配置，选择最优设置
     *
     * 遍历线程数 [1, 2, 4, 8, 16]，分别测试推理延迟。
     * 选择延迟最低且满足 target_latency_ms 的配置。
     * 若 CPU 所有配置均不达标且 Vulkan 可用，尝试 GPU 加速。
     */
    void autoTune() {
        std::vector<BenchmarkResult> results;

        // ---- CPU benchmark ----
        for (int t : kThreadOptions) {
            // 不超过硬件线程数
            int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (t > hw_threads) continue;

            BenchmarkResult r = benchmark(t, false);
            if (r.success) {
                results.push_back(r);
                std::cout << "[ModelInferencer]   CPU " << t << "线程: "
                          << r.avg_latency_ms << " ms"
                          << (r.avg_latency_ms <= target_latency_ms ? " ✓" : "")
                          << std::endl;
            }
        }

        if (results.empty()) {
            std::cerr << "[ModelInferencer] benchmark 全部失败，使用默认 2 线程"
                      << std::endl;
            num_threads = 2;
            return;
        }

        // ---- 选最优 CPU 配置 ----
        // 首选满足目标延迟的配置中延迟最低的
        // 若无满足的，选延迟最低的
        auto best = results[0];
        for (const auto& r : results) {
            bool best_meets = best.avg_latency_ms <= target_latency_ms;
            bool curr_meets = r.avg_latency_ms <= target_latency_ms;
            if (curr_meets && !best_meets) {
                best = r;  // 当前达标而 best 不达标，选当前
            } else if (curr_meets == best_meets) {
                // 达标状态相同，选延迟更低的
                if (r.avg_latency_ms < best.avg_latency_ms) {
                    best = r;
                }
            }
        }

        num_threads = best.num_threads;
        std::cout << "[ModelInferencer] CPU 最优: " << num_threads
                  << " 线程, " << best.avg_latency_ms << " ms"
                  << (best.avg_latency_ms <= target_latency_ms ? " ✓" : " ✗")
                  << std::endl;

        // ---- GPU 回退 ----
        if (best.avg_latency_ms > target_latency_ms) {
            // CPU 不达标，尝试 GPU 加速
            // 直接运行 GPU benchmark 检测 Vulkan 可用性
            std::cout << "[ModelInferencer] CPU 未达标，尝试 GPU 加速..." << std::endl;
            BenchmarkResult gpu_r = benchmark(1, true);
            if (gpu_r.success) {
                std::cout << "[ModelInferencer]   GPU: "
                          << gpu_r.avg_latency_ms << " ms"
                          << (gpu_r.avg_latency_ms <= target_latency_ms ? " ✓" : " ✗")
                          << std::endl;
                if (gpu_r.avg_latency_ms <= best.avg_latency_ms ||
                    gpu_r.avg_latency_ms <= target_latency_ms) {
                    use_gpu = true;
                    num_threads = 1;  // GPU 模式下线程数无关紧要
                    std::cout << "[ModelInferencer] 启用 GPU 加速" << std::endl;
                }
            } else {
                std::cout << "[ModelInferencer] Vulkan GPU 不可用或推理失败，保持 CPU 模式"
                          << std::endl;
            }
        }

        // ---- 同步 net.opt 到最终选择的配置 ----
        net.opt.num_threads = num_threads;
        net.opt.use_vulkan_compute = use_gpu;
    }

    /**
     * @brief 在指定配置下运行 benchmark
     *
     * @param threads CPU 线程数
     * @param gpu     是否启用 Vulkan
     * @return BenchmarkResult 包含平均延迟和成功标志
     */
    BenchmarkResult benchmark(int threads, bool gpu) {
        BenchmarkResult result;
        result.num_threads = threads;
        result.use_gpu     = gpu;

        // 保存当前 opt 设置，结束后恢复
        int  saved_threads = net.opt.num_threads;
        bool saved_gpu     = net.opt.use_vulkan_compute;

        // 使用随机输入数据
        ncnn::Mat audio_in(kAudioW, kAudioH, kAudioC);
        ncnn::Mat face_in(kFaceW, kFaceH, kFaceC);
        audio_in.fill(0.0f);
        face_in.fill(0.0f);

        // ---- 预热 ----
        for (int i = 0; i < kBenchmarkWarmupIters; ++i) {
            net.opt.use_vulkan_compute = gpu;
            net.opt.num_threads = gpu ? 1 : threads;
            ncnn::Extractor ex = net.create_extractor();
            ex.input(kAudioInput, audio_in);
            ex.input(kFaceInput, face_in);
            ncnn::Mat out;
            ex.extract(kOutputName, out);
        }

        // ---- 正式测试 ----
        std::vector<float> latencies;
        latencies.reserve(kBenchmarkTestIters);

        for (int i = 0; i < kBenchmarkTestIters; ++i) {
            net.opt.use_vulkan_compute = gpu;
            net.opt.num_threads = gpu ? 1 : threads;
            ncnn::Extractor ex = net.create_extractor();
            ex.input(kAudioInput, audio_in);
            ex.input(kFaceInput, face_in);

            auto t0 = std::chrono::steady_clock::now();
            ncnn::Mat out;
            int ret = ex.extract(kOutputName, out);
            auto t1 = std::chrono::steady_clock::now();

            if (ret != 0) {
                // 恢复 opt 设置
                net.opt.num_threads = saved_threads;
                net.opt.use_vulkan_compute = saved_gpu;
                return result;
            }

            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            latencies.push_back(ms);
        }

        // ---- 恢复 opt 设置 ----
        net.opt.num_threads = saved_threads;
        net.opt.use_vulkan_compute = saved_gpu;

        // ---- 计算平均延迟 ----
        float sum = 0.0f;
        for (float ms : latencies) {
            sum += ms;
        }
        result.avg_latency_ms = sum / static_cast<float>(latencies.size());
        result.success = true;
        return result;
    }

    // ========================================================================
    // 推理
    // ========================================================================

    /**
     * @brief 执行单帧推理
     *
     * @param audio 预处理后的音频梅尔特征
     * @param face  预处理后的人脸图像
     * @return ncnn::Mat 输出图像，空 Mat 表示失败
     */
    ncnn::Mat doInfer(const ncnn::Mat& audio, const ncnn::Mat& face) {
        // ---- 输入校验 ----
        if (audio.empty() || face.empty()) {
            std::cerr << "[ModelInferencer] 推理失败：输入数据为空" << std::endl;
            return ncnn::Mat();
        }

        // ---- 创建提取器 ----
        net.opt.num_threads = num_threads;
        net.opt.use_vulkan_compute = use_gpu;
        ncnn::Extractor ex = net.create_extractor();

        // ---- 注入输入 ----
        ex.input(kAudioInput, audio);
        ex.input(kFaceInput, face);

        // ---- 推理 + 计时 ----
        auto t0 = std::chrono::steady_clock::now();
        ncnn::Mat result;
        int ret = ex.extract(kOutputName, result);
        auto t1 = std::chrono::steady_clock::now();

        if (ret != 0) {
            std::cerr << "[ModelInferencer] 推理失败 (ret=" << ret << ")" << std::endl;
            return ncnn::Mat();
        }

        // ---- 更新性能计数器 ----
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        inference_count.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            total_latency_ms += ms;
            if (ms < min_latency_ms) min_latency_ms = ms;
            if (ms > max_latency_ms) max_latency_ms = ms;
        }

        return result;
    }
};

// ============================================================================
// ModelInferencer 公有接口实现
// ============================================================================

ModelInferencer::ModelInferencer()
    : impl_(std::make_unique<Impl>()) {}

ModelInferencer::~ModelInferencer() = default;

ModelInferencer::ModelInferencer(ModelInferencer&&) noexcept = default;

ModelInferencer& ModelInferencer::operator=(ModelInferencer&&) noexcept = default;

// ==================== 初始化 ====================

bool ModelInferencer::Init(const std::string& model_dir) {
    // 自动拼接 .param/.bin 路径
    std::string param_path = model_dir + "/Wav2Lip-SD-GAN-opt.param";
    std::string bin_path   = model_dir + "/Wav2Lip-SD-GAN-opt.bin";
    return Init(param_path, bin_path);
}

bool ModelInferencer::Init(const std::string& param_path,
                           const std::string& bin_path) {
    if (impl_->initialized) {
        std::cerr << "[ModelInferencer] 重复初始化，跳过" << std::endl;
        return true;
    }
    bool ok = impl_->doInit(param_path, bin_path);
    if (!ok) {
        std::cerr << "[ModelInferencer] 初始化失败" << std::endl;
    }
    return ok;
}

bool ModelInferencer::IsInitialized() const {
    return impl_->initialized;
}

// ==================== 推理 ====================

ncnn::Mat ModelInferencer::Infer(const ncnn::Mat& audio_feat,
                                 const ncnn::Mat& face_input) {
    if (!impl_->initialized) {
        std::cerr << "[ModelInferencer] 推理失败：未初始化，请先调用 Init()" << std::endl;
        return ncnn::Mat();
    }
    return impl_->doInfer(audio_feat, face_input);
}

// ==================== 性能自动调优 ====================

int ModelInferencer::GetThreadCount() const {
    return impl_->num_threads;
}

void ModelInferencer::SetThreadCount(int n) {
    int max_t = static_cast<int>(std::thread::hardware_concurrency());
    impl_->num_threads = std::clamp(n, 1, max_t);

    // 启用 GPU 时线程数设置不生效
    if (impl_->use_gpu) {
        std::cout << "[ModelInferencer] 当前为 GPU 模式，线程数设置将在关闭 GPU 后生效"
                  << std::endl;
    }
}

bool ModelInferencer::IsGPUEnabled() const {
    return impl_->use_gpu;
}

bool ModelInferencer::EnableGPU(bool enable) {
    if (enable) {
        // 尝试检测 Vulkan 支持
        // 创建一个临时提取器检测 GPU 推理
        impl_->net.opt.use_vulkan_compute = true;
        impl_->net.opt.num_threads = 1;
        ncnn::Extractor ex = impl_->net.create_extractor();

        ncnn::Mat test_audio(impl_->kAudioW, impl_->kAudioH, impl_->kAudioC);
        ncnn::Mat test_face(impl_->kFaceW, impl_->kFaceH, impl_->kFaceC);
        test_audio.fill(0.0f);
        test_face.fill(0.0f);
        ex.input(impl_->kAudioInput, test_audio);
        ex.input(impl_->kFaceInput, test_face);

        ncnn::Mat out;
        int ret = ex.extract(impl_->kOutputName, out);
        if (ret != 0) {
            std::cerr << "[ModelInferencer] GPU 模式不可用：Vulkan 推理失败"
                      << std::endl;
            impl_->use_gpu = false;
            return false;
        }
        impl_->use_gpu = true;
        std::cout << "[ModelInferencer] GPU 加速已启用" << std::endl;
    } else {
        impl_->use_gpu = false;
        std::cout << "[ModelInferencer] GPU 加速已关闭，使用 CPU 推理" << std::endl;
    }
    return true;
}

float ModelInferencer::GetTargetLatencyMs() const {
    return impl_->target_latency_ms;
}

void ModelInferencer::SetTargetLatencyMs(float ms) {
    impl_->target_latency_ms = std::max(ms, 1.0f);
}

// ==================== 性能统计 ====================

int64_t ModelInferencer::GetInferenceCount() const {
    return impl_->inference_count.load(std::memory_order_relaxed);
}

float ModelInferencer::GetAvgLatencyMs() const {
    int64_t count = impl_->inference_count.load(std::memory_order_relaxed);
    if (count == 0) return 0.0f;
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return static_cast<float>(impl_->total_latency_ms / count);
}

float ModelInferencer::GetMinLatencyMs() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return (impl_->min_latency_ms == std::numeric_limits<float>::max())
               ? 0.0f
               : impl_->min_latency_ms;
}

float ModelInferencer::GetMaxLatencyMs() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return impl_->max_latency_ms;
}

void ModelInferencer::ResetStats() {
    impl_->inference_count.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    impl_->total_latency_ms = 0.0;
    impl_->min_latency_ms   = std::numeric_limits<float>::max();
    impl_->max_latency_ms   = 0.0f;
}

void ModelInferencer::PrintStats() const {
    int64_t count = GetInferenceCount();
    std::cout << "[ModelInferencer] === 性能统计 ===" << std::endl;
    std::cout << "[ModelInferencer]   推理次数:     " << count << std::endl;
    if (count > 0) {
        std::cout << "[ModelInferencer]   平均延迟:     "
                  << GetAvgLatencyMs() << " ms" << std::endl;
        std::cout << "[ModelInferencer]   最小延迟:     "
                  << GetMinLatencyMs() << " ms" << std::endl;
        std::cout << "[ModelInferencer]   最大延迟:     "
                  << GetMaxLatencyMs() << " ms" << std::endl;
    }
    std::cout << "[ModelInferencer]   线程数:       " << GetThreadCount() << std::endl;
    std::cout << "[ModelInferencer]   GPU 加速:     "
              << (IsGPUEnabled() ? "yes" : "no") << std::endl;
    std::cout << "[ModelInferencer] ===================" << std::endl;
}

}  // namespace model
}  // namespace digital_human
