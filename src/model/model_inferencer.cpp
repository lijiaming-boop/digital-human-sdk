#include "model/model_inferencer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include <gpu.h>
#include <net.h>

namespace {

// Only set defaults before the first OpenMP region; explicit process settings
// always take precedence over SDK defaults.
void ConfigureOpenMPPassiveWait() {
    if (std::getenv("OMP_WAIT_POLICY") == nullptr) {
#ifdef _WIN32
        _putenv_s("OMP_WAIT_POLICY", "PASSIVE");
#else
        setenv("OMP_WAIT_POLICY", "PASSIVE", 0);
#endif
    }
    // GCC/libgomp-specific: disable the default long barrier spin.
    if (std::getenv("GOMP_SPINCOUNT") == nullptr) {
#ifdef _WIN32
        _putenv_s("GOMP_SPINCOUNT", "0");
#else
        setenv("GOMP_SPINCOUNT", "0", 0);
#endif
    }
}

#if NCNN_VULKAN
// ncnn 的 Vulkan instance 是进程级对象。多个 ModelInferencer 可以并存，
// 因此以引用计数管理它，避免一个实例析构时破坏另一个实例正在使用的 GPU。
std::mutex g_vulkan_instance_mutex;
int g_vulkan_instance_users = 0;

bool AcquireVulkanInstance(int& device_index, int& device_count,
                           std::string& failure_reason) {
    std::lock_guard<std::mutex> lock(g_vulkan_instance_mutex);

    if (g_vulkan_instance_users == 0) {
        const int ret = ncnn::create_gpu_instance();
        if (ret != 0) {
            failure_reason = "create_gpu_instance failed (ret="
                           + std::to_string(ret) + ")";
            std::cerr << "[ModelInferencer] " << failure_reason << std::endl;
            return false;
        }
    }

    const int gpu_count = ncnn::get_gpu_count();
    device_count = gpu_count;
    if (gpu_count <= 0) {
        failure_reason = "Vulkan is compiled in, but no usable physical GPU was found";
        std::cerr << "[ModelInferencer] " << failure_reason << std::endl;
        if (g_vulkan_instance_users == 0) {
            ncnn::destroy_gpu_instance();
        }
        return false;
    }

    device_index = ncnn::get_default_gpu_index();
    ++g_vulkan_instance_users;
    return true;
}

void ReleaseVulkanInstance() {
    std::lock_guard<std::mutex> lock(g_vulkan_instance_mutex);
    if (g_vulkan_instance_users <= 0) {
        return;
    }

    --g_vulkan_instance_users;
    if (g_vulkan_instance_users == 0) {
        ncnn::destroy_gpu_instance();
    }
}
#endif

}  // namespace

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
    bool  use_gpu          = true;   ///< 是否启用 Vulkan GPU 加速
    float target_latency_ms = 50.0f; ///< 目标推理延迟阈值（毫秒）
    VulkanStatus vulkan_status;

    // ---- 模型 IO blob 名称（与 Wav2Lip-SD-GAN-opt.param 一致） ----
    static constexpr const char* kAudioInput = "audio_sequences";
    static constexpr const char* kFaceInput  = "face_sequences";
    static constexpr const char* kOutputName = "output";

    // ---- 默认输入形状（与 model_loader 保持一致） ----
    // 音频: 80 mel bins × 16 时间帧 (ncnn: w=时间, h=bins)
    static constexpr int kAudioW = 16;
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

    // Retain source paths so a pre-start thread-count change can rebuild ncnn
    // layers with the new load-time configuration.
    std::string param_path;
    std::string bin_path;

#if NCNN_VULKAN
    bool gpu_instance_acquired = false;
    int  gpu_device_index = -1;
#endif

    ~Impl() {
        // GPU 网络必须先销毁，才能销毁它依赖的进程级 Vulkan instance。
        net.clear();
        ReleaseVulkan();
    }

    void ReleaseVulkan() {
#if NCNN_VULKAN
        if (gpu_instance_acquired) {
            ReleaseVulkanInstance();
            gpu_instance_acquired = false;
            gpu_device_index = -1;
        }
#endif
    }

    bool EnsureVulkan() {
#if NCNN_VULKAN
        if (gpu_instance_acquired) {
            return true;
        }
        vulkan_status.compiled = true;
        std::string failure_reason;
        int device_count = 0;
        if (!AcquireVulkanInstance(gpu_device_index, device_count, failure_reason)) {
            vulkan_status.available = false;
            vulkan_status.enabled = false;
            vulkan_status.device_count = device_count;
            vulkan_status.device_index = -1;
            vulkan_status.message = failure_reason;
            return false;
        }
        gpu_instance_acquired = true;
        vulkan_status.device_count = device_count;
        vulkan_status.device_index = gpu_device_index;
        vulkan_status.message = "Vulkan device enumerated; real inference pending";
        return true;
#else
        vulkan_status.compiled = false;
        vulkan_status.available = false;
        vulkan_status.enabled = false;
        vulkan_status.message = "ncnn was built without Vulkan support";
        std::cerr << "[ModelInferencer] " << vulkan_status.message << std::endl;
        return false;
#endif
    }

    bool loadNet(int threads, bool gpu) {
        if (gpu && !EnsureVulkan()) {
            return false;
        }

        // CPU 回退前先释放 GPU 网络和 instance，确保不会留下仅修改 option
        // 但没有真正运行 GPU pipeline 的半初始化状态。
#if NCNN_VULKAN
        if (!gpu && gpu_instance_acquired) {
            net.clear();
            ReleaseVulkan();
        }
#endif

        net.clear();
        // Winograd/GEMM pipeline construction happens in load_param/load_model.
        // These options therefore must be assigned before either load call.
        net.opt.num_threads = gpu ? 1 : threads;
        net.opt.use_vulkan_compute = gpu;
#if NCNN_VULKAN
        if (gpu) {
            net.set_vulkan_device(gpu_device_index);
        }
#endif

        if (net.load_param(param_path.c_str()) != 0) {
            std::cerr << "[ModelInferencer] ?? param ??: "
                      << param_path << std::endl;
            net.clear();
            if (gpu) ReleaseVulkan();
            return false;
        }
        if (net.load_model(bin_path.c_str()) != 0) {
            std::cerr << "[ModelInferencer] ?? bin ??: "
                      << bin_path << std::endl;
            net.clear();
            if (gpu) ReleaseVulkan();
            return false;
        }
        return true;
    }

    bool WarmupCurrentNet() {
        ncnn::Mat audio_in(kAudioW, kAudioH, kAudioC);
        ncnn::Mat face_in(kFaceW, kFaceH, kFaceC);
        audio_in.fill(0.0f);
        face_in.fill(0.0f);

        ncnn::Extractor ex = net.create_extractor();
        ex.input(kAudioInput, audio_in);
        ex.input(kFaceInput, face_in);
        ncnn::Mat warmup_out;
        const int ret = ex.extract(kOutputName, warmup_out);
        if (ret != 0 || warmup_out.empty()) {
            std::cerr << "[ModelInferencer] warmup inference failed (ret=" << ret
                      << ", output_empty=" << warmup_out.empty() << ")" << std::endl;
            return false;
        }
        return true;
    }

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
        this->param_path = param_path;
        this->bin_path = bin_path;
#if NCNN_VULKAN
        vulkan_status.compiled = true;
#else
        vulkan_status.compiled = false;
#endif
        // This must run before the first OpenMP parallel region.
        ConfigureOpenMPPassiveWait();
        // Select Vulkan pipelines before loading the model; fall back on CPU.
        if (!loadNet(num_threads, use_gpu)) {
            if (!use_gpu || !loadNet(num_threads, false)) {
                return false;
            }
            use_gpu = false;
            std::cerr << "[ModelInferencer] GPU unavailable; falling back to CPU"
                      << std::endl;
        }

        // ---- warmup ----
        ncnn::Mat audio_in(kAudioW, kAudioH, kAudioC);
        ncnn::Mat face_in(kFaceW, kFaceH, kFaceC);
        audio_in.fill(0.0f);
        face_in.fill(0.0f);

        net.opt.num_threads = num_threads;
        ncnn::Extractor ex = net.create_extractor();
        ex.input(kAudioInput, audio_in);
        ex.input(kFaceInput, face_in);
        ncnn::Mat warmup_out;
        int ret = ex.extract(kOutputName, warmup_out);
        if ((ret != 0 || warmup_out.empty()) && use_gpu) {
            std::cerr << "[ModelInferencer] GPU warmup failed; falling back to CPU" << std::endl;
            vulkan_status.available = false;
            vulkan_status.enabled = false;
            vulkan_status.message = "Vulkan model warmup failed; CPU fallback selected";
            if (loadNet(num_threads, false)) {
                use_gpu = false;
                ncnn::Extractor cpu_ex = net.create_extractor();
                cpu_ex.input(kAudioInput, audio_in);
                cpu_ex.input(kFaceInput, face_in);
                ret = cpu_ex.extract(kOutputName, warmup_out);
            }
        }
        if (ret != 0 || warmup_out.empty()) {
            std::cerr << "[ModelInferencer] warmup 推理失败 (ret="
                      << ret << ")，模型可能不兼容" << std::endl;
            return false;
        }

#if NCNN_VULKAN
        if (use_gpu) {
            vulkan_status.available = true;
            vulkan_status.enabled = true;
            vulkan_status.message = "Vulkan device passed real Wav2Lip warmup inference";
            std::cout << "[ModelInferencer] Vulkan GPU enabled (device="
                      << gpu_device_index << ", total_devices="
                      << ncnn::get_gpu_count() << ")" << std::endl;
        }
#endif

        // ---- 自动调优 ----
        // Runtime-only thread tuning is invalid for Winograd/GEMM layers.
        // Use SetThreadCount before Start() to reload with a measured value.

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
    if (max_t < 1) max_t = 4;  // hardware_concurrency() 可能返回 0
    const int requested_threads = std::clamp(n, 1, max_t);
    if (requested_threads == impl_->num_threads) {
        return;
    }

    // Safe only while no Infer call is executing (normally before Start()).
    // Reload instead of changing net.opt so load-time convolution pipelines
    // receive the requested thread count.
    if (impl_->initialized && !impl_->use_gpu) {
        if (!impl_->loadNet(requested_threads, false)) {
            std::cerr << "[ModelInferencer] failed to reload model for "
                      << requested_threads << " threads" << std::endl;
            return;
        }
    }
    impl_->num_threads = requested_threads;

    // 启用 GPU 时线程数设置不生效
    if (impl_->use_gpu) {
        std::cout << "[ModelInferencer] 当前为 GPU 模式，线程数设置将在关闭 GPU 后生效"
                  << std::endl;
    }
}

bool ModelInferencer::IsGPUEnabled() const {
    return impl_->use_gpu;
}

VulkanStatus ModelInferencer::GetVulkanStatus() const {
    VulkanStatus status = impl_->vulkan_status;
    status.enabled = impl_->use_gpu;
    return status;
}

bool ModelInferencer::EnableGPU(bool enable) {
    if (!impl_->initialized) {
        std::cerr << "[ModelInferencer] GPU mode can only be changed after Init()"
                  << std::endl;
        return false;
    }

    if (enable) {
        if (impl_->use_gpu) {
            return true;
        }

        // GPU 模式必须重载网络。Winograd/GEMM/Vulkan pipeline 均在
        // load_param/load_model 阶段创建，不能只改运行时 option。
        if (!impl_->loadNet(impl_->num_threads, true)) {
            impl_->loadNet(impl_->num_threads, false);
            impl_->use_gpu = false;
            impl_->vulkan_status.enabled = false;
            return false;
        }

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
        if (ret != 0 || out.empty()) {
            impl_->loadNet(impl_->num_threads, false);
            std::cerr << "[ModelInferencer] GPU 模式不可用：Vulkan 推理失败"
                      << std::endl;
            impl_->use_gpu = false;
            impl_->vulkan_status.available = false;
            impl_->vulkan_status.enabled = false;
            impl_->vulkan_status.message = "Vulkan model inference failed; CPU fallback selected";
            return false;
        }
        impl_->use_gpu = true;
        impl_->vulkan_status.available = true;
        impl_->vulkan_status.enabled = true;
        impl_->vulkan_status.message = "Vulkan device passed real Wav2Lip inference";
        std::cout << "[ModelInferencer] GPU 加速已启用" << std::endl;
    } else {
        if (impl_->use_gpu && !impl_->loadNet(impl_->num_threads, false)) {
            std::cerr << "[ModelInferencer] failed to switch back to CPU" << std::endl;
            return false;
        }
        impl_->use_gpu = false;
        impl_->vulkan_status.enabled = false;
        if (impl_->vulkan_status.available) {
            impl_->vulkan_status.message = "Vulkan verified, currently disabled by caller";
        }
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
