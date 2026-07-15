#pragma once

#include <memory>
#include <string>
#include <ncnn/mat.h>

namespace digital_human {
namespace model {

/**
 * @brief Wav2Lip-SD-GAN 模型推理器
 *
 * 负责加载 Wav2Lip 模型，接收 InputProcessor 处理好的音频梅尔频谱特征和
 * 人脸图像数据（均为 ncnn::Mat），运行前向推理计算，输出口型同步人脸图像。
 *
 * 关键特性：
 * - ncnn::Net 在 Init 时一次性加载，整个生命周期复用，避免频繁创建销毁
 * - ncnn::Extractor 每次 Infer 时创建（轻量级，可多次创建）
 * - Init 时自动运行 benchmark，遍历线程数 [1, 2, 4, 8, 16] 选最优
 * - 若 CPU 最佳延迟仍 > target_latency_ms 且 ncnn 有 Vulkan 支持，自动启用 GPU
 * - 内置性能计数器（总推理次数、平均/最大/最小延迟）
 *
 * 采用 PIMPL（Pointer to Implementation）模式，隐藏内部实现细节。
 */
class ModelInferencer {
public:
    /// @brief 构造一个未初始化的推理器
    ModelInferencer();

    /// @brief 析构函数（在 .cpp 中完成，确保 Impl 完整类型可用）
    ~ModelInferencer();

    /// @brief 禁止拷贝构造
    ModelInferencer(const ModelInferencer&) = delete;

    /// @brief 禁止拷贝赋值
    ModelInferencer& operator=(const ModelInferencer&) = delete;

    /// @brief 允许移动构造
    ModelInferencer(ModelInferencer&&) noexcept;

    /// @brief 允许移动赋值
    ModelInferencer& operator=(ModelInferencer&&) noexcept;

    // ==================== 初始化 ====================

    /**
     * @brief 从模型目录初始化推理器（自动拼接 .param/.bin 路径）
     *
     * 执行流程：
     *   1. 加载模型参数和权重文件
     *   2. 运行一次 warmup 推理
     *   3. 自动 benchmark 各线程配置，选择最优设置
     *
     * @param model_dir 模型文件所在目录，应包含 Wav2Lip-SD-GAN-opt.param/.bin
     * @return true  初始化成功，模型就绪
     * @return false 初始化失败（文件不存在、加载失败等）
     */
    bool Init(const std::string& model_dir);

    /**
     * @brief 从显式路径初始化推理器
     *
     * @param param_path .param 文件的完整路径
     * @param bin_path   .bin 文件的完整路径
     * @return true  初始化成功
     * @return false 初始化失败
     */
    bool Init(const std::string& param_path, const std::string& bin_path);

    /// @brief 检查推理器是否已成功初始化
    bool IsInitialized() const;

    // ==================== 推理 ====================

    /**
     * @brief 运行单帧推理
     *
     * @param audio_feat 预处理后的音频梅尔特征，ncnn::Mat 形状 (w=mel_bins, h=frames, c=1)
     * @param face_input 预处理后的人脸图像，ncnn::Mat 形状 (w=96, h=96, c=6)
     * @return ncnn::Mat 输出口型同步人脸图像（3通道），空 Mat 表示推理失败
     */
    ncnn::Mat Infer(const ncnn::Mat& audio_feat, const ncnn::Mat& face_input);

    // ==================== 性能自动调优 ====================

    /// @brief 获取当前使用的 CPU 线程数
    int GetThreadCount() const;

    /**
     * @brief 手动设置 CPU 线程数（覆盖 autoTune 的结果）
     *
     * 设置后立即生效，后续 Infer 调用将使用新线程数。
     * 取值范围 [1, max_threads]，超出自动 clamp。
     *
     * @param n 目标线程数
     */
    void SetThreadCount(int n);

    /// @brief 检查当前是否启用了 GPU（Vulkan）加速
    bool IsGPUEnabled() const;

    /**
     * @brief 手动启用或禁用 GPU（Vulkan）加速
     *
     * @param enable true=开启 Vulkan, false=关闭
     * @return true  设置成功（ncnn 支持 Vulkan）
     * @return false 设置失败（ncnn 未编译 Vulkan 支持）
     */
    bool EnableGPU(bool enable);

    /// @brief 获取目标推理延迟阈值（毫秒），用于 autoTune 判定
    float GetTargetLatencyMs() const;

    /// @brief 设置目标推理延迟阈值
    void SetTargetLatencyMs(float ms);

    // ==================== 性能统计 ====================

    /// @brief 获取累计推理次数
    int64_t GetInferenceCount() const;

    /// @brief 获取平均推理延迟（毫秒）
    float GetAvgLatencyMs() const;

    /// @brief 获取最小推理延迟（毫秒）
    float GetMinLatencyMs() const;

    /// @brief 获取最大推理延迟（毫秒）
    float GetMaxLatencyMs() const;

    /**
     * @brief 重置所有性能计数器
     *
     * 清零推理次数、总延迟、最小/最大延迟记录。
     */
    void ResetStats();

    /**
     * @brief 打印性能统计信息到 stdout
     *
     * 输出格式：
     *   [ModelInferencer] === Stats ===
     *   [ModelInferencer]   Inference count: N
     *   [ModelInferencer]   Avg latency: X.XX ms
     *   [ModelInferencer]   Min latency: X.XX ms
     *   [ModelInferencer]   Max latency: X.XX ms
     *   [ModelInferencer]   Threads: N | GPU: yes/no
     */
    void PrintStats() const;

private:
    /// @brief PIMPL 模式：前向声明实现结构体
    struct Impl;

    /// @brief 指向实现的唯一指针
    std::unique_ptr<Impl> impl_;
};

}  // namespace model
}  // namespace digital_human
