# 推理线程架构

## 1. 概述

推理线程 (`InferenceWorker`) 是流水线的核心计算节点。它从输入队列获取 `InferenceTask`（包含 Mel 特征和对齐人脸），执行 Wav2Lip 模型前向传播，将生成的唇形同步图像推送到渲染队列。

## 2. 架构

```
MelFeatureQueue ──┐
                  ├──► InferenceWorker (推理线程)
ProcessedFaceQueue ─┘       │
                            ├── 1. 获取 InferenceTask
                            ├── 2. Mel cv::Mat → ncnn::Mat
                            ├── 3. Face cv::Mat → ncnn::Mat (6通道)
                            ├── 4. ModelInferencer.Infer()
                            ├── 5. 失败重试 (最多3次)
                            └── 6. Push InferenceOutputPacket → RenderQueue
```

## 3. 推理任务

```cpp
struct InferenceTask {
    MelFeaturePacket      mel;          ///< Mel 频谱特征 (80×T cv::Mat)
    ProcessedFacePacket   face;         ///< 对齐人脸 + 遮罩 + M_inv
    int                   retry_count = 0;   ///< 当前重试次数
    int64_t               enqueue_time_ms;   ///< 入队时间（用于队列积压检测）
    
    static constexpr int kMaxRetries = 3;
    bool CanRetry() const { return retry_count < kMaxRetries; }
};
```

## 4. 推理前张量转换

```cpp
// 梅尔频谱: cv::Mat (rows=T, cols=80, CV_32F) → ncnn::Mat (w=80, h=T, c=1)
ncnn::Mat MelToNCNN(const cv::Mat& mel);

// 对齐人脸: cv::Mat (96×96 3ch) → ncnn::Mat (w=96, h=96, c=6)
// 前 3 通道 = 人脸 BGR, 后 3 通道 = 人脸 BGR 副本（Wav2Lip 要求）
ncnn::Mat FaceToNCNN(const cv::Mat& face);
```

## 5. 推理失败重试

```
Infer():
  1. 执行 ModelInferencer.Infer()
  2. 成功 → 推送结果
  3. 失败 → retry_count++
     a. 若 CanRetry(): 将 Task 放回队列头部，记录日志
     b. 否则: 记录失败日志，丢弃 Task
```

## 6. 队列积压检测

每秒检查输入队列深度：
- 深度 < 10: 正常
- 深度 10-30: 发出 WARN 日志
- 深度 > 30: 发出 ERROR 日志，触发反压

## 7. 性能测量

每次推理记录：
- 推理延迟 (ms)
- 队列等待时间 (ms)
- 输出到滑动窗口 EWMA

## 8. 线程退出

遵循 ThreadBase 生命周期：
```
Start() → Run() 循环 → Stop() → IsStopping()=true → 退出
```
退出前确保：
1. 当前推理完成
2. 未推送的结果已丢弃
3. 输出队列 Push EOS
