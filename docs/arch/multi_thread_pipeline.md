# 多线程处理架构技术方案

## 1. 概述

本文档定义 Digital Human SDK 的多线程处理架构。该架构负责编排音频处理、视频处理、模型推理、音视频同步和最终渲染的整个流水线。

### 1.1 现有模块全景

```
音频流水线:
  AudioLoader → NoiseReduction → AudioFramer → VAD → PreEmphasis
    → RMSNormalize → MelFeatureExtract → CMVN

视频流水线:
  ImageLoader → FaceDetector → FaceAlignigner → FaceMaskGenerator

推理与输出:
  ModelLoader → ModelInferencer (Wav2Lip) → OutputProcessor

同步与调度:
  AudioPlayer (PortAudio) → AVSync → FrameScheduler → AudioSyncScheduler
```

### 1.2 设计目标

| 维度 | 目标 |
|------|------|
| 吞吐量 | 达到 25+ FPS 实时处理 |
| 延迟 | 端到端延迟 < 200ms |
| 线程安全 | 无数据竞争，无死锁 |
| 优雅退出 | 停止信号后 500ms 内完成清理 |
| 可观测性 | 每阶段延迟、队列深度可监控 |

---

## 2. 线程模型

### 2.1 线程职责

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline 架构总览                           │
│                                                                 │
│  AudioProducer ──► AudioProcess ──► ┌──────────────────┐       │
│      (Thread 1)      (Thread 2)     │   SyncPoint      │       │
│                                      │  (匹配 AV 时间戳) │       │
│  VideoProducer ──► VideoProcess ──► └──────┬───────────┘       │
│      (Thread 3)      (Thread 4)            │                   │
│                                            ▼                   │
│                                    InferenceWorker              │
│                                      (Thread 5)                │
│                                            │                   │
│                                            ▼                   │
│                                   OutputProcessor               │
│                                      (Thread 6)                │
│                                            │                   │
│                                            ▼                   │
│                                    RenderThread                 │
│                                      (Thread 7)                │
│                                            │                   │
│                                            ▼                   │
│                                    AudioPlayback                │
│                                  (PortAudio 回调线程)           │
└─────────────────────────────────────────────────────────────────┘
```

| 线程 | 名称 | 职责 | 输入队列 | 输出队列 |
|------|------|------|----------|----------|
| T1 | `AudioProducer` | 读取音频数据 (文件/流) | — | `AudioRawQueue` |
| T2 | `AudioProcessor` | 音频特征提取流水线 | `AudioRawQueue` | `MelFeatureQueue` |
| T3 | `VideoProducer` | 读取视频帧 (文件/相机) | — | `VideoRawQueue` |
| T4 | `VideoProcessor` | 人脸检测/对齐/遮罩 | `VideoRawQueue` | `ProcessedFaceQueue` |
| T5 | `InferenceWorker` | Wav2Lip 模型推理 | `MelFeatureQueue` + `ProcessedFaceQueue` | `InferenceOutputQueue` |
| T6 | `OutputProcessor` | 输出后处理 (融合/锐化) | `InferenceOutputQueue` | `OutputFrameQueue` |
| T7 | `RenderThread` | 音视频同步 + 帧调度 + 渲染 | `OutputFrameQueue` | — |

> **AudioPlayback** 由 PortAudio 内部管理，通过回调驱动，不属于 Pipeline 线程池。

### 2.2 线程优先级建议

| 线程 | 优先级 | 理由 |
|------|--------|------|
| AudioProducer | 高 | 音频欠载会导致卡顿 |
| AudioPlayback (PortAudio) | 最高 (实时) | 音频中断可闻 |
| RenderThread | 高 | 帧率稳定性关键 |
| InferenceWorker | 普通 | CPU 密集，优先级高反而阻塞其他线程 |
| VideoProducer/Processor | 普通 | |

---

## 3. 线程间通信

### 3.1 ThreadSafeQueue — 无锁变体

所有阶段间通信均通过 `ThreadSafeQueue<T>`，内部使用 `std::mutex` + `std::condition_variable`。

```cpp
template <typename T>
class ThreadSafeQueue {
public:
    void Push(T item);
    bool TryPop(T& item);                  // 非阻塞
    bool WaitAndPop(T& item, int timeout_ms = -1);  // 阻塞带超时
    void Stop();                            // 唤醒所有等待者，标记停止
    size_t Size() const;
    bool Empty() const;
    bool IsStopped() const;
    void Clear();                           // 清空队列

private:
    mutable std::mutex      mutex_;
    std::queue<T>           queue_;
    std::condition_variable cv_;
    bool                    stopped_ = false;
};
```

### 3.2 数据包定义

每个阶段传递的数据包裹统一的时间戳和元数据：

```cpp
/// @brief 通用数据包头
struct PacketHeader {
    int64_t pts_ms;        ///< 呈现时间戳（毫秒）
    int64_t seq_id;        ///< 单调递增序列号
    int64_t source_id;     ///< 数据源标识
    StatusCode status;     ///< 处理状态
};

/// @brief 各阶段数据包
using AudioRawPacket      = Packet<std::vector<float>>;        // PCM float 数据
using MelFeaturePacket    = Packet<cv::Mat>;                   // Mel 频谱特征
using VideoFramePacket    = Packet<cv::Mat>;                   // 原始视频帧
using ProcessedFacePacket = Packet<ProcessedFaceData>;         // 对齐+遮罩
using InferenceInput      = std::pair<MelFeaturePacket, ProcessedFacePacket>;
using InferenceOutputPacket = Packet<ncnn::Mat>;
using OutputFramePacket   = Packet<cv::Mat>;
```

### 3.3 队列容量与反压

| 队列 | 最大长度 | 背压策略 |
|------|---------|----------|
| `AudioRawQueue` | 500ms 音频 | 丢弃旧数据（覆盖写） |
| `VideoRawQueue` | 30 帧 | 丢弃旧帧 |
| `MelFeatureQueue` | 60 帧 | 阻塞 Push |
| `ProcessedFaceQueue` | 30 帧 | 阻塞 Push |
| `InferenceOutputQueue` | 30 帧 | 阻塞 Push |
| `OutputFrameQueue` | 10 帧 | 丢弃旧帧 |

---

## 4. 同步机制

### 4.1 AV 时间戳匹配（SyncPoint）

`InferenceWorker` 需要同时持有匹配的 Mel 特征和人脸数据。匹配策略：

```
算法：最小时间戳差匹配
1. 从 MelFeatureQueue 取出一帧 audio_feat (pts_a)
2. 从 ProcessedFaceQueue 取出一帧 face_data (pts_v)
3. 计算 drift = pts_v - pts_a
4. 如果 |drift| ≤ threshold (40ms@25fps):
   → 配对成功，送入推理
5. 如果 drift > threshold (视频超前):
   → 丢弃 face_data，取下一帧重试
6. 如果 drift < -threshold (音频超前):
   → 丢弃 audio_feat，取下一帧重试
```

### 4.2 RenderThread 同步

渲染线程使用 `AudioSyncScheduler`（已实现）做最终同步判定：

```
每个输出帧:
  ScheduleResult = AudioSyncScheduler.ScheduleFrame(frame_id, pts_ms)
  switch (result.action):
    DISPLAY    → 显示帧
    DROP       → 丢弃帧，继续
    DUPLICATE  → 重复上一帧
    WAIT       → 等待 result.wait_time_ms
```

---

## 5. 消息协议与序列化

### 5.1 内部消息格式

```cpp
template <typename T>
struct Packet {
    PacketHeader header;
    T            payload;

    bool IsValid() const { return header.status == StatusCode::OK; }
    bool IsEOS()   const { return header.status == StatusCode::EOS; }
    bool IsError() const { return header.status == StatusCode::ERROR; }

    static Packet<T> EOS() {
        Packet<T> pkt;
        pkt.header.status = StatusCode::EOS;
        return pkt;
    }
};

enum class StatusCode : int8_t {
    OK      = 0,
    ERROR   = 1,    ///< 可恢复错误
    FATAL   = 2,    ///< 致命错误，触发停止
    EOS     = 3,    ///< 流结束 (End of Stream)
    SKIP    = 4,    ///< 跳过此帧（同步丢弃）
};
```

### 5.2 ProcessedFaceData 结构

```cpp
struct ProcessedFaceData {
    cv::Mat                    aligned_face;   ///< 96×96 对齐人脸 (BGR uint8)
    cv::Mat                    M_inv;           ///< 逆仿射变换矩阵 2×3
    cv::Mat                    face_mask;       ///< 口唇遮罩 (CV_32FC1)
    cv::Mat                    original_face;   ///< 原始人脸裁剪 (BGR uint8)
    cv::Rect                   face_rect;       ///< 人脸在原图中的矩形
    std::vector<cv::Point2f>   landmarks_96;    ///< 96x96 空间的关键点
};
```

---

## 6. 死锁分析与预防

### 6.1 死锁四大条件

| 条件 | 本架构 | 预防措施 |
|------|--------|----------|
| 互斥 | 存在 (mutex) | 无法消除，但严格控制范围 |
| 持有并等待 | 不存在 | 每个线程最多持有一个锁 |
| 非抢占 | 存在 | 使用超时 `WaitAndPop(timeout_ms)` |
| 循环等待 | 不存在 | 数据流为 DAG，无环 |

### 6.2 锁层级

```
层级 0: ThreadSafeQueue::mutex_   (每个队列独立)
```

所有线程只持有 **一个** 队列锁，从不嵌套加锁。推理线程需要从两个队列取数据：

```
// ✅ 安全：先取 Mel，再取 Face，每次只持一个锁
auto mel = melQueue_.WaitAndPop(timeout);
if (!mel) continue;
auto face = faceQueue_.WaitAndPop(timeout);
if (!face) continue;
```

### 6.3 活锁预防

队列 `WaitAndPop` 使用指数退避超时：

```
timeout = min(initial_ms * 2^retry, max_ms)
initial_ms = 1ms, max_ms = 100ms
```

---

## 7. 线程生命周期管理

### 7.1 状态机

```
INIT ──► RUNNING ──► STOPPING ──► STOPPED
                    │
                    └──► ERROR (FATAL)
```

| 状态 | 说明 |
|------|------|
| `INIT` | 线程创建，资源分配 |
| `RUNNING` | 主循环处理数据 |
| `STOPPING` | 收到停止信号，处理完当前项后退出 |
| `STOPPED` | 线程已 Join，资源已释放 |
| `ERROR` | 发生致命错误 |

### 7.2 启动序列

```
1. 创建所有队列
2. 启动 Consumer 线程（从下游到上游）:
   a. RenderThread
   b. OutputProcessor
   c. InferenceWorker
   d. VideoProcessor
   e. AudioProcessor
3. 启动 Producer 线程:
   a. VideoProducer
   b. AudioProducer
4. 等待 Producer 发送 EOS
```

### 7.3 停止序列

```
1. 调用 Pipeline::Stop()
2. 标记所有队列为 stopped
3. 等待所有线程 Join（带超时 2000ms）
4. 清空队列
5. 释放资源
```

### 7.4 EOS 传播机制

当 Producer 读取完所有数据后，向输出队列推送 `Packet::EOS()`。
下游线程收到 EOS 后，处理完当前帧，继续向下游转发 EOS，然后退出。

```
AudioProducer → AudioRawQueue → AudioProcessor → MelFeatureQueue → ...
                                ↑ EOS                              ↑ EOS
```

---

## 8. 错误处理策略

### 8.1 错误分类

| 类型 | 示例 | 处理方式 |
|------|------|----------|
| 可恢复错误 | 单帧人脸检测失败 | 跳过该帧，`Packet(StatusCode::SKIP)` |
| 资源错误 | 模型加载失败 | `Packet(StatusCode::FATAL)` → 触发 Pipeline::Stop() |
| 超时错误 | 队列等待超时 | 重试，超 3 次后 FATAL |
| 线程异常 | 处理函数抛出未捕获异常 | `std::current_exception()` → FATAL → 上游 Stop |

### 8.2 异常安全

```cpp
void AudioProcessor::Run() {
    try {
        while (!IsStopping()) {
            auto pkt = inputQueue_.WaitAndPop(kPopTimeoutMs);
            if (!pkt || pkt->IsEOS()) break;
            auto result = ProcessOne(*pkt);
            outputQueue_.Push(std::move(result));
        }
    } catch (const std::exception& e) {
        LogError("AudioProcessor fatal: {}", e.what());
        outputQueue_.Push(Packet<T>::Fatal(e.what()));
        Stop();  // 触发本线程和上游停止
    }
}
```

---

## 9. 性能设计

### 9.1 内存管理

- **预分配**：`ThreadSafeQueue` 底层使用 `std::deque` 避免频繁扩容
- **移动语义**：`Packet` 中的 `cv::Mat` 和 `ncnn::Mat` 通过 `std::move` 传递，零拷贝
- **共享帧池**：`VideoFramePacket` 使用引用计数 `cv::Mat`，避免帧数据复制

### 9.2 缓存优化

- 队列容量限制防止消费者跟不上时无限堆积
- `VideoRawQueue` 使用有界队列 + 丢弃策略，保证始终处理最新帧

### 9.3 延迟关键路径

实时场景下最关键的延迟路径：

```
AudioProducer → AudioProcessor → InferenceWorker → OutputProcessor → RenderThread
```

该路径上的队列应配置较小容量（低 latency）和较高优先级。

---

## 10. 可观测性

### 10.1 监控指标

```cpp
struct PipelineMetrics {
    // 每线程统计
    std::array<ThreadMetrics, 7> threads;
    
    // 队列深度
    std::array<size_t, 7> queue_depths;
    
    // 端到端延迟
    double avg_e2e_latency_ms;
    double p95_latency_ms;
    
    // 帧统计
    int64_t frames_processed;
    int64_t frames_dropped;
    double  actual_fps;
};

struct ThreadMetrics {
    std::string name;
    int64_t     items_processed;
    double      avg_process_time_ms;
    double      max_process_time_ms;
    int64_t     errors;
};
```

### 10.2 日志关键点

每个线程在处理每帧时输出 TRACE 级别日志，在帧率异常时输出 WARN。

---

## 11. 文件结构

```
include/core/
├── pipeline.h                  # MediaPipeline 主类
├── thread_safe_queue.h         # ThreadSafeQueue<T>
├── packet.h                    # Packet<T> + 数据结构定义
├── thread_base.h               # ThreadBase 基类 (生命周期管理)

src/core/
├── pipeline.cpp                # MediaPipeline 实现
├── thread_safe_queue.cpp       # (模板实例化)
├── thread_base.cpp             # ThreadBase 实现

include/core/workers/
├── audio_producer.h            # AudioProducer
├── audio_processor.h           # AudioProcessor  
├── video_producer.h            # VideoProducer
├── video_processor.h           # VideoProcessor
├── inference_worker.h          # InferenceWorker
├── output_processor_worker.h   # OutputProcessorWorker
├── render_worker.h             # RenderWorker

src/core/workers/
├── audio_producer.cpp
├── audio_processor.cpp
├── video_producer.cpp
├── video_processor.cpp
├── inference_worker.cpp
├── output_processor_worker.cpp
├── render_worker.cpp

tests/
├── test_thread_safe_queue.cpp
├── test_pipeline.cpp
├── test_packet.cpp
```

---

## 12. 实现顺序

| 阶段 | 内容 | 依赖 |
|------|------|------|
| Phase 1 | `ThreadSafeQueue<T>` + `Packet<T>` | — |
| Phase 2 | `ThreadBase` 基类 | Phase 1 |
| Phase 3 | 各 Worker 实现 | Phase 2 |
| Phase 4 | `MediaPipeline` 编排 | Phase 3 |
| Phase 5 | 测试用例 | Phase 4 |

---

## 13. 测试策略

### 13.1 单元测试

| 测试 | 覆盖 |
|------|------|
| `ThreadSafeQueue` 单生产者单消费者 | 正确入/出队，顺序保证 |
| `ThreadSafeQueue` 多生产者多消费者 | 线程安全，无数据竞争 |
| `ThreadSafeQueue` Stop 信号 | WaitAndPop 在停止后立即返回 |
| `Packet` 状态管理 | EOS/FATAL/SKIP 传播 |
| `ThreadBase` 生命周期 | INIT→RUNNING→STOPPING→STOPPED 状态转换 |

### 13.2 集成测试

| 测试 | 覆盖 |
|------|------|
| 两阶段流水线 (Producer → Processor) | 50ms 帧间隔，1000 帧 |
| 完整流水线 | 模拟音视频输入，验证帧输出 |
| 异常恢复 | 中间阶段抛出异常，验证下游收到 FATAL |
| 优雅退出 | 运行中 Stop()，验证 500ms 内退出 |
| 反压测试 | Processor 慢于 Producer，验证队列背压 |

### 13.3 压力测试

| 测试 | 条件 |
|------|------|
| 高吞吐 | 1000 帧音频+视频，验证无数据丢失 |
| 长时间运行 | 10000 帧，验证无内存泄漏 |
| 多消费者争用 | 4 个 InferenceWorker 竞争消费 |
