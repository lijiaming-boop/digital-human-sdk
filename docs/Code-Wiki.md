# Digital Human SDK — Code Wiki

> 基于 Wav2Lip-SD-GAN 的数字人口型同步 SDK，提供完整的音视频处理流水线：
> 音频特征提取、人脸检测对齐、模型推理、口唇融合渲染。

- **项目名称**：Digital Human SDK
- **版本**：0.1.0
- **语言/标准**：C++17
- **构建系统**：CMake ≥ 3.16
- **目标平台**：Ubuntu 22.04 / Windows WSL2
- **命名空间**：`digital_human::<module>::`（snake_case 模块名）
- **封装范式**：Pimpl（Pointer to Implementation）模式

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [模块职责详解](#4-模块职责详解)
   - 4.1 [音频处理模块（audio）](#41-音频处理模块audio)
   - 4.2 [核心编排模块（core）](#42-核心编排模块core)
   - 4.3 [模型模块（model）](#43-模型模块model)
5. [关键类与函数说明](#5-关键类与函数说明)
6. [数据流与线程模型](#6-数据流与线程模型)
7. [依赖关系](#7-依赖关系)
8. [构建与运行方式](#8-构建与运行方式)
9. [配置参数参考](#9-配置参数参考)
10. [测试体系](#10-测试体系)
11. [编码规范与设计模式](#11-编码规范与设计模式)

---

## 1. 项目概述

### 1.1 项目定位

Digital Human SDK 是一套**端到端的数字人口型同步解决方案**，核心能力为：

- 输入一段 PCM 音频与对应视频帧序列
- 实时提取音频 Mel 特征并匹配视频帧中的人脸
- 通过 Wav2Lip-SD-GAN 模型推理生成口型同步人脸
- 将生成结果与原始人脸融合、逆变换并渲染输出
- 以音频为主时钟（Audio Master Clock）实现帧级音视频同步

### 1.2 设计目标

| 目标 | 实现手段 |
|------|---------|
| 实时性 | 7 阶段流水线 + 多线程并行；RingBuffer 解耦 PortAudio 回调 |
| 可扩展性 | 模块化设计、Pimpl 封装、统一 Packet 模板 |
| 健壮性 | 队列超时、心跳检测、推理重试、反压机制 |
| 可移植性 | 不使用 `-march=native`，Debug/Release 编译选项分离 |
| 可测试性 | 模块独立可测，五级测试金字塔 |

---

## 2. 整体架构

### 2.1 分层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层（examples/）                     │
│         demo_app / full_pipeline_test / 各模块 *_test        │
└──────────────────────────────┬──────────────────────────────┘
                               │ 依赖
┌──────────────────────────────▼──────────────────────────────┐
│              编排层（digital_human::core::Pipeline）         │
│  统一调度 AudioProcessor / VideoProcessor / InferenceWorker / │
│  RenderThread / AVMatcher，并通过 ThreadSafeQueue 解耦         │
└──────────────────────────────┬──────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   audio 模块     │   │   core 模块     │   │   model 模块     │
│  (PCM → Mel)    │   │ (检测/对齐/同步) │   │ (Wav2Lip 推理)   │
└─────────────────┘   └─────────────────┘   └─────────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                   ┌───────────────────────────────┐
                   │     第三方依赖层               │
                   │ OpenCV / FFmpeg / ncnn / dlib  │
                   │ PortAudio / OpenMP             │
                   └───────────────────────────────┘
```

### 2.2 数据流全景

```
音频侧:  PCM float
   → NoiseReduction(谱减法)
   → AudioFramer(frame=400 / hop=160)
   → VoiceActivityDetector(能量+过零率)
   → PreEmphasis(α=0.97)
   → RMSNormalize(target=0.056)
   → MelFeatureExtract(FFT=512, Mel=80)
   → CMVN(倒谱均值方差归一化)
   → ncnn::Mat (80×T×1)

视频侧:  cv::Mat BGR
   → FaceDetector(dlib HOG + 68 关键点)
   → FaceAligner(仿射对齐到 96×96)
   → FaceMaskGenerator(口唇遮罩 CV_32FC1)
   → ProcessedFaceData

推理侧:  Mel + AlignedFace
   → ModelInferencer(Wav2Lip-SD-GAN, auto-tune)
   → ncnn::Mat 输出 (448×96 RGB float)

渲染侧:  InferenceOutputPacket
   → OutputProcessor
        → OutputToMat (RGB float → BGR uint8)
        → InverseTransform (M_inv 逆仿射)
        → FaceFusion (mask alpha 混合)
        → Sharpen (USM 非锐化)
        → ColorBlend (色彩融合)
   → FrameScheduler(DISPLAY / DROP / DUPLICATE / WAIT)
   → OutputFramePacket (cv::Mat)
```

### 2.3 线程模型

```
AudioProducer ─► AudioProcessor ─► ┌─ AVMatcher ─┐
                                   │  (PTS 匹配)  │
VideoProducer ─► VideoProcessor ─►└──────────────┘
                                          │
                                          ▼
                                    InferenceWorker
                                          │
                                          ▼
                                    OutputProcessor
                                          │
                                          ▼
                                      RenderThread
                                          │
                                          ▼
                              FrameCallback / OutputQueue
```

| 线程 | 职责 | 关键类 |
|------|------|--------|
| AudioProducer | 上游推 PCM 数据 | 外部 |
| AudioProcessor | PCM → Mel（7 阶段流水线） | `AudioProcessor` |
| VideoProducer | 上游推视频帧 | 外部 |
| VideoProcessor | 检测 → 对齐 → 遮罩 | `VideoProcessor` |
| AVMatcher | Mel ↔ Face 按 PTS 匹配 | `Pipeline::Impl::MatcherThread` |
| InferenceWorker | Wav2Lip 推理 + 失败重试 | `InferenceWorker` |
| RenderThread | 融合 + 同步 + 帧间隔调节 | `RenderThread` |
| AudioPlayback | PortAudio 音频回调（系统） | `AudioPlayer` |

---

## 3. 目录结构

```
digital-human-sdk/
├── CMakeLists.txt                 # 顶层 CMake 配置（依赖查找 + 子目录）
├── CMakePresets.json              # vcpkg 工具链预设
├── build.sh                       # WSL 构建脚本（含 .so 软链修复）
├── README.md                      # 项目说明
├── Code-Wiki.md                   # 本文档
│
├── include/                       # 公共头文件（PUBLIC）
│   ├── audio/                     # 音频处理模块头文件
│   ├── core/                      # 核心编排模块头文件
│   └── model/                     # 模型模块头文件
│
├── src/                           # 实现源文件（PRIVATE）
│   ├── CMakeLists.txt             # 库构建配置
│   ├── audio/                     # 音频实现
│   ├── core/                     # 核心实现
│   └── model/                     # 模型实现
│
├── examples/                      # 测试与示例程序
│   ├── CMakeLists.txt             # 各测试 target 定义
│   ├── demo.cpp                   # 简单 demo
│   ├── full_pipeline_test.cpp     # 全链路集成测试
│   └── ... (40+ 测试程序)
│
├── assets/                         # 测试资源
│   ├── face.jpg                   # 测试用图像
│   └── diagnose/                  # 诊断输出帧
│
└── .claude/skills/code-style.md   # 代码风格规范
```

---

## 4. 模块职责详解

### 4.1 音频处理模块（audio）

**命名空间**：`digital_human::audio`
**职责**：将原始 PCM 音频转换为模型可识别的 Mel 频谱特征。

#### 模块清单

| 类 | 文件 | 职责 |
|----|------|------|
| `AudioLoader` | `audio_loader.{h,cpp}` | 加载 WAV/MP3 等音频文件为 PCM float |
| `NoiseReduction` | `audio_noise_reduction.{h,cpp}` | 谱减法降噪，参数：noiseFrames=10, oversub=0.02 |
| `AudioFramer` | `audio_framer.{h,cpp}` | 分帧：frame=400samples(25ms), hop=160(10ms) |
| `VoiceActivityDetector` | `audio_vad.{h,cpp}` | VAD：能量阈值 + 过零率 + hangover 平滑 |
| `PreEmphasis` | `audio_preemphasis.{h,cpp}` | 预加重滤波 α=0.97 |
| `RMSNormalize` | `audio_rms_normalize.{h,cpp}` | RMS 归一化 target=0.056 |
| `MelFeatureExtract` | `audio_mel_feature_extract.{h,cpp}` | FFT → Mel 滤波 → Log，输出 80 bins |
| `CMVN` | `audio_cmvn.{h,cpp}` | 倒谱均值方差归一化 |
| `RingBuffer` | `audio_ring_buffer.{h,cpp}` | 无锁 SPSC 环形缓冲区，解耦 PortAudio 回调 |
| `AudioPlayer` | `audio_player.{h,cpp}` | PortAudio 播放器，提供精确 DAC 时钟 |

#### 处理顺序

```cpp
// 完整音频特征提取流水线（位于 AudioProcessor::Run 内）
samples = AudioLoader.load(path).samples;
samples = NoiseReduction.process(samples, sampleRate);
frames  = AudioFramer.frame(samples, {400, 160});
frames  = VoiceActivityDetector.filter(frames);
for (auto& f : frames) {
    f = PreEmphasis.process(f);
    f = RMSNormalize.process(f);
}
mel = MelFeatureExtract.extract(frames, {512, 80, 16000, 0, 8000});
mel = CMVN.process(mel);  // → ncnn::Mat 输入
```

### 4.2 核心编排模块（core）

**命名空间**：`digital_human::core`
**职责**：流水线编排、线程管理、音视频同步、人脸检测对齐。

#### 关键类

| 类 | 文件 | 职责 |
|----|------|------|
| `Pipeline` | `pipeline.{h,cpp}` | 顶层编排器，统一管理 7 个线程 + 7 条队列 |
| `AudioProcessor` | `audio_processor.{h,cpp}` | 音频处理线程，封装 7 阶段流水线 |
| `VideoProcessor` | `video_processor.{h,cpp}` | 视频处理线程：检测→对齐→遮罩 |
| `InferenceWorker` | `inference_worker.{h,cpp}` | 推理线程：失败重试 + 反压 + EWMA 延迟 |
| `RenderThread` | `render_thread.{h,cpp}` | 渲染线程：融合 + 同步 + 帧间隔调节 |
| `FrameScheduler` | `frame_scheduler.{h,cpp}` | 帧调度器，DROP/DUPLICATE/DISPLAY/WAIT 决策 |
| `AVSync` | `av_sync.{h,cpp}` | 音视频同步器（Audio Master Clock） |
| `AudioSyncScheduler` | `audio_sync_scheduler.{h,cpp}` | 组合 AudioPlayer+AVSync+FrameScheduler |
| `FaceDetector` | `face_detector.{h,cpp}` | dlib HOG 人脸检测 + 68 关键点 |
| `FaceAligner` | `face_aligner.{h,cpp}` | 仿射变换对齐到 96×96 |
| `FaceMaskGenerator` | `face_mask_generator.{h,cpp}` | 生成口唇 alpha 遮罩（CV_32FC1） |
| `ImageLoader` | `image_loader.{h,cpp}` | 图像加载（文件/内存/批量） |
| `ThreadBase` | `thread_base.h` | 线程基类：状态机 + 异常捕获 + 超时 Wait |
| `ThreadSafeQueue` | `thread_safe_queue.h` | 模板化线程安全队列（有界/无界 + 心跳） |

#### 数据包定义（packet.h）

```cpp
enum class StatusCode : int8_t {
    OK, ERROR, FATAL, EOS, SKIP, TIMEOUT
};

template <typename T>
struct Packet {
    PacketHeader header;   // pts_ms, seq_id, status, cost_ms
    T            payload;
    // 静态工厂：Make / EOS / Fatal / Skip
};

// 具体类型别名
using AudioRawPacket       = Packet<std::vector<float>>;
using MelFeaturePacket     = Packet<cv::Mat>;
using VideoFramePacket     = Packet<cv::Mat>;
using ProcessedFacePacket  = Packet<ProcessedFaceData>;
using InferenceOutputPacket = Packet<InferenceOutputData>;
using RenderPacket         = Packet<RenderTaskData>;
using OutputFramePacket    = Packet<cv::Mat>;
```

### 4.3 模型模块（model）

**命名空间**：`digital_human::model`
**职责**：Wav2Lip 模型加载、推理、输出后处理。

| 类 | 文件 | 职责 |
|----|------|------|
| `ModelLoader` | `model_loader.{h,cpp}` | 异步加载模型（回调通知 + warmup） |
| `ModelInferencer` | `model_inferencer.{h,cpp}` | Wav2Lip 推理器，支持 auto-tune 与 Vulkan |
| `OutputProcessor` | `output_processor.{h,cpp}` | 输出后处理：格式转换 + 融合 + 锐化 + 色彩 |

#### ModelInferencer 特性

- **单次加载**：`ncnn::Net` 在 `Init()` 时一次性加载，全生命周期复用
- **轻量推理**：`ncnn::Extractor` 每次推理创建
- **自动调优**：遍历线程数 [1, 2, 4, 8, 16] 选最优；若 CPU 延迟超阈值且 ncnn 编译了 Vulkan，自动启用 GPU
- **性能计数**：累计推理次数、平均/最小/最大延迟

#### OutputProcessor 处理管道

```
ncnn::Mat (RGB float, 448×96)
  → OutputToMat         → cv::Mat (BGR uint8, 96×96)
  → InverseTransform    → 原始坐标空间（用 M_inv 逆仿射）
  → FaceFusion          → mask × generated + (1-mask) × original
  → Sharpen             → USM 非锐化（默认 strength=1.0）
  → ColorBlend          → α × generated + (1-α) × original（α=0.7）
```

---

## 5. 关键类与函数说明

### 5.1 Pipeline 类（核心入口）

```cpp
class Pipeline {
public:
    bool Init(const PipelineConfig& config);       // 创建队列与模块
    bool Start();                                    // 启动 7 个线程
    void Stop();                                     // 幂等停止

    bool PushAudio(const std::vector<float>& pcm, int64_t pts_ms);
    bool PushVideo(const cv::Mat& frame, int64_t pts_ms);
    void MarkAudioEOS();
    void MarkVideoEOS();

    bool GetOutputFrame(OutputFramePacket& frame, int timeout_ms = -1);

    void Pause();
    void Resume();
    PipelineMetrics GetMetrics() const;
};
```

**生命周期**：`Init` → `Start` → `PushAudio/PushVideo` → `MarkEOS` → `GetOutputFrame` → `Stop`

**线程安全**：所有公有方法线程安全；内部状态通过队列和原子变量隔离。

### 5.2 ThreadBase 类（线程基础设施）

```cpp
class ThreadBase {
public:
    explicit ThreadBase(std::string name);
    bool Start();                  // INIT → RUNNING
    void Stop();                    // RUNNING → STOPPING（非阻塞）
    bool Wait(int timeout_ms = -1); // 阻塞直到 STOPPED
    void Shutdown();                // Stop + Wait

    bool IsRunning() const;
    bool IsStopping() const;        // Run 循环退出条件
    ThreadState GetState() const;

protected:
    virtual void Run() = 0;        // 子类实现主循环
};
```

**状态机**：

```
INIT ──Start()──► RUNNING ──Stop()──► STOPPING ──Wait()──► STOPPED
                                          │
                                          └──异常──► ERROR
```

**关键设计**：
- 异常被基类捕获并转换为 `ERROR` 状态
- `Wait(timeout)` 通过 `run_exited_` 原子标志轮询，避免 detach 同一 `std::thread` 造成 UB

### 5.3 ThreadSafeQueue 模板

```cpp
template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_capacity = 0,
                             const std::string& name = "queue",
                             size_t heartbeat_timeout_ms = 0);

    bool Push(T&& item);                 // 阻塞入队
    bool TryPush(T&& item);              // 非阻塞入队
    bool WaitAndPop(T& item, int timeout_ms = -1);
    bool TryPop(T& item);
    size_t TryPopBatch(std::vector<T>& items, size_t max_count = 0);

    void Heartbeat();
    bool IsHealthy() const;
    bool CheckDeadlock(size_t stall_timeout_ms) const;

    void Stop();
    QueueMetrics GetMetrics() const;
};
```

**特性**：
- 有界/无界可配置
- `not_empty_cv_` / `not_full_cv_` 双条件变量
- 心跳检测 + 死锁检测
- 移动语义 + Emplace 原位构造
- 完整运行时指标（pushes / pops / overflow / peak）

### 5.4 AVSync 同步器

```cpp
class AVSync {
public:
    void Init(const SyncConfig& config);
    void UpdateAudioClock(int64_t samples_consumed);
    void SetAudioClockMs(double ms);
    double GetAudioClockMs() const;
    SyncResult GetSyncStatus(double video_pts_ms) const;
    SyncResult Sync(int64_t samples_consumed, double video_pts_ms);
};
```

**同步判定逻辑**：

```
drift = video_pts - audio_clock
   |drift| ≥ max_drift   → SEVERE_OFFSET（应丢弃）
   drift >  threshold     → VIDEO_AHEAD（视频超前）
   drift < -threshold    → VIDEO_BEHIND（视频滞后，应丢弃）
   否则                  → SYNCED
```

### 5.5 FrameScheduler 帧调度器

```cpp
class FrameScheduler {
public:
    void Init(const SchedulerConfig& config);
    ScheduleResult ScheduleFrame(int frame_id, double pts_ms);
    void OnFrameDisplayed(double actual_display_time_ms);
    FrameStats GetStats() const;
};
```

**调度动作**：

| 动作 | 触发条件 | 含义 |
|------|---------|------|
| `DISPLAY` | 实际 PTS ≈ 期望 PTS | 正常显示 |
| `DROP` | 实际 PTS 落后 > 半帧间隔 | 丢弃滞后帧 |
| `DUPLICATE` | 实际 PTS 超前 > 半帧间隔 | 重复上一帧填补 |
| `WAIT` | 视频超前需等待音频 | 等待音频追赶 |

通过 EMA 平滑（默认 α=0.5）稳定实际帧率。

### 5.6 ModelInferencer 推理器

```cpp
class ModelInferencer {
public:
    bool Init(const std::string& model_dir);
    bool Init(const std::string& param_path, const std::string& bin_path);
    ncnn::Mat Infer(const ncnn::Mat& audio_feat, const ncnn::Mat& face_input);

    // 自动调优
    int  GetThreadCount() const;
    void SetThreadCount(int n);
    bool IsGPUEnabled() const;
    bool EnableGPU(bool enable);

    // 性能统计
    int64_t GetInferenceCount() const;
    float   GetAvgLatencyMs() const;
    void    PrintStats() const;
};
```

**输入输出**：
- 输入 `audio_feat`：ncnn::Mat，形状 (w=80, h=T, c=1)
- 输入 `face_input`：ncnn::Mat，形状 (w=96, h=96, c=6)
- 输出：ncnn::Mat，3 通道口型同步人脸（RGB float）

### 5.7 OutputProcessor 输出后处理

```cpp
class OutputProcessor {
public:
    cv::Mat OutputToMat(const ncnn::Mat& model_output,
                        int face_w = 96, int face_h = 96);
    cv::Mat InverseTransform(const cv::Mat& processed_face,
                             const cv::Mat& M_inv,
                             const cv::Size& original_size);
    cv::Mat FaceFusion(const cv::Mat& original_image,
                       const cv::Mat& generated_face,
                       const cv::Mat& face_mask);
    cv::Mat Sharpen(const cv::Mat& image, float strength = 1.0f);
    cv::Mat ColorBlend(const cv::Mat& generated, const cv::Mat& original,
                       float alpha = 0.7f);
    cv::Mat Process(const ncnn::Mat& model_output,
                    const cv::Mat& original_face,
                    const cv::Mat& face_mask,
                    const cv::Mat& M_inv);  // 一键全流程
};
```

### 5.8 AudioSyncScheduler 组合调度器

```cpp
class AudioSyncScheduler {
public:
    bool Init(const AudioSyncConfig& config);
    bool LoadAudio(const std::vector<float>& samples, int channels);
    bool Play();
    bool Pause();
    bool Resume();
    bool Stop();

    ScheduleResult ScheduleFrame(int frame_id, double video_pts_ms);
    void OnFrameDisplayed(double actual_display_time_ms);

    double GetDriftMs() const;
    double GetAudioClockMs() const;
    SyncStatus GetSyncStatus() const;
};
```

封装了 `AudioPlayer + AVSync + FrameScheduler` 三件套，适用于离线播放场景。

---

## 6. 数据流与线程模型

### 6.1 七队列架构

Pipeline 内部维护 7 条 `ThreadSafeQueue`，每条都有界（容量见 [§9](#9-配置参数参考)）：

```
[1] audio_raw_queue        : AudioRawPacket        ← Pipeline.PushAudio
[2] video_raw_queue        : VideoFramePacket      ← Pipeline.PushVideo
[3] mel_feature_queue      : MelFeaturePacket      ← AudioProcessor 输出
[4] processed_face_queue   : ProcessedFacePacket   ← VideoProcessor 输出
[5] inference_task_queue  : InferenceTask          ← AVMatcher 输出
[6] inference_output_queue : InferenceOutputPacket ← InferenceWorker 输出
[7] output_frame_queue     : OutputFramePacket     ← RenderThread 输出
```

### 6.2 七线程协作

```
                     ┌──────────────┐
                     │ AudioProducer│ (外部)
                     └──────┬───────┘
                            ▼
                   audio_raw_queue
                            ▼
                  ┌─────────────────┐
                  │ AudioProcessor  │  7阶段流水线
                  └────────┬─────────┘
                           ▼
                  mel_feature_queue ──┐
                                       ├─► ┌─────────────┐
                  processed_face_queue┘    │  AVMatcher  │ (PTS 匹配)
                          ▲                └──────┬──────┘
                  ┌───────┴────────┐             │
                  │ VideoProcessor │             ▼
                  └───────┬────────┘     inference_task_queue
                          ▲                       │
                  ┌───────┴────────┐               ▼
                  │ VideoProducer  │      ┌──────────────────┐
                  └────────────────┘     │ InferenceWorker  │ (重试+反压)
                                         └─────────┬────────┘
                                                   ▼
                                         inference_output_queue
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │ RenderThread │ (融合+同步)
                                          └──────┬───────┘
                                                 ▼
                                         output_frame_queue
                                                 │
                                                 ▼
                                       Pipeline.GetOutputFrame
```

### 6.3 EOS 传播

当外部调用 `MarkAudioEOS()` / `MarkVideoEOS()` 时：

1. Producer 不再推数据
2. AudioProcessor 处理完最后一批后向 `mel_feature_queue` 推 `EOS` 包
3. AVMatcher 收到 EOS 后向 `inference_task_queue` 推 `EOS` 任务
4. InferenceWorker 收到 EOS 后向 `inference_output_queue` 推 `EOS` 包
5. RenderThread 收到 EOS 后排空 `drain_max_frames` 帧再退出

### 6.4 错误传播

任何阶段产生 `FATAL` 包会沿流水线传播，最终触发 Pipeline.Stop()。

---

## 7. 依赖关系

### 7.1 第三方库

| 依赖 | 版本 | 用途 | 查找方式 |
|------|------|------|---------|
| OpenCV | 4.5+ | 图像 IO、矩阵运算、滤波 | `find_package(OpenCV)` |
| ncnn | 20240820 | Wav2Lip-SD-GAN 神经网络推理 | `find_package(ncnn)` |
| FFmpeg | 4.x+ | 音视频编解码（libavformat/codec/util/swresample/swscale） | `pkg_check_modules` |
| PortAudio | 19+ | 音频播放（AudioPlayer） | `pkg_check_modules(portaudio-2.0)` |
| dlib | 19.24+ | HOG 人脸检测 + 68 关键点 | `find_package(dlib)` |
| OpenMP | 4.5+ | 并行化（Mel/图像处理） | `find_package(OpenMP)` |

### 7.2 模块内依赖图

```
audio 模块
  ├── AudioLoader       → FFmpeg (libavformat/codec/util)
  ├── NoiseReduction    → OpenCV (FFT)
  ├── AudioFramer       → (无外部依赖)
  ├── VoiceActivityDetector → (无外部依赖)
  ├── PreEmphasis       → (无外部依赖)
  ├── RMSNormalize      → (无外部依赖)
  ├── MelFeatureExtract → OpenCV (dft + Mel 滤波)
  ├── CMVN              → OpenCV (矩阵归一化)
  ├── RingBuffer        → <atomic> 无锁实现
  └── AudioPlayer       → PortAudio

core 模块
  ├── Pipeline          → AudioProcessor / VideoProcessor / InferenceWorker
  │                      / RenderThread / ModelInferencer / OutputProcessor
  ├── AudioProcessor    → audio 模块全部（除 AudioPlayer）
  ├── VideoProcessor    → FaceDetector / FaceAligner / FaceMaskGenerator
  ├── InferenceWorker   → ModelInferencer
  ├── RenderThread      → OutputProcessor / AudioPlayer / FrameScheduler
  ├── FrameScheduler    → (无外部依赖)
  ├── AVSync            → (无外部依赖)
  ├── FaceDetector      → dlib (face_detection + shape_predictor)
  ├── FaceAligner       → OpenCV (warpAffine)
  ├── FaceMaskGenerator → OpenCV (fillPoly + GaussianBlur)
  ├── ImageLoader       → OpenCV (imread/imdecode)
  ├── ThreadBase        → <thread> <atomic>
  └── ThreadSafeQueue   → <mutex> <condition_variable>

model 模块
  ├── ModelLoader       → ncnn::Net
  ├── ModelInferencer   → ncnn::Net + ncnn::Extractor
  └── OutputProcessor   → OpenCV (cvtColor + warpAffine + addWeighted)
```

### 7.3 CMake 链接关系

```cmake
# src/CMakeLists.txt
add_library(digital_human_core SHARED ${SRC_FILES})

target_include_directories(digital_human_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(digital_human_core PUBLIC
    ${OpenCV_LIBS}
    ncnn
    PkgConfig::FFMPEG
    dlib::dlib
    ${PORTAUDIO_LIBRARIES}
    OpenMP::OpenMP_CXX
)
```

下游 target（如 `full_pipeline_test`）只需 `target_link_libraries(... PRIVATE digital_human_core)`，即可自动获得所有 PUBLIC 链接。

---

## 8. 构建与运行方式

### 8.1 系统要求

| 项目 | 最低要求 |
|------|---------|
| 操作系统 | Ubuntu 22.04 / Windows WSL2 |
| 编译器 | GCC 11+ / Clang 14+ |
| CMake | 3.16+ |
| CPU | 4 核 x86_64（支持 AVX2） |
| 内存 | 8 GB |
| 音频设备 | ALSA 兼容（PortAudio） |
| GPU（可选） | Vulkan 1.1+ |

### 8.2 依赖安装（Ubuntu 22.04）

```bash
# 基础工具
sudo apt install -y build-essential cmake pkg-config git

# OpenCV 4
sudo apt install -y libopencv-dev

# FFmpeg
sudo apt install -y libavformat-dev libavcodec-dev \
    libavutil-dev libswresample-dev libswscale-dev

# PortAudio
sudo apt install -y portaudio19-dev

# dlib
sudo apt install -y libdlib-dev

# ncnn（源码安装，推荐 tag 20240820）
git clone https://github.com/Tencent/ncnn.git
cd ncnn && git checkout 20240820
mkdir build && cd build
cmake .. -DNCNN_VULKAN=OFF \
         -DNCNN_BUILD_EXAMPLES=OFF \
         -DNCNN_BUILD_TOOLS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 8.3 构建命令

#### 完整构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### 仅构建核心库

```bash
make -j$(nproc) digital_human_core
```

#### 使用 WSL + vcpkg 预设

```bash
./build.sh                          # 自动应用 preset 并修复 .so 软链
# 或手动
cmake --preset vcpkg -B build
cmake --build build
```

### 8.4 构建产物

```
build/
├── lib/
│   └── libdigital_human_core.so      # 核心共享库
└── bin/
    ├── full_pipeline_test            # 全链路集成测试（162 测试）
    ├── inference_render_pipeline_test # 推理+渲染联调（21 测试）
    ├── pipeline_test                 # 线程基础设施（76 测试）
    ├── audio_processor_test          # 音频处理器（23 测试）
    ├── inference_worker_test         # 推理线程（45 测试）
    ├── render_thread_test            # 渲染线程（28 测试）
    ├── frame_scheduler_test          # 帧调度器（62 测试）
    ├── audio_sync_test               # 音频同步（37 测试）
    └── ... 各模块独立测试
```

### 8.5 运行示例

#### Pipeline 完整调用

```cpp
#include "core/pipeline.h"
using namespace digital_human::core;

PipelineConfig config;
config.audio_sample_rate = 16000;
config.target_fps        = 25.0;

Pipeline pipeline;
pipeline.Init(config);
pipeline.Start();

// 输入数据
pipeline.PushAudio(pcm_data, pts_ms);
pipeline.PushVideo(frame, pts_ms);

// 标记 EOS
pipeline.MarkAudioEOS();
pipeline.MarkVideoEOS();

// 获取输出
OutputFramePacket result;
while (pipeline.GetOutputFrame(result, 1000)) {
    cv::imshow("Output", result.payload);
    cv::waitKey(1);
}

pipeline.Stop();
```

#### 模型推理最小示例

```cpp
#include "model/model_inferencer.h"
using namespace digital_human::model;

ModelInferencer model;
if (!model.Init("models/Wav2Lip-SD-GAN-opt")) {
    return -1;  // 加载失败
}

ncnn::Mat audio_feat = /* Mel 特征 (80×T×1) */;
ncnn::Mat face_input = /* 对齐人脸 (96×96×6) */;

ncnn::Mat output = model.Infer(audio_feat, face_input);
model.PrintStats();
```

### 8.6 常见问题

| 问题 | 解决方案 |
|------|---------|
| `ncnn 找不到` | `export CMAKE_PREFIX_PATH=/usr/local/lib/cmake:$CMAKE_PREFIX_PATH` |
| OpenCV 版本不匹配 | Ubuntu 22.04 自带 4.5.4，向下兼容 |
| dlib 警告 `dlib_INCLUDE_DIRS` 已弃用 | 不影响编译，可忽略 |
| PortAudio "No audio devices" | WSL2 默认无音频设备，不影响功能测试；生产环境需 ALSA |
| 推理线程一直重试 | 确认 `models/` 下有 `.param` 和 `.bin` 文件 |
| 渲染帧率不稳定 | 调整 `RenderConfig::target_fps`，确保 `enable_frame_pacing = true` |

---

## 9. 配置参数参考

### 9.1 PipelineConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `audio_sample_rate` | 16000 | 音频采样率（Hz） |
| `audio_channels` | 1 | 声道数 |
| `audio_frame_size` | 400 | 帧长（samples @16kHz=25ms） |
| `audio_hop_size` | 160 | 帧移（samples @16kHz=10ms） |
| `target_fps` | 25.0 | 目标帧率 |
| `face_size` | 96 | 对齐人脸尺寸 |
| `sync_threshold_ms` | 30.0 | 同步阈值 |
| `max_drift_ms` | 100.0 | 最大允许漂移 |
| `av_match_threshold_ms` | 40.0 | 音视频匹配阈值 |
| `audio_raw_queue_size` | 30 | 音频原始数据队列容量 |
| `mel_queue_size` | 60 | Mel 特征队列容量 |
| `video_raw_queue_size` | 30 | 视频原始帧队列容量 |
| `face_queue_size` | 30 | 处理后人脸队列容量 |
| `infer_queue_size` | 30 | 推理输出队列容量 |
| `output_queue_size` | 10 | 最终输出队列容量 |
| `pop_timeout_ms` | 100 | 队列弹出超时 |
| `shutdown_timeout_ms` | 2000 | 关闭超时 |

### 9.2 AudioProcessorConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `sample_rate` | 16000 | 采样率 |
| `channels` | 1 | 声道数 |
| `frame_size` | 400 | 帧长 |
| `hop_size` | 160 | 帧移 |
| `mel_bins` | 80 | Mel 滤波器组数 |
| `nfft` | 512 | FFT 点数 |
| `window_capacity` | 4800 | 滑动窗口容量（300ms） |

提供 `AutoConfigure(target_sample_rate)` 方法自适应计算帧参数。

### 9.3 InferenceWorkerConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `input_queue_warn_threshold` | 10 | 队列深度警告阈值 |
| `input_queue_error_threshold` | 30 | 队列深度错误阈值 |
| `max_retries` | 3 | 推理失败最大重试次数 |
| `pop_timeout_ms` | 100 | 队列弹出超时 |
| `backlog_check_interval` | 1000 | 积压检测间隔（ms） |
| `latency_warn_threshold_ms` | 100.0f | 延迟警告阈值 |

### 9.4 RenderConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `target_fps` | 25.0 | 目标帧率 |
| `sync_threshold_ms` | 30.0 | 同步阈值 |
| `max_drift_ms` | 100.0 | 最大允许漂移 |
| `render_queue_warn` | 5 | 渲染队列深度警告阈值 |
| `drain_max_frames` | 30 | 退出时排空帧数 |
| `pop_timeout_ms` | 100 | 队列弹出超时 |
| `enable_frame_pacing` | true | 启用帧间隔调节 |
| `enable_audio_sync` | true | 启用音频同步 |
| `enable_display` | true | 是否显示窗口 |

### 9.5 SyncConfig / SchedulerConfig

```cpp
struct SyncConfig {
    int    audio_sample_rate = 16000;
    double sync_threshold_ms = 30.0;
    double max_drift_ms      = 100.0;
};

struct SchedulerConfig {
    double target_fps        = 25.0;
    int    max_pending_frames = 10;
    double smoothing_factor  = 0.5;   // EMA α
    bool   enable_smoothing  = true;
};
```

---

## 10. 测试体系

### 10.1 测试金字塔

| 层级 | 覆盖范围 | 命令 | 用例数 |
|------|---------|------|--------|
| Level 0 | 模块独立测试（无需模型） | 各 `*_test.cpp` | - |
| Level 1 | 线程基础设施（队列/线程基类） | `./bin/pipeline_test` | 76 |
| Level 2 | 音频/推理/渲染线程 | `./bin/audio_processor_test` 等 | 23~62 |
| Level 3 | 推理+渲染联调 | `./bin/inference_render_pipeline_test` | 21 |
| Level 4 | 全链路端到端 | `./bin/full_pipeline_test` | 162 |

### 10.2 关键测试 target

| 测试程序 | 验证目标 |
|---------|---------|
| `full_pipeline_test` | 端到端流水线（无需 Wav2Lip 模型） |
| `inference_render_pipeline_test` | 推理 + 渲染联调 |
| `pipeline_test` | ThreadSafeQueue / ThreadBase 基础设施 |
| `audio_processor_test` | 7 阶段音频特征提取 |
| `inference_worker_test` | 推理重试 + 反压 + EWMA 延迟 |
| `render_thread_test` | 渲染 + 音视频同步 |
| `frame_scheduler_test` | DROP/DUPLICATE/DISPLAY 决策 |
| `audio_sync_test` | AudioSyncScheduler 全链路 |
| `bugfix_verification_test` | V1 审查问题修复验证 |
| `bugfix_v2_test` | V2 审查问题修复验证 |
| `v1_review_fixes_test` | 7 项主要问题逐条验证 |
| `lip_sync_diagnose_test` | 验证模型输出随音频变化 |

### 10.3 全量回归脚本

```bash
cd build
tests=(
    full_pipeline_test
    inference_render_pipeline_test
    pipeline_test
    audio_processor_test
    inference_worker_test
    render_thread_test
    frame_scheduler_test
    audio_sync_test
)
for t in "${tests[@]}"; do
    echo "=== $t ==="
    ./bin/$t 2>/dev/null
done
```

### 10.4 调试命令

```bash
# Valgrind 内存检测
valgrind --leak-check=full ./bin/full_pipeline_test 2>/dev/null

# 指定模型目录
export MODEL_DIR=/path/to/models
```

---

## 11. 编码规范与设计模式

### 11.1 Pimpl 模式

所有公有类采用 Pimpl（Pointer to Implementation）模式，符合以下约定：

**头文件侧**：
```cpp
class MyClass {
public:
    MyClass();
    ~MyClass();
    MyClass(const MyClass&) = delete;            // 禁拷贝
    MyClass& operator=(const MyClass&) = delete;
    MyClass(MyClass&&) noexcept;                  // 允许移动
    MyClass& operator=(MyClass&&) noexcept;

    // 公有接口...

private:
    struct Impl;                                  // 前向声明
    std::unique_ptr<Impl> impl_;                   // 唯一指针
};
```

**实现文件侧**：
```cpp
struct MyClass::Impl {
    // 真实成员
    int value;
    std::string name;
};

MyClass::MyClass() : impl_(std::make_unique<Impl>()) {}
MyClass::~MyClass() = default;
MyClass::MyClass(MyClass&&) noexcept = default;
MyClass& MyClass::operator=(MyClass&&) noexcept = default;

// 方法实现 → 转发到 impl_
```

**收益**：
- 减少 .h 文件对第三方头文件的依赖（编译时间↓）
- 隐藏实现细节（ABI 稳定）
- 强制 RAII 与异常安全

### 11.2 命名空间约定

- **根命名空间**：`digital_human`
- **模块子命名空间**：`digital_human::audio` / `digital_human::core` / `digital_human::model`
- **模块名**：snake_case
- **历史遗留**：`sdk_entry.cpp` 中存在 `DigitalHuman::Initsdk()` 旧式 API，待清理

### 11.3 注释规范

- 所有公有 API 使用 **中文 Doxygen** 注释
- 复杂逻辑添加 `// ============` 分段注释
- 头文件包含模块概览注释（参见 `pipeline.h`）

### 11.4 CMake 规范

- **target 级配置**：避免全局 `include_directories` / `link_directories` / `add_compile_options`
- **PUBLIC vs PRIVATE**：
  - `PUBLIC`：下游 target 自动继承（OpenCV、ncnn 等核心依赖）
  - `PRIVATE`：仅本 target 使用（PORTAUDIO_INCLUDE_DIRS）
- **CONFIGURE_DEPENDS**：`file(GLOB_RECURSE ... CONFIGURE_DEPENDS ...)` 自动感知源文件增删
- **按构建类型选择编译选项**：
  ```cmake
  target_compile_options(digital_human_core PRIVATE
      $<$<CONFIG:Debug>:-O0;-g;-DDEBUG>
      $<$<CONFIG:Release>:-O2;-DNDEBUG>
      $<$<CONFIG:RelWithDebInfo>:-O2;-g;-DNDEBUG>
      $<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>
      -Wall -Wextra
  )
  ```
- **不使用 `-march=native`** 以保证二进制可移植性

### 11.5 线程安全约定

- 所有共享状态通过 `std::atomic` 或 `ThreadSafeQueue` 隔离
- `ThreadBase` 强制子类实现 `Run()`，并通过 `IsStopping()` 协作退出
- 队列提供 `Heartbeat()` / `CheckDeadlock()` 机制检测停滞
- `ThreadBase::Wait(timeout)` 通过 `run_exited_` 原子标志轮询，避免 `std::thread` 并发操作 UB

### 11.6 错误处理约定

- 公有 API 返回 `bool` 表示成功/失败
- 内部错误通过 `Packet<T>::Fatal()` 沿流水线传播
- `ThreadBase` 捕获未处理异常并转为 `ThreadState::ERROR`
- 资源释放通过 RAII 保证（`std::unique_ptr` 全程使用）

### 11.7 资源管理

- `ncnn::Net` 在 `ModelInferencer::Init` 一次性加载，全生命周期复用
- `ncnn::Extractor` 每次推理创建（轻量级）
- `cv::Mat` 通过移动语义避免拷贝
- Pipeline 退出时按上游→下游顺序 `Stop() + Wait()`

---

## 附录：参考文档

| 资源 | 链接 |
|------|------|
| 项目 README | [README.md](file:///c:/Users/27013/Desktop/digital-human-sdk/README.md) |
| 代码审查报告 | [Digital Human SDK 项目代码审查报告.md](file:///c:/Users/27013/Desktop/digital-human-sdk/Digital%20Human%20SDK%20项目代码审查报告.md) |
| 顶层 CMake | [CMakeLists.txt](file:///c:/Users/27013/Desktop/digital-human-sdk/CMakeLists.txt) |
| 源 CMake | [src/CMakeLists.txt](file:///c:/Users/27013/Desktop/digital-human-sdk/src/CMakeLists.txt) |
| 测试 CMake | [examples/CMakeLists.txt](file:///c:/Users/27013/Desktop/digital-human-sdk/examples/CMakeLists.txt) |
| 代码风格 | [.claude/skills/code-style.md](file:///c:/Users/27013/Desktop/digital-human-sdk/.claude/skills/code-style.md) |
| Pipeline 接口 | [include/core/pipeline.h](file:///c:/Users/27013/Desktop/digital-human-sdk/include/core/pipeline.h) |
| 数据包定义 | [include/core/packet.h](file:///c:/Users/27013/Desktop/digital-human-sdk/include/core/packet.h) |
| 线程基类 | [include/core/thread_base.h](file:///c:/Users/27013/Desktop/digital-human-sdk/include/core/thread_base.h) |
| 线程安全队列 | [include/core/thread_safe_queue.h](file:///c:/Users/27013/Desktop/digital-human-sdk/include/core/thread_safe_queue.h) |

---

*本文档由代码静态分析自动生成，反映 `master` 分支截至 2026-07-20 的代码状态。*
