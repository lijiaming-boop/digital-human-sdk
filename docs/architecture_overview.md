# Digital Human SDK 架构设计与优化方向

## 一、项目概述

Digital Human SDK 是一个基于 **Wav2Lip-SD-GAN** 的 C++ 数字人口型同步引擎。输入一段音频和一张人脸图像，输出口型与音频同步的数字人视频。项目在 **WSL2 + Ubuntu 22.04** 环境下开发，目标平台为 Linux x86_64。

### 核心技术栈

| 技术 | 用途 |
|------|------|
| C++17 | 开发语言 |
| CMake 3.16+ | 构建系统 |
| OpenCV 4.5.4+ | 图像处理、人脸检测、矩阵运算 |
| ncnn 20240820 | 神经网络推理引擎（腾讯） |
| FFmpeg | 音频加载/解码、视频编码 |
| PortAudio 19+ | 实时音频播放 |
| dlib | 68 点人脸关键点检测 |
| OpenMP | 并行计算加速 |

---

## 二、整体架构设计

### 2.1 架构层次

系统分为三层：

```
┌─────────────────────────────────────────────┐
│              应用层 (examples/)               │
│   demo / full_pipeline_test / 各模块单元测试   │
├─────────────────────────────────────────────┤
│               SDK 核心层 (src/)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │  Audio   │ │  Core    │ │  Model   │     │
│  │  模块    │ │  模块    │ │  模块    │     │
│  └──────────┘ └──────────┘ └──────────┘     │
├─────────────────────────────────────────────┤
│              公共 API 层 (include/)           │
│     audio/    core/    model/    video/       │
├─────────────────────────────────────────────┤
│           基础设施 / 第三方依赖                 │
│  OpenCV  ncnn  FFmpeg  PortAudio  dlib  OMP  │
└─────────────────────────────────────────────┘
```

### 2.2 多线程流水线架构（核心设计）

系统采用 **7 线程协调流水线** 模型，以音频时钟为主时钟实现音视频同步：

```
                     ┌──────────┐
 AudioSrc ──────────►│  Audio   │──► AudioRawQueue
 (文件/麦克风)        │  Loader  │
                     └──────────┘       │
                                        ▼
                               ┌──────────────────┐
                               │  AudioProcessor   │──► MelFeatureQueue
                               │  (VAD→Framer→Mel) │
                               └──────────────────┘       │
                                                          ▼
                     ┌──────────┐                ┌────────────────┐
 VideoSrc ──────────►│  Video   │──► VideoRaw───►│  Inference     │──► InferOutputQueue
 (图片/摄像头)       │  Loader  │      Queue     │  Worker        │
                     └──────────┘       │        │  (Wav2Lip推理)  │
                                        ▼        └────────────────┘
                               ┌──────────────────┐       │
                               │  VideoProcessor   │       │
                               │ (Detect→Align→    │       │
                               │  MaskGen)         │       │
                               └──────────────────┘       │
                                                          ▼
                                                ┌──────────────────┐
                                                │ OutputProcessor   │──► OutputFrameQueue
                                                │ (后处理+融合)      │
                                                └──────────────────┘       │
                                                                          ▼
                                                                 ┌────────────────┐
                                                                 │  RenderThread   │──► FrameCallback
                                                                 │ (同步+渲染+调速) │
                                                                 └────────────────┘
                                                                          │
                                                                  ┌───────┴───────┐
                                                                  ▼               ▼
                                                           视频编码/文件    屏幕显示/回调
```

#### 线程职责表

| 线程 | 名称 | 输入队列 | 输出队列 | 优先级 |
|------|------|---------|---------|--------|
| T1 | AudioProducer | 音频文件/设备 | AudioRawQueue | 高 |
| T2 | AudioProcessor | AudioRawQueue | MelFeatureQueue | 中 |
| T3 | VideoProducer | 图片文件/设备 | VideoRawQueue | 高 |
| T4 | VideoProcessor | VideoRawQueue | ProcessedFaceQueue | 中 |
| T5 | InferenceWorker | MelFeatureQueue + ProcessedFaceQueue | InferenceOutputQueue | 中 |
| T6 | OutputProcessor | InferenceOutputQueue | OutputFrameQueue | 中 |
| T7 | RenderThread | OutputFrameQueue | FrameCallback | 高 |
| — | AudioPlayback | RingBuffer | 音频硬件 | 实时 |

### 2.3 设计模式

- **PIMPL 模式**：所有公共类使用 `unique_ptr<Impl>` 隐藏实现，保证 ABI 稳定性和编译封装
- **生产者-消费者**：线程间通过 `ThreadSafeQueue<T>` 传递带时间戳的数据包 `Packet<T>`
- **音频主时钟同步**：`AVSync` 计算音视频漂移，`FrameScheduler` 做出 DISPLAY/DROP/DUPLICATE/WAIT 决策
- **状态机**：`ThreadBase` 定义 INIT→RUNNING→STOPPING→STOPPED→ERROR 生命周期

### 2.4 线程安全模型

- `ThreadSafeQueue<T>`：基于 `mutex` + `condition_variable`，支持有界/无界模式，带心跳和死锁检测
- 单一锁层级：每个线程最多持有一把锁，避免嵌套锁和死锁
- 原子变量：用于度量、标志位和共享状态
- 无锁 RingBuffer：音频播放使用 SPSC（单生产者单消费者）无锁环形缓冲区

---

## 三、功能模块详解

### 3.1 音频处理模块 (`src/audio/`)

| 组件 | 功能 |
|------|------|
| AudioLoader | 基于 FFmpeg 加载音频文件 → PCM float 采样 |
| NoiseReduction | 谱减法降噪 |
| AudioFramer | 将 PCM 切分为重叠帧（400/160 样本） |
| VoiceActivityDetector | 基于能量+过零率的语音活动检测 |
| PreEmphasis | 预加重滤波 `y[n]=x[n]-α·x[n-1]` |
| RMSNormalize | RMS 归一化到目标电平 |
| MelFeatureExtract | FFT → Mel 滤波器组 → Log → Mel 频谱图 |
| CMVN | 倒谱均值方差归一化 |
| RingBuffer | 无锁 SPSC 环形缓冲区（PortAudio→处理线程） |
| AudioPlayer | PortAudio 实时播放 + DAC 时钟查询 |

### 3.2 核心模块 (`src/core/`)

| 组件 | 功能 |
|------|------|
| Pipeline | 流水线总控：线程编排、配置管理、生命周期、度量收集 |
| ThreadBase | 抽象线程基类：状态机 + 生命周期管理 |
| ThreadSafeQueue | 线程安全队列：有界/无界、心跳、超时 |
| Packet | 类型安全数据包：PTS、序列号、状态码、耗时 |
| ImageLoader | 图像加载：文件/内存/批量导入 |
| FaceDetector | 人脸检测：dlib HOG + 68 点关键点 |
| FaceAligner | 人脸对齐：仿射变换 → 96×96 标准脸 |
| FaceMaskGenerator | 口唇遮罩生成：精确 alpha 遮罩 96×96 |
| AVSync | 音视频漂移计算、同步状态判定 |
| FrameScheduler | 帧调度决策：DISPLAY/DROP/DUPLICATE/WAIT + EMA 平滑 |
| AudioSyncScheduler | 音频同步调度器：整合 AudioPlayer + AVSync + FrameScheduler |
| AudioProcessor | 音频特征提取线程编排 |
| InferenceWorker | 推理线程：Wav2Lip 推理、重试、积压检测 |
| RenderThread | 渲染线程：融合、同步、帧率控制、回调 |

### 3.3 模型模块 (`src/model/`)

| 组件 | 功能 |
|------|------|
| ModelLoader | 异步模型加载（.param/.bin）+ warmup |
| ModelInferencer | Wav2Lip-SD-GAN 推理：线程数自动调优、Vulkan GPU 支持 |
| OutputProcessor | 后处理：ncnn→cv::Mat、逆变换、人脸融合、锐化、色彩混合 |

### 3.4 待实现模块

| 模块 | 状态 | 说明 |
|------|------|------|
| `src/video/` | 空目录 | 视频编码/解码、视频文件输出 |
| `src/utils/` | 空目录 | 通用工具函数 |

---

## 四、优化方向

### 4.1 性能优化

| 方向 | 现状 | 优化建议 |
|------|------|---------|
| **GPU 推理加速** | ncnn Vulkan 支持已编写但未充分测试 | 启用 `-DNCNN_VULKAN=ON`，利用 GPU 加速 Wav2Lip 推理（当前 CPU 单帧 ~75ms） |
| **推理流水线并行** | InferenceWorker 单实例串行推理 | 引入多实例推理池 + 帧序重排，将推理吞吐提升 2-3x |
| **SIMD 指令优化** | 依赖编译器自动向量化 | 在 Mel 特征提取、图像融合等热点手写 SSE/AVX2  intrinsics |
| **内存池化** | 频繁 `new/delete` Packet 和 cv::Mat | 引入 `PacketPool` / `MatPool` 对象池，减少内存分配开销 |
| **零拷贝通道** | 线程间队列传递数据存在拷贝 | 音频帧和图像帧使用共享指针或 ring buffer 引用传递 |
| **模型量化** | 使用 FP32 推理 | 将 Wav2Lip 模型量化到 FP16/INT8，推理速度提升 2-4 倍 |

### 4.2 架构优化

| 方向 | 现状 | 优化建议 |
|------|------|---------|
| **配置系统** | 各模块独立 `Config` 结构体，硬编码默认值 | 引入统一 JSON/YAML 配置加载，支持运行时热更新 |
| **插件化模块** | 模块间编译期硬链接 | 定义模块接口抽象，支持运行时动态加载（如替换不同推理后端） |
| **错误处理** | Pipeline 错误处理较为基础 | 引入分级错误恢复机制：模块级重试 → 线程级重启 → 流水线降级 |
| **观测性** | 基础 Metrics 收集 | 接入 Prometheus / OpenTelemetry 指标暴露，支持实时监控面板 |
| **异步日志** | 直接使用 printf/cout | 引入 spdlog 异步日志，生产级日志轮转、等级过滤 |

### 4.3 功能扩展

| 方向 | 建议 |
|------|------|
| **视频输入** | 支持 MP4 视频文件作为输入（抽取音频流 + 视频帧序列） |
| **视频输出** | 实现 FFmpeg `VideoEncoder` 模块，直接输出 MP4 文件 |
| **实时流输入** | 支持 RTMP/RTSP 网络流输入，实现直播级数字人 |
| **多人脸支持** | 扩展 FaceDetector 支持多人脸场景，可指定目标人脸 |
| **背景替换** | 集成绿幕/人像分割，支持自定义背景 |
| **TTS 集成** | 内置 TTS 引擎接口，文本直接驱动数字人 |
| **批处理模式** | 支持批量处理脚本，命令行一次性传入多组音视频对 |

### 4.4 工程化优化

| 方向 | 建议 |
|------|------|
| **CI/CD** | 配置 GitHub Actions 自动化构建 + 测试 + 代码检查 |
| **单元测试增强** | 当前 205 个测试覆盖主要模块，增加 Pipeline 集成测试异常场景 |
| **性能基准测试** | 引入 Google Benchmark 对核心热点做回归性能测试 |
| **文档自动化** | 使用 Doxygen 从代码注释生成 API 文档 |
| **ABI 兼容性** | 完善导出符号控制，提供稳定 C 语言 API 层 |
| **Docker 镜像** | 提供预构建 Docker 开发/运行镜像，降低环境配置成本 |

### 4.5 同步算法优化

| 方向 | 现状 | 优化建议 |
|------|------|---------|
| **自适应同步阈值** | `sync_threshold_ms` 固定值 | 根据历史漂移量和帧率动态调整阈值 |
| **丢帧策略** | 简单丢弃或等待 | 智能丢帧：根据音频内容重要性（VAD 结果）决定是否丢帧 |
| **音频时钟漂移补偿** | 未考虑音频设备时钟漂移 | 引入时钟漂移估计算法，周期性校正主时钟 |
| **预卷缓冲** | 固定预卷帧数 | 动态预卷：根据当前系统负载自动调整缓冲深度 |

---

## 五、数据流全景

```
输入音频 ──→ AudioLoader ──→ NoiseReduction
    │                           │
    │                           ▼
    │                   VoiceActivityDetector
    │                           │
    │                           ▼
    │                   PreEmphasis → RMSNormalize
    │                           │
    │                           ▼
    │                   AudioFramer → MelFeatureExtract
    │                           │
    │                           ▼
    │                       CMVN → MelFeatureQueue
    │                                   │
输入图片 ──→ ImageLoader ──→ FaceDetector
    │                           │
    │                           ▼
    │                   FaceAligner (96×96)
    │                           │
    │                           ▼
    │                   FaceMaskGenerator
    │                           │
    │                           ▼
    │                   ProcessedFaceQueue
    │                           │
    └───────────────┬───────────┘
                    ▼
           InferenceWorker (Wav2Lip)
                    │
                    ▼
           OutputProcessor (融合+锐化)
                    │
                    ▼
           RenderThread (AVSync + FrameScheduler)
                    │
              ┌─────┴─────┐
              ▼           ▼
           FrameCallback  视频编码 (待实现)
```

---

## 六、构建关系

```
CMakeLists.txt (根)
  ├── src/CMakeLists.txt ──→ libdigital_human_core.so
  │     ├── src/audio/*.cpp
  │     ├── src/core/*.cpp
  │     ├── src/model/*.cpp
  │     └── (预留) src/video/*.cpp, src/utils/*.cpp
  │
  └── examples/CMakeLists.txt ──→ 38 个测试可执行文件
        ├── full_pipeline_test
        ├── inference_render_pipeline_test
        ├── pipeline_test
        ├── audio_processor_test
        ├── inference_worker_test
        ├── render_thread_test
        ├── frame_scheduler_test
        ├── audio_sync_test
        └── ... (30 个模块单元测试)
```
